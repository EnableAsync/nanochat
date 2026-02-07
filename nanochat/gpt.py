"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    # 这些超参数由 --depth 旋钮统一缩放（见 train_gpt.py 中的 depth_to_config）
    # depth 越大 → n_layer / n_head / n_embd 同步增大，保持模型各维度均衡
    sequence_len: int = 2048       # 训练序列长度，也决定滑动窗口的尺寸
    vocab_size: int = 32768        # 词表大小（实际会在 GPT.__init__ 中 pad 到 64 的倍数）
    n_layer: int = 12              # Transformer 层数
    n_head: int = 6                # Query 头数（决定 head_dim = n_embd / n_head）
    n_kv_head: int = 6             # KV 头数；< n_head 时启用 GQA（Grouped-Query Attention），推理更省显存
    n_embd: int = 768              # 隐藏维度
    # 滑动窗口注意力模式串，按层循环铺开。最后一层强制为 L（全上下文）
    # L=长窗口（full context）, S=短窗口（half context）
    # 例："SSSL" → 前三层用半上下文窗口，第四层用全上下文，如此循环
    # 短窗口层降低 O(T²) 注意力开销，长窗口层保留远距离依赖
    window_pattern: str = "SSSL"


def norm(x):
    # 无参数 RMSNorm：只做 x / rms(x)，不带可学习的 γ/β
    # 为什么够用：模型已有足够的线性层来学习缩放，省掉 LayerNorm 的额外参数和计算
    # 相比 LayerNorm 还省去了均值中心化步骤，更快
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """隔层启用 Value Embedding（ResFormer 风格），最后一层始终有。
    通过奇偶对齐保证最后一层 has_ve=True，中间层交替省显存。"""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # RoPE 旋转位置编码的几何直觉：
    # 把 head_dim 拆成 d 对 (x1, x2)，每对视为二维平面上的向量
    # 对每一对施加角度为 θ·pos 的旋转矩阵 [[cos, -sin], [sin, cos]]
    # 两个 token 的点积只依赖它们的相对位置差 → 自带相对位置编码
    # 低维对转得快（高频），高维对转得慢（低频）→ 类似傅里叶基
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 把最后一维拆成两半
    y1 = x1 * cos + x2 * sin        # 2D 旋转的展开式
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head      # GQA: 多个 Q 头共享一组 KV 头，减少 KV 缓存大小
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Q/K/V 投影，无 bias（现代 LLM 标准做法，减少参数，效果不损）
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # 输出投影，初始化为零 → 初始时注意力层不做事
        # Value Embedding gate：只取输入前 32 维来计算逐头 gate
        # 零初始化 → sigmoid(0)=0.5 * 2 = 1.0 → 初始时 VE 以 1:1 混入 V
        # 用少量通道做 gate 既省计算，又提供足够的输入依赖性
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()  # B=batch, T=序列长度, C=n_embd

        # 投影得到 Q/K/V，直接 reshape 为 (B, T, H, D) —— Flash Attention 的原生布局，无需转置
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)       # (B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)    # (B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)    # (B, T, n_kv_head, head_dim)

        # Value Embedding 残差（ResFormer）：用输入依赖的 gate 将 token embedding 混入 V
        # 动机：让深层也能直接访问原始 token 信息，缓解信息遗忘
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # gate 范围 (0, 2)：< 1 时削弱 VE，> 1 时增强；初始值 1.0（中性）
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve  # 将 VE 按 gate 混入 V

        # RoPE：对 Q/K 施加旋转位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK Norm：防止注意力 logits 爆炸，替代传统的 1/√d 缩放

        # 注意力计算（FA3 on Hopper, PyTorch SDPA fallback）
        # window_size = (left, 0)：left>0 为滑动窗口，left=-1 为全上下文
        if kv_cache is None:
            # 训练路径：因果注意力 + 可选滑动窗口
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推理路径：使用 KV cache 避免重复计算历史 token 的 K/V
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 最后一层处理完后才推进 cache 位置，确保所有层看到一致的位置
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 合并多头 → 输出投影回残差流
        y = y.contiguous().view(B, T, -1)  # (B, T, n_embd)
        y = self.c_proj(y)  # 初始化为零 → 训练开始时注意力对残差无贡献
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)    # 上投影，扩展 4 倍
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)   # 下投影，零初始化

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² 激活：比 GELU 更稀疏（~50% 神经元为零），且平方使梯度更平滑
        x = self.c_proj(x)      # 省掉 SwiGLU 的额外 gate 投影，参数更少
        return x


class Block(nn.Module):
    """Pre-Norm 残差结构：先 norm 再进子层，残差连接加回去。
    这是现代 LLM 的标准结构（GPT-2 风格），比 Post-Norm 更稳定。"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 注意 norm(x) 作用于子层输入，而 x 本身保留在残差路径中
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # 注意力子层
        x = x + self.mlp(norm(x))                                       # MLP 子层
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        注意：此 __init__ 在 meta device 上下文中运行（!!）
        因此这里只做形状/类型定义，不涉及实际数据。
        所有参数数据（权重、buffer 等）在 init_weights() 中真正初始化。
        """
        super().__init__()
        self.config = config
        # 计算每层的滑动窗口大小
        self.window_sizes = self._compute_window_sizes(config)
        # 将词表大小 pad 到 64 的倍数 → 对齐 DDP bucket 和 Tensor Core（16/64 对齐优化矩阵乘）
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # token embedding（不与 lm_head 共享）
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)  # 独立的 unembedding 层
        # 逐层可学习缩放标量（灵感来自 modded-nanogpt）
        # resid_lambdas: 缩放残差流（初始 1.0 = 标准残差连接）
        # x0_lambdas: 混入初始 embedding（初始 0.1 = 微弱的跳连到输入）
        # 让模型自适应学习每层的残差强度
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # meta 占位，真正初始化在 init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # meta 占位
        # Value Embedding（ResFormer 风格）：隔层放置以平衡效果和显存
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim  # VE 维度与 KV 一致，因为它混入 V
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # RoPE buffer：预计算 10 倍序列长度的旋转矩阵（内存开销很小，避免动态扩展）
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False → 不保存到 checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        集中初始化所有权重，清晰明确。

        初始化策略及原因：
        - wte (embedding):     Normal(0, 1)    — 标准嵌入初始化
        - lm_head:             Normal(0, 0.001) — 极小值初始化使初始输出接近均匀分布
        - attn.c_q/c_k/c_v:   Uniform          — 均匀分布避免正态的尾部离群值
        - attn.c_proj:         零初始化         — 训练开始时注意力子层输出为零，残差直通
        - mlp.c_fc:            Uniform          — 同上
        - mlp.c_proj:          零初始化         — MLP 子层输出也为零，残差直通
        这种"输出投影零初始化"让深层网络初始时等价于恒等映射 → 训练更稳定
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        # Uniform[-s, s] 的标准差 = s/√3，因此 s = √3 * target_std 才能匹配 Normal 的 std
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # √3 / √d_model
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # 用 Uniform 避免正态分布的尾部离群值
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # 零初始化 → 残差直通
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)  # 零初始化 → 残差直通

        # 逐层缩放标量
        self.resid_lambdas.fill_(1.0)   # 1.0 → 标准残差连接
        self.x0_lambdas.fill_(0.1)      # 0.1 → 初始时从输入 embedding 微弱引入信息

        # Value Embedding 初始化（与 c_v 相同的均匀分布）
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE Gate 零初始化 → sigmoid(0) = 0.5，乘以 2 = 1.0（中性初始 gate 值）
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Embedding 转 bf16：优化器对 embedding 的精度要求不高，bf16 省一半显存
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # 预计算 RoPE 的 cos/sin 表
        # base=10000 控制频率范围：base 越大 → 低频更多 → 能编码更长距离
        if device is None:
            device = self.transformer.wte.weight.device
        # 每两个通道一个频率：θ_i = 1 / base^(2i / d)
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 位置序列
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 外积得到 (seq_len, head_dim/2) 的频率表
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # 添加 batch 和 head 维度以便后续广播: (1, seq_len, 1, head_dim/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """参数分组策略：不同类型参数用不同优化器和学习率。
        - Embedding/Scalar → AdamW（这些参数不适合 Muon 的正交化）
        - Transformer 矩阵参数 → Muon（利用正交化加速收敛）
        - LR 按 1/√(d_model/768) 缩放（µP 风格的宽度缩放）
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # 将参数分为 6 组
        matrix_params = list(self.transformer.h.parameters())       # 注意力 + MLP 的矩阵参数 → Muon
        value_embeds_params = list(self.value_embeds.parameters())   # VE embedding → AdamW
        embedding_params = list(self.transformer.wte.parameters())   # token embedding → AdamW
        lm_head_params = list(self.lm_head.parameters())             # unembedding → AdamW
        resid_params = [self.resid_lambdas]                          # 残差缩放标量 → AdamW
        x0_params = [self.x0_lambdas]                                # x0 跳连标量 → AdamW（独立 beta1）
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # µP 风格 LR 缩放：模型越宽，AdamW 参数的学习率越小（基准 768 维）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # AdamW 参数组（embedding、lm_head、标量）
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),  # 很小的 LR，缓慢调整
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # 更高的 beta1 使 x0 更平滑
        ]
        # Muon 参数组：按 shape 分组，相同 shape 的参数 stack 成一个张量一起更新（效率高）
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        # 单 GPU 用 MuonAdamW，多 GPU 用 DistMuonAdamW（自带 ZeRO-2 风格通信）
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]  # 保存初始 LR，供 scheduler 使用
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # 获取当前序列长度对应的 RoPE cos/sin（预计算缓存）
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # 推理时需要偏移到 cache 中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # ---- Transformer 主干 ----
        x = self.transformer.wte(idx)  # token → embedding
        x = norm(x)                     # embedding 后立即 norm（稳定训练）
        x0 = x  # 保存初始 embedding，用于 x0 跳连
        for i, block in enumerate(self.transformer.h):
            # 残差缩放 + x0 跳连：x = λ_resid * x + λ_x0 * x0
            # λ_resid 控制残差流强度，λ_x0 让深层直接访问初始 embedding 信息
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Value Embedding：仅在有 VE 的层查表（隔层放置）
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)  # 最终 norm

        # ---- Logit 计算 ----
        softcap = 15  # logit 软上限：tanh 将输出压缩到 [-15, 15]，防止极端 logit
        logits = self.lm_head(x)                           # (B, T, padded_vocab_size) ← 最大的张量
        logits = logits[..., :self.config.vocab_size]       # 裁掉 padding 部分
        logits = logits.float()                             # 转 fp32 确保 softcap 和 loss 的数值精度
        logits = softcap * torch.tanh(logits / softcap)     # 平滑压缩，梯度在边界处不截断（比 clamp 好）

        if targets is not None:
            # 训练：计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理：直接返回 logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """朴素自回归流式推理。假设 batch=1，输入/输出是 Python list/int。"""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)      # (1, T, vocab_size)
            logits = logits[:, -1, :]       # 只取最后一个 token 的 logits
            if top_k is not None and top_k > 0:
                # Top-K 采样：把 top_k 之外的 logit 设为 -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature  # 温度缩放：> 1 更随机，< 1 更确定
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 贪心解码
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token  # 流式 yield 每个生成的 token
