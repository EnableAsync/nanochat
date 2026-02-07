# 阶段 1：模型核心

对应 issue: #1

涉及文件：
- `nanochat/gpt.py` — GPT Transformer 架构（~455 行）
- `nanochat/optim.py` — AdamW + Muon 混合优化器（~534 行）
- `nanochat/common.py` — 工具函数（~259 行）
- `nanochat/flash_attention.py` — Flash Attention 适配层（~180 行）

---

## 1. GPT 模型（`gpt.py`）

### 1.1 整体架构

```
输入 token ids (B, T)
    │
    ▼
┌─ wte ─┐  Embedding 查表 → (B, T, n_embd)
└────────┘
    │
    ▼
  norm()    RMSNorm（无可学习参数）
    │
    │ 保存为 x0（初始 embedding，供后续层跳连使用）
    ▼
╔═══════════════════ Block × n_layer ═══════════════════╗
║                                                        ║
║  x = resid_lambdas[i] * x + x0_lambdas[i] * x0       ║
║      ↑ 可学习标量，缩放残差 + 混入初始 embedding       ║
║                                                        ║
║  ┌─ CausalSelfAttention ─────────────────────────┐    ║
║  │  Q = c_q(norm(x))     K = c_k(norm(x))        │    ║
║  │  V = c_v(norm(x)) + gate * ValueEmbed(idx)    │    ║
║  │        ↓                                       │    ║
║  │  RoPE(Q), RoPE(K) → QK Norm                   │    ║
║  │        ↓                                       │    ║
║  │  Flash Attention（滑动窗口 / 全局）             │    ║
║  │        ↓                                       │    ║
║  │  c_proj → 输出                                 │    ║
║  └────────────────────────────────────────────────┘    ║
║  x = x + attn_output                                  ║
║                                                        ║
║  ┌─ MLP ─────────────────────────────────────────┐    ║
║  │  c_fc(norm(x)) → ReLU² → c_proj → 输出        │    ║
║  └────────────────────────────────────────────────┘    ║
║  x = x + mlp_output                                   ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    │
    ▼
  norm()    最终 RMSNorm
    │
    ▼
┌─ lm_head ─┐  Linear → logit softcap → logits (B, T, vocab_size)
└────────────┘
```

### 1.2 GPTConfig

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 上下文长度
    vocab_size: int = 32768       # 词表大小
    n_layer: int = 12             # Transformer 层数（"depth" 旋钮）
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数（< n_head 则为 GQA）
    n_embd: int = 768             # 隐藏维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

**核心设计**：整个项目通过 `--depth`（即 `n_layer`）一个旋钮控制所有超参。
训练脚本会根据 depth 自动计算 `n_embd`, `n_head` 等，用户不需要手动设置。

### 1.3 与经典 GPT-2 的关键差异

| 特性 | GPT-2 (2019) | nanochat (2026) | 为什么改 |
|---|---|---|---|
| 位置编码 | 可学习绝对位置 embedding | **RoPE**（旋转位置编码） | 更好的长度泛化 |
| Normalization | LayerNorm（有 γ, β 参数） | **RMSNorm（无参数）** | 更快、参数更少 |
| 激活函数 | GELU | **ReLU²** (`relu(x)²`) | 更稀疏、计算更快 |
| Embedding / lm_head | 权重共享（tied） | **不共享（untied）** | 解耦输入/输出表示 |
| Attention | 标准多头注意力 | **GQA + 滑动窗口** | 推理更快、省显存 |
| 额外特性 | 无 | Value Embedding, x0 跳连, logit softcap | 训练稳定性和效果 |

### 1.4 核心组件详解

#### 1.4.1 RMSNorm（L42-44）

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

- **纯函数**，无可学习参数（相比 LayerNorm 省去 γ 和 β）
- 公式：`x / sqrt(mean(x²) + eps)`
- 只做缩放，不做平移 — 足以稳定训练

#### 1.4.2 Rotary Position Embedding — RoPE（L51-57, L243-258）

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]     # 将最后一维拆成两半
    y1 = x1 * cos + x2 * sin             # 旋转
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

**直觉理解**：
- 把每对相邻维度看作二维平面上的点
- 对每个位置 t，按角度 θ_t 旋转该点
- 位置越远，旋转角度差越大 → Q·K 的内积自然编码相对位置
- 不需要单独的位置 embedding 参数

**预计算**（`_precompute_rotary_embeddings`）：
```python
# 每个维度对应一个基频
inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
# 每个时间步 t 和频率组合 → 旋转角度
freqs = torch.outer(t, inv_freq)
cos, sin = freqs.cos(), freqs.sin()
```

预计算结果缓存为 buffer，shape `(1, seq_len, 1, head_dim/2)` —— batch 和 head 维度广播。

#### 1.4.3 CausalSelfAttention（L59-118）

**线性投影**：Q/K/V 分别投影（不是一次性投出再 split）
```python
self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)  # GQA: n_kv_head ≤ n_head
self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
```

**GQA (Grouped-Query Attention)**：
- `n_kv_head < n_head` 时，多个 Q 头共享同一组 K/V 头
- 推理时省显存（KV cache 更小），训练质量几乎不损失

**Value Embedding (ResFormer)**（L86-89）：
```python
if ve is not None:
    ve = ve.view(B, T, n_kv_head, head_dim)
    gate = 2 * sigmoid(self.ve_gate(x[..., :32]))  # gate ∈ (0, 2)
    v = v + gate * ve
```
- 隔层（`has_ve`）从 token id 直接查一个"初始 value"
- 用可学习 gate 控制混入比例。gate 初始化为 0 → sigmoid(0)=0.5 → ×2=1.0（中性）
- 类似残差学习：让 V 在"当前层计算的 V"基础上叠加"全局 token 先验"

**QK Norm**（L94）：
```python
q, k = norm(q), norm(k)
```
旋转后对 Q 和 K 做 RMSNorm，防止注意力分数过大导致 softmax 饱和。

**滑动窗口**（L260-287）：
```python
window_pattern: str = "SSSL"  # S=半窗口, L=全窗口
```
- 按 pattern 循环分配每层的窗口大小
- S 层只看前 `seq_len/2` 个 token → 省计算
- 最后一层强制为 L（全局注意力），确保信息充分汇聚

#### 1.4.4 MLP（L121-131）

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)  # 扩展 4 倍
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)  # 压缩回去

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²：先 ReLU 再平方
        x = self.c_proj(x)
        return x
```

**ReLU²** vs GELU：
- ReLU² 更稀疏（负值直接为 0），有利于特征选择
- 平方操作让正值梯度 `2*relu(x)` 随激活值线性增长，训练动态更好
- 计算比 GELU 更快（无近似或查表）

#### 1.4.5 Block 残差结构（L134-143）

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # Pre-Norm
    x = x + self.mlp(norm(x))
    return x
```

**Pre-Norm**：先 norm 再进子层，残差连接在 norm 外面。相比 Post-Norm 训练更稳定。

#### 1.4.6 GPT.forward()（L388-423）

```python
x = self.transformer.wte(idx)  # Embedding
x = norm(x)
x0 = x  # 保存初始 embedding

for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0  # 残差缩放 + x0 跳连
    ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

x = norm(x)
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]  # 裁掉 padding
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)  # softcap=15
```

**resid_lambdas / x0_lambdas**：
- `resid_lambdas[i]`：缩放残差流。初始化 1.0（中性）
- `x0_lambdas[i]`：混入初始 embedding。初始化 0.1（小权重）
- 思想：让深层也能直接访问输入表示，减轻信息在深层中的衰减

**Logit Softcap**：
```python
logits = 15 * tanh(logits / 15)
```
- 平滑地将 logits 压到 [-15, 15] 范围
- 防止训练后期 logits 爆炸 → 避免过度自信的预测

### 1.5 初始化策略（`init_weights`，L189-241）

| 参数 | 初始化方式 | 原因 |
|---|---|---|
| `wte` (embedding) | Normal(0, 1.0) | 标准随机初始化 |
| `lm_head` | Normal(0, 0.001) | 很小 → 初始输出接近均匀分布 |
| `c_q, c_k, c_v, c_fc` | Uniform(-s, s), s=√3/√d | Uniform 避免异常值 |
| `c_proj, mlp.c_proj` | **全零** | Block 初始为恒等映射 |
| `resid_lambdas` | 1.0 | 标准残差连接 |
| `x0_lambdas` | 0.1 | 少量 x0 混入 |
| `ve_gate` | 全零 | gate=sigmoid(0)×2=1.0（中性） |

**核心思想**：c_proj 零初始化 → attn 和 mlp 的输出都是 0 → 训练开始时每个 Block 就是恒等映射。
这保证了初始时梯度能顺畅流过所有层，不会梯度消失/爆炸。

### 1.6 优化器配置（`setup_optimizer`，L348-386）

参数分为 6 组，使用两种不同的优化器：

| 参数组 | 优化器 | 学习率 | 备注 |
|---|---|---|---|
| `lm_head` | AdamW | 0.004 × scale | 输出层 |
| `wte` (embedding) | AdamW | 0.2 × scale | 输入 embedding |
| `value_embeds` | AdamW | 0.2 × scale | 与 embedding 同 LR |
| `resid_lambdas` | AdamW | 0.005 | 标量，小 LR |
| `x0_lambdas` | AdamW | 0.5 | 标量，较大 LR，beta1=0.96 |
| Transformer 矩阵 | **Muon** | 0.02 | 按 shape 分组 stack |

**LR 缩放**：`scale = (d_model / 768)^{-0.5}` — 模型越宽，AdamW 的 LR 越小。

### 1.7 FLOP 估算（`estimate_flops`，L292-317）

```
FLOPs/token = 6 × (非 embedding 参数量) + Σ_layer 12·h·q·effective_seq
```
- 每个 matmul 参数：forward 2 FLOPs, backward 4 FLOPs → 共 6
- attention 的 Q@K 和 attn@V 另算，考虑滑动窗口的有效序列长度

### 1.8 generate()（L426-454）

朴素自回归生成（无 KV cache 版本）：
- 每步重新跑整个序列的 forward → O(T²) 复杂度
- 支持 temperature 和 top-k 采样
- 用 `yield` 返回 token，支持流式输出

> 注：高效推理版本在 `engine.py`（阶段 5），使用 KV cache。

---

## 2. 优化器（`optim.py`）

### 2.1 整体设计

nanochat 使用**混合优化器**：不同类型的参数用不同的优化算法。

```
参数分类:
├─ Embedding / Scalar 参数 → AdamW（标准一阶优化器）
└─ Transformer 矩阵参数  → Muon（带正交化的动量 SGD）

两个实现:
├─ MuonAdamW      — 单 GPU 版（调试/参考用）
└─ DistMuonAdamW  — 分布式版（实际训练用）
```

### 2.2 AdamW（L20-50）

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)                         # 1. 解耦权重衰减
    exp_avg.lerp_(grad, 1 - beta1_t)                 # 2. 更新一阶动量 m
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)     # 3. 更新二阶动量 v
    bias1 = 1 - beta1_t ** step_t                    # 4. 偏差修正
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)         # 5. 参数更新
```

关键点：
- **`@torch.compile`**：编译为单个 CUDA 图，消除 Python 调用开销
- **0-D CPU tensor**：超参数（lr, beta1 等）用 0 维 tensor 而非 Python float，避免每次值变化时重新编译
- **`lerp_`**：`a.lerp_(b, t)` 等价于 `a = a + t*(b-a) = (1-t)*a + t*b`，更简洁且 fuse 效果更好

### 2.3 Muon 优化器（L52-147）

**Muon = MomentUm Orthogonalized by Newton-schulz**

核心思想：用动量 SGD 算出更新方向后，将其**正交化**（投影到最近的正交矩阵），得到更好的更新方向。

#### Muon 步骤拆解（`muon_step_fused`）

```
梯度 G
  │
  ▼
[Nesterov Momentum]
  momentum_buffer.lerp_(G, 1-μ)       # EMA
  g = G.lerp_(momentum_buffer, μ)      # Nesterov 前瞻
  │
  ▼
[Polar Express 正交化]                  # 5 步迭代
  X = g / ‖g‖                          # 归一化
  repeat 5 times:
    A = X^T @ X    (or X @ X^T for wide matrix)
    B = b*A + c*A²
    X = a*X + X@B  (or a*X + B@X)
  g = X                                # 近似正交矩阵 UV^T
  │
  ▼
[NorMuon 方差归约]
  对每行/列计算 scaling，使更新量在不同神经元间均匀
  │
  ▼
[Cautious Update]
  mask = (g * params) >= 0             # 只更新同方向的维度
  params -= lr * g + lr * wd * params * mask
```

#### 为什么正交化有效？

标准梯度下降中，参数更新 `ΔW = -lr * G` 的 scale 取决于 G 的奇异值 — 大奇异值方向更新大，小方向更新小。
正交化后，`ΔW ≈ -lr * U @ V^T`，所有方向的更新幅度相同。这相当于一种**预条件**（preconditioning），让优化在所有方向上均匀推进。

#### Polar Express（L80-88, L114-127）

这是一种替代 Newton-Schulz 迭代的正交化方法（2025 年论文）：
- 系数 `(a, b, c)` 经过优化，5 步迭代即可收敛
- 区分"高矩阵"和"宽矩阵"以减少计算量
- 全部在 bfloat16 下稳定运行

#### NorMuon 方差归约（L129-140）

正交化后，不同行/列的更新 scale 不均匀。NorMuon 用一个二阶动量来自适应地归一化每行/列的更新幅度。

#### Cautious Update（L142-146）

```python
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```
- 只在"更新方向与参数同号"的位置施加权重衰减
- 避免对已经在"正确方向"衰减的参数反向推动

### 2.4 分布式版本（`DistMuonAdamW`，L297-533）

通过 3 阶段异步通信最大化 overlap：

```
Phase 1: 启动所有异步 reduce 操作
         ├─ AdamW 小参数: all_reduce
         ├─ AdamW 大参数: reduce_scatter（每个 rank 拿 1/N）
         └─ Muon: stack 所有同 shape 参数 → reduce_scatter

Phase 2: 逐组等待 reduce → 计算更新 → 启动 all_gather
         （前面的 gather 和后面的 compute 重叠执行）

Phase 3: 等待所有 gather → 复制回原参数
```

**ZeRO-2 风格分片**：
- 每个 rank 只存自己那份参数的优化器状态（exp_avg, exp_avg_sq / momentum_buffer）
- 更新完自己那份后，all_gather 广播给其他 rank
- 显存占用约减少 N 倍（N = GPU 数量）

---

## 3. Flash Attention 适配层（`flash_attention.py`）

### 3.1 设计目标

提供统一 API，自动选择后端：

| 硬件 | 后端 | 说明 |
|---|---|---|
| Hopper (H100, sm90) | **Flash Attention 3** | 最快，通过 `kernels` 包加载 |
| Ada/Ampere/Blackwell | PyTorch **SDPA** fallback | FA3 未编译支持 |
| MPS / CPU | PyTorch **SDPA** fallback | 通用兼容 |

### 3.2 两个公开 API

```python
# 训练：无 KV cache
flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(W, 0))

# 推理：有 KV cache
flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
```

输入格式 `(B, T, H, D)` — FA3 的原生布局。SDPA 需要转置为 `(B, H, T, D)`。

### 3.3 滑动窗口 SDPA 实现（L61-94）

当 SDPA 后端不支持原生滑动窗口时，手动构建 attention mask：
```python
mask = col_idx <= row_idx                      # causal mask
mask = mask & ((row_idx - col_idx) <= window)  # + 滑动窗口
```

---

## 4. 工具函数（`common.py`）

### 4.1 分布式辅助

| 函数 | 作用 |
|---|---|
| `get_dist_info()` | 返回 `(ddp, rank, local_rank, world_size)` |
| `compute_init()` | 初始化设备、随机种子、DDP 进程组 |
| `compute_cleanup()` | 销毁进程组 |
| `print0()` | 只在 rank 0 打印，避免多 GPU 输出混乱 |

### 4.2 设备自动检测

```python
def autodetect_device_type():
    # 优先级: CUDA > MPS > CPU
```

### 4.3 Peak FLOPS 表

`get_peak_flops()` 硬编码了各 GPU 的 BF16 理论峰值 FLOPS，用于计算 MFU（Model FLOPS Utilization）。
未知 GPU 返回 `inf` → MFU 显示 0% 而非错误值。

### 4.4 文件下载

`download_file_with_lock()` 使用文件锁防止多 rank 并发下载同一文件。

---

## 附录：阅读建议

1. **先读 `GPTConfig` 和 `GPT.forward()`**：理解数据流向
2. **再读 `CausalSelfAttention.forward()`**：最复杂但最重要的部分
3. **然后读 `init_weights()`**：理解"为什么训练能收敛"
4. **最后读 `setup_optimizer()`**：理解参数分组策略
5. **`optim.py` 可以先看 `MuonAdamW`（单 GPU 版）**，理解算法后再看 `DistMuonAdamW`
6. `flash_attention.py` 和 `common.py` 是辅助模块，快速浏览即可
