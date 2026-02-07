"""
统一的 Flash Attention 接口，自动切换 FA3/SDPA。

导出 `flash_attn` 模块，API 与 FA3 完全一致，但在非 Hopper GPU
（包括 Blackwell）、MPS 和 CPU 上自动回退到 PyTorch SDPA。

用法（FA3 的直接替代品）：
    from nanochat.flash_attention import flash_attn

    # 训练（无 KV cache）
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # 推理（有 KV cache）
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# 硬件检测：尝试在 Hopper+ GPU 上加载 FA3
# =============================================================================
def _load_flash_attention_3():
    """尝试加载 Flash Attention 3（仅限 Hopper GPU, sm90）。
    为什么只 sm90：FA3 的 kernel 只为 Hopper 编译。
    Ada (sm89) 和 Blackwell (sm100) 目前需要 SDPA 回退。"""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 仅支持 sm90（Hopper），其他架构回退到 SDPA
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


# =============================================================================
# SDPA 辅助函数
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    基于 PyTorch SDPA 的注意力实现，支持滑动窗口。
    q, k, v 格式为 (B, H, T, D)。
    分三种情况处理：全上下文 / 单 token 生成 / 滑动窗口。
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # 情况1: 全上下文 + 等长 → 直接用 is_causal=True（SDPA 内置优化）
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # 情况2: 单 token 生成（推理时最常见）→ 裁剪 KV 到窗口范围
    if Tq == 1:
        if window >= 0 and window < Tk:
            # 只保留最近 window+1 个 key
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # 情况3: 需要显式 mask（chunk 推理或滑动窗口 + 非等长）
    device = q.device
    # 构建因果 mask：col_idx <= row_idx（考虑 Tq != Tk 时的偏移）
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # 叠加滑动窗口约束：(row - col) <= window
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# 公共 API：与 FA3 完全相同的接口
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    训练用 Flash Attention（无 KV cache）。

    参数:
        q, k, v: 形状 (B, T, H, D) — FA3 的原生布局
        causal: 是否使用因果 mask
        window_size: (left, right) 滑动窗口，-1 表示无限制

    返回:
        输出张量，形状 (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA 回退：需要转置 (B, T, H, D) → (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)  # Q 和 K 头数不同时启用 GQA
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # 转回 (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    推理用 Flash Attention（带 KV cache）。

    FA3 会原地更新 k_cache/v_cache。SDPA 回退也保持相同行为。

    参数:
        q: 查询，形状 (B, T_new, H, D)
        k_cache, v_cache: 预分配的缓存张量，形状 (B, T_max, H_kv, D)
        k, v: 新的 key/value，形状 (B, T_new, H_kv, D)
        cache_seqlens: cache 中当前位置，形状 (B,) int32
        causal: 是否使用因果 mask
        window_size: (left, right) 滑动窗口，-1 表示无限制

    返回:
        输出张量，形状 (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA 回退：手动管理 KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # 假设 batch 内所有序列位置一致

    # 将新的 k, v 写入 cache（原地更新，与 FA3 行为一致）
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # 取出 cache 中到当前位置的全部 KV
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # 转为 SDPA 格式: (B, T, H, D) → (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # 转回 (B, T, H, D)


# =============================================================================
# 导出：flash_attn 模块接口（FA3 的直接替代品）
# 用 SimpleNamespace 模拟模块，使 `flash_attn.flash_attn_func(...)` 语法可用
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
