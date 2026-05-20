from typing import *
import torch
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention


__all__ = [
    'sparse_adaptive_attention',
    'topk_kv_selection',
]


# =====================================================================
# Self-Attention: Adaptive Resolution
# =====================================================================

def sparse_adaptive_attention(
    qkv: SparseTensor,
    window_size: int = 32,
    full_attn_threshold: int = 8192,
) -> SparseTensor:
    """
    自适应 self-attention.

    小序列 → Full (全局感受野, 零开销)
    大序列 → Flat Windowed (无 shift, RoPE 提供位置感知)

    Args:
        qkv: SparseTensor [N, *, 3, H, C]
        window_size: 大序列降级时的窗口大小
        full_attn_threshold: 切换阈值
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3

    if qkv.feats.shape[0] <= full_attn_threshold:
        return sparse_scaled_dot_product_attention(qkv)
    else:
        return sparse_windowed_scaled_dot_product_self_attention(
            qkv, window_size, shift_window=(0, 0, 0)
        )


# =====================================================================
# Cross-Attention: Probe-Select-Attend
# =====================================================================

def topk_kv_selection(
    q: SparseTensor,
    kv: torch.Tensor,
    topk: int,
    num_heads: int,
    n_probes: int = 64,
) -> torch.Tensor:
    """
    从 KV 序列中选出与 Q 最相关的 top-k 个 KV token.

    训练时随机采样 probe Q (空间均匀覆盖);
    推理时等距采样 (确定性).

    Args:
        q:   SparseTensor, feats [T, H, C]
        kv:  Tensor [B, L, 2, H, C]
        topk: 保留的 KV 数量
        num_heads: head 数量
        n_probes: 采样的 probe Q 数量

    Returns:
        kv_selected: [B, topk, 2, H, C]
    """
    B = len(q.layout)
    assert kv.shape[0] >= B, f"KV batch {kv.shape[0]} < Q batch {B}"

    kv = kv[:B]
    L, H, C = kv.shape[1], kv.shape[3], kv.shape[4]

    if topk >= L:
        return kv

    k_all = kv[:, :, 0, :, :]                              # [B, L, H, C]
    is_training = q.feats.requires_grad

    probes = []
    for i in range(B):
        sl = q.layout[i]
        n_tokens = (sl.stop - sl.start) if isinstance(sl, slice) else sl.numel()
        qi = q.feats[sl] if n_tokens > 0 else k_all[i]     # [Ni, H, C]
        n = min(n_probes, qi.shape[0])
        if is_training:
            indices = torch.randperm(qi.shape[0], device=qi.device)[:n]
        else:
            indices = torch.linspace(0, qi.shape[0] - 1, n, device=qi.device).long()
        probes.append(qi[indices])

    topk_indices = []
    for i in range(B):
        sim = torch.einsum(
            'nhc,lhc->nl', probes[i].float(), k_all[i].float()
        ) / (C ** 0.5)
        sim_avg = sim.mean(dim=0)                           # [L]
        _, idx = torch.topk(sim_avg, k=topk, dim=-1)
        topk_indices.append(idx)

    topk_indices = torch.stack(topk_indices)                # [B, topk]
    idx_exp = topk_indices[:, :, None, None, None].expand(-1, -1, 2, H, C)
    return torch.gather(kv, dim=1, index=idx_exp)