"""
Sparse Pooled Compute Attention (SPCA)
=======================================

核心思想: 将 70K tokens 池化到 ~2K super-tokens, 在 super-tokens 上完成
          整个 block 的计算 (self-attn + cross-attn + MLP), 然后广播回原始 tokens.
          这样 MLP/Cross-attn 只处理 2K tokens, 节省 ~35× 计算量.

设计:
  1. Pool: 按空间 stride 将 tokens 合并为 super-tokens (均值池化)
  2. 在 super-tokens 上执行完整的 transformer block
  3. Unpool: 广播 super-token 的残差增量回原始 tokens
  4. (可选) 极轻量局部 scatter: 1-ring 邻域均值, 无新参数

计算量对比 (N=70K, pool_stride=4 → P≈2K):
                         Original      SPCA
  QKV projection:  70K×768×2304=124B   2K×768×2304=3.5B
  Self-attention:  70K×32²=72M         2K²=4M (full attn!)
  Cross-attention: 70K×512=35.8M       2K×512=1M
  MLP (ratio=4):   70K×768×3072×2=330B 2K×768×3072×2=9.4B
  ─────────────────────────────────────────────────
  Total per layer: ~484B               ~14B  → **~35× faster per layer**

池化映射在首次计算后缓存到 SparseTensor.spatial_cache, 后续层复用.
"""

from typing import *
import torch
import math
from .. import SparseTensor
from .. import ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_pooled_compute_attention',
]


def get_or_create_pool_mapping(
    tensor: SparseTensor,
    pool_stride: int,
    cache_name: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    获取或创建空间池化映射 (缓存复用).

    Returns:
        inverse: [T] 原始 token → pool bucket 的索引
        pool_batch: [P] 每个 pool bucket 的 batch index
        pool_coords: [P, 4] 池化后的坐标 (batch, x//S, y//S, z//S)
        P: 池化后 token 数
    """
    if cache_name is None:
        cache_name = f'spca_pool_{pool_stride}'

    cached = tensor.get_spatial_cache(cache_name)
    if cached is not None:
        return cached

    coords = tensor.coords  # [T, 4]: (batch, x, y, z)

    # 池化坐标
    pooled_coords = coords.clone()
    pooled_coords[:, 1:] = coords[:, 1:] // pool_stride

    # 编码为唯一 ID
    MAX_COORD = pooled_coords[:, 1:].max().item() + 1
    OFFSET = torch.tensor(
        [MAX_COORD ** 3, MAX_COORD ** 2, MAX_COORD, 1],
        device=coords.device, dtype=torch.long
    )
    pooled_ids = (pooled_coords.long() * OFFSET).sum(dim=1)

    # 唯一 bucket
    unique_ids, inverse = torch.unique(pooled_ids, return_inverse=True)
    P = unique_ids.shape[0]

    # batch index per bucket
    pool_batch = torch.zeros(P, device=coords.device, dtype=torch.int32)
    pool_batch.scatter_(0, inverse, coords[:, 0].int())

    # 池化坐标 (取每个 bucket 内任意一个的坐标)
    pool_coords = torch.zeros(P, 4, device=coords.device, dtype=coords.dtype)
    pool_coords.scatter_(0, inverse.unsqueeze(1).expand(-1, 4), pooled_coords)

    result = (inverse, pool_batch, pool_coords, P)
    tensor.register_spatial_cache(cache_name, result)
    return result


def pool_sparse_tensor(
    x: SparseTensor,
    pool_stride: int,
) -> Tuple[SparseTensor, torch.Tensor]:
    """
    池化 SparseTensor: 空间上每 pool_stride³ 合并为一个 super-token.

    Args:
        x: SparseTensor with feats [T, D]
        pool_stride: 空间步幅

    Returns:
        pooled: SparseTensor with feats [P, D] (均值池化)
        inverse: [T] 原始 → 池化的映射索引
    """
    inverse, pool_batch, pool_coords, P = get_or_create_pool_mapping(x, pool_stride)

    T, D = x.feats.shape
    feats = x.feats

    # 均值池化
    pooled_sum = torch.zeros(P, D, device=feats.device, dtype=feats.dtype)
    pooled_cnt = torch.zeros(P, 1, device=feats.device, dtype=torch.float32)

    pooled_sum.scatter_add_(0, inverse.unsqueeze(1).expand(-1, D), feats)
    pooled_cnt.scatter_add_(0, inverse.unsqueeze(1), torch.ones(T, 1, device=feats.device))
    pooled_feats = (pooled_sum / pooled_cnt.clamp(min=1)).to(feats.dtype)

    # 构造池化后的 SparseTensor
    # 重新计算 layout (按 batch 分组)
    B = len(x.layout)
    layout = []
    start = 0
    for b in range(B):
        mask = pool_batch == b
        count = mask.sum().item()
        layout.append(slice(start, start + count))
        start += count

    # 按 batch 排序 (确保同一 batch 的 tokens 连续)
    sort_idx = torch.argsort(pool_batch)
    pooled_feats = pooled_feats[sort_idx]
    sorted_coords = pool_coords[sort_idx]

    # 更新 inverse 映射到排序后的索引
    unsort_idx = torch.empty_like(sort_idx)
    unsort_idx[sort_idx] = torch.arange(P, device=feats.device)
    inverse_sorted = unsort_idx[inverse]

    pooled = SparseTensor(
        feats=pooled_feats,
        coords=sorted_coords,
        shape=torch.Size([B, D]),
        layout=layout,
    )

    return pooled, inverse_sorted


def unpool_feats(
    pooled_feats: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """
    广播池化后的特征回原始 token 级别.

    Args:
        pooled_feats: [P, D]
        inverse: [T] 映射

    Returns:
        [T, D]
    """
    return pooled_feats[inverse]
