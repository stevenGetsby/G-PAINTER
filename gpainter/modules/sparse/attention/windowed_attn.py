from typing import *
import torch
import math
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.
    All heavy computation happens here and results are cached.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        fwd_indices (torch.Tensor): [M] gather indices (original → sorted)
        bwd_indices (torch.Tensor): [M] scatter indices (sorted → original)
        cu_seqlens (torch.Tensor): [num_windows+1] cumulative sequence lengths (on GPU, int32)
        max_seqlen (int): maximum window size
        is_uniform (bool): whether all windows have exactly window_size tokens
        seq_batch_indices (List[int]): batch index for each window (debug only)
    """
    DIM = tensor.coords.shape[1] - 1
    device = tensor.device
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=device)

    # Compute seq_lens on GPU, then build cu_seqlens directly on GPU
    seq_lens_full = torch.bincount(shifted_indices)
    mask = seq_lens_full != 0
    seq_lens_gpu = seq_lens_full[mask]                              # [num_windows] on GPU
    cu_seqlens = torch.zeros(seq_lens_gpu.shape[0] + 1, device=device, dtype=torch.int32)
    torch.cumsum(seq_lens_gpu, dim=0, out=cu_seqlens[1:])
    cu_seqlens = cu_seqlens.int()
    max_seqlen = seq_lens_gpu.max().item()
    ws_scalar = window_size[0] if isinstance(window_size, tuple) else window_size
    is_uniform = bool(seq_lens_gpu.min().item() == ws_scalar and max_seqlen == ws_scalar)

    # batch indices for debug
    seq_batch_indices_gpu = torch.arange(seq_lens_full.shape[0], device=device, dtype=torch.int32) // OFFSET[0]
    seq_batch_indices = seq_batch_indices_gpu[mask].tolist()

    return fwd_indices, bwd_indices, cu_seqlens, max_seqlen, is_uniform, seq_batch_indices


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Optimizations vs naive implementation:
      1. cu_seqlens pre-computed on GPU and cached — no CPU-GPU sync per forward
      2. is_uniform flag cached — avoids Python-level iteration over window list
      3. index_select for gather — fused kernel, better memory coalescing
      4. All metadata (cu_seqlens, max_seqlen) persist in spatial_cache across residual connections

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    # ── Retrieve or compute cached partition (all on GPU) ──────────
    cache_name = f'window_partition_{window_size}_{shift_window}'
    cache = qkv.get_spatial_cache(cache_name)
    if cache is None:
        fwd_indices, bwd_indices, cu_seqlens, max_seqlen, is_uniform, seq_batch_indices = \
            calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(cache_name,
            (fwd_indices, bwd_indices, cu_seqlens, max_seqlen, is_uniform, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, cu_seqlens, max_seqlen, is_uniform, seq_batch_indices = cache

    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]

    # ── Gather: reorder tokens into window-contiguous layout ──────
    # index_select is faster than fancy indexing for large tensors (fused kernel)
    qkv_feats = torch.index_select(qkv.feats, 0, fwd_indices)   # [M, 3, H, C]

    if DEBUG:
        M = fwd_indices.shape[0]
        qkv_coords = qkv.coords[fwd_indices]
        num_windows = cu_seqlens.shape[0] - 1
        sl_cpu = cu_seqlens.cpu()
        for i in range(num_windows):
            s, e = sl_cpu[i].item(), sl_cpu[i+1].item()
            seq_coords = qkv_coords[s:e]
            assert (seq_coords[:, 0] == seq_batch_indices[i]).all(), \
                "SparseWindowedScaledDotProductSelfAttention: batch index mismatch"
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                "SparseWindowedScaledDotProductSelfAttention: window size exceeded"

    # ── Attention ─────────────────────────────────────────────────
    if is_uniform:
        # Fast path: all windows are exactly window_size → packed batch (no cu_seqlens overhead)
        num_windows = cu_seqlens.shape[0] - 1
        qkv_feats = qkv_feats.reshape(num_windows, window_size, 3, H, C)
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)
            out = xops.memory_efficient_attention(q, k, v)
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
        out = out.reshape(num_windows * window_size, H, C)
    else:
        # Variable-length path: cu_seqlens already on GPU, no CPU-GPU sync
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=1)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            # Convert cu_seqlens back to seq_lens list for xformers (required by API)
            sl_cpu = cu_seqlens.cpu()
            seq_lens = (sl_cpu[1:] - sl_cpu[:-1]).tolist()
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
            out = xops.memory_efficient_attention(q, k, v, mask)[0]
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_feats, cu_seqlens, max_seqlen
            )

    # ── Scatter: restore original token order ─────────────────────
    out = torch.index_select(out, 0, bwd_indices)

    if DEBUG:
        qkv_coords = qkv.coords[fwd_indices][bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), \
            "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)
