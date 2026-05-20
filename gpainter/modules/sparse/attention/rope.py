from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor


class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,  # 3D 坐标
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        # x: [T, H, C] -> [T, H, C//2] complex
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # phases: [T, freq_dim] -> [T, 1, freq_dim] 为 heads 维度广播
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(
        self,
        q: SparseTensor,
        k: Optional[SparseTensor] = None,
        use_spatial_coords: bool = True
    ) -> Union[SparseTensor, Tuple[SparseTensor, SparseTensor]]:
        # 检查 coords 维度
        assert q.coords.shape[-1] == self.dim + 1, \
            f"Coords last dim must be {self.dim + 1} (batch_idx + {self.dim}D coords), got {q.coords.shape[-1]}"

        # 获取或计算 phases
        cache_key = f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}_spatial{use_spatial_coords}'
        phases = self._get_phases_cached(q, cache_key, use_spatial_coords)

        # 应用 RoPE
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))

        if k is None:
            return q_embed

        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        return q_embed, k_embed

    def _get_phases_cached(
        self,
        sparse_tensor: SparseTensor,
        cache_key: str,
        use_spatial_coords: bool
    ) -> torch.Tensor:
        """
        获取或计算 RoPE phases，使用 spatial_cache 持久缓存.

        spatial_cache 在 SparseTensor.replace() 时自动传播 (共享 dict 引用),
        因此 phases 仅在第一个 block 首次计算, 后续所有 block 直接复用.
        旧版使用 _cache: 每次 replace() 丢失, 8 个 block 重复计算 8 次.
        """
        cached = sparse_tensor.get_spatial_cache(cache_key)
        if cached is not None:
            return cached

        # 计算 phases
        if use_spatial_coords:
            coords = sparse_tensor.coords[:, 1:].float()
            phases = self._get_phases(coords.reshape(-1))
            phases = phases.reshape(*coords.shape[:-1], -1)
        else:
            T = sparse_tensor.feats.shape[0]
            indices = torch.arange(T, device=sparse_tensor.coords.device, dtype=torch.float32)
            phases = self._get_phases(indices)

        # Padding 到 head_dim // 2
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], padn, device=phases.device),
                torch.zeros(*phases.shape[:-1], padn, device=phases.device)
            )], dim=-1)

        # 缓存到 spatial_cache (通过 replace() 自动传播, 跨 block 复用)
        sparse_tensor.register_spatial_cache(cache_key, phases)

        return phases