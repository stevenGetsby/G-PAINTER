from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
from ..dataset_toolkits.mesh2block import BLOCK_GRID


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if self.mlp[0].weight.dtype != t_freq.dtype:
            t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class DenseFlowModel(nn.Module):
    """
    Dense flow model for Stage 1: 64³ occupancy prediction.
    Patchifies 64³ volume into 8³ patches → 512 tokens → full attention.
    """
    def __init__(
        self,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        bottleneck_dim: int = 128,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        resolution = BLOCK_GRID
        in_channels = 1
        out_channels = 1

        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            grid_size = resolution // patch_size
            coords = torch.stack(torch.meshgrid(
                torch.arange(grid_size), torch.arange(grid_size), torch.arange(grid_size),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)
            self.register_buffer("grid_coords", coords, persistent=False)

        self.input_layer = nn.Sequential(
            nn.Linear(in_channels * patch_size**3, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, model_channels)
        )
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_mode='full', use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"), share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm, qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

        grid_size = resolution // patch_size
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[DenseFlowModel] BLOCK_GRID={resolution}, patch={patch_size}, "
              f"tokens={grid_size**3}, ch={model_channels}x{num_blocks}, "
              f"heads={self.num_heads}, fp16={use_fp16}, params={n_params:.1f}M")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self):
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
            f"Input shape mismatch, got {x.shape}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        t_emb = self.t_embedder(t * 1000)
        if hasattr(self, "grid_coords"):
            h = h + self.pos_embedder(self.grid_coords).unsqueeze(0)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        return h
