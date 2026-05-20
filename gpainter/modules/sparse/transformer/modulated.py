from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ..attention.modules import SparseMultiHeadRMSNorm
from ..attention.rope import SparseRotaryPositionEmbedder
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet, SparseSwiGLUFFN
from .moe import SparseMoEFFN


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin", "windowed"] = "shift_window",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        full_attn_threshold: int = 8192,
        cross_attn_topk: int = 0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            full_attn_threshold=full_attn_threshold,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            cross_attn_topk=cross_attn_topk,
        )
        if use_moe:
            self.mlp = SparseMoEFFN(
                channels,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=moe_top_k,
            )
        else:
            self.mlp = SparseSwiGLUFFN(
                channels,
                mlp_ratio=mlp_ratio,
            )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 9 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(9, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mca) + shift_mca
        h = self.cross_attn(h, context)
        h = h * gate_mca
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class PooledSparseTransformerCrossBlock(nn.Module):
    """
    Pooled Sparse Transformer Block: 全部计算在池化后的 super-tokens 上完成.

    70K tokens → pool → 2K super-tokens → (self-attn + cross-attn + MLP) → unpool → 70K tokens

    MLP/Cross-attn 只处理 ~2K tokens, 节省 ~35× 计算量.
    self-attn 在 2K tokens 上用 full attention (天然全局感受野).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        cross_attn_topk: int = 0,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        pool_stride: int = 4,
        **kwargs,  # 吸收 window_size 等不需要的参数
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.pool_stride = pool_stride
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        # self-attn: full attention on pooled tokens (P ≈ 2K, 很快)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            cross_attn_topk=cross_attn_topk,
        )
        if use_moe:
            self.mlp = SparseMoEFFN(
                channels,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=moe_top_k,
            )
        else:
            self.mlp = SparseSwiGLUFFN(
                channels,
                mlp_ratio=mlp_ratio,
            )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 9 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        from ..attention.spca import pool_sparse_tensor, unpool_feats

        # ---- Pool: 70K → ~2K (缓存输入池化结果, 避免二次计算) ----
        x_pooled, inverse = pool_sparse_tensor(x, self.pool_stride)
        pooled_input_feats = x_pooled.feats.clone()  # 保存池化输入

        # ---- adaLN modulation (在 pool 空间) ----
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = mod.chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(9, dim=1)

        # ---- Self-Attention (full, on ~2K tokens) ----
        h = x_pooled.replace(self.norm1(x_pooled.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x_pooled = x_pooled + h

        # ---- Cross-Attention (on ~2K tokens) ----
        h = x_pooled.replace(self.norm2(x_pooled.feats))
        h = h * (1 + scale_mca) + shift_mca
        h = self.cross_attn(h, context)
        h = h * gate_mca
        x_pooled = x_pooled + h

        # ---- MLP (on ~2K tokens) ----
        h = x_pooled.replace(self.norm3(x_pooled.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x_pooled = x_pooled + h

        # ---- Unpool: 广播残差回 70K (无二次池化) ----
        delta = x_pooled.feats - pooled_input_feats  # [P, D]
        delta_unpooled = unpool_feats(delta, inverse)  # [T, D]

        return x.replace(x.feats + delta_unpooled)

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
