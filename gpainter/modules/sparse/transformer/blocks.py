from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseSwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA / PaLM / JiT style).

    SwiGLU(x) = SiLU(W_gate · x) ⊙ (W_up · x)  then  W_down

    3 个投影矩阵, hidden = 2/3 × mlp_ratio × C 使总参数量 ≈ 普通 GELU FFN.
    """
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = SparseLinear(channels, hidden)
        self.w_up = SparseLinear(channels, hidden)
        self.w_down = SparseLinear(hidden, channels)

    def forward(self, x: SparseTensor) -> SparseTensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        h = gate.replace(F.silu(gate.feats) * up.feats)
        return self.w_down(h)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
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
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
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

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
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
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
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
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: SparseTensor, context: torch.Tensor):
        h = x.replace(self.norm1(x.feats))
        h = self.self_attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)
