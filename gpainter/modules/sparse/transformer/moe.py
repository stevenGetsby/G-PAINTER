"""
Sparse Mixture-of-Experts (MoE) FFN
====================================

设计参考 UltraShape / DeepSeek-V2:
  - Top-K 路由: 每个 token 选 top-k 个 expert (默认 top-2)
  - 共享 expert: 所有 token 都经过, 提供基线容量
  - Load-Balancing Loss: 鼓励 expert 均匀使用, 通过 autograd trick 隐式回传

与 UltraShape 的区别:
  - 输入是 SparseTensor [T, C] (flat tokens), 不是 [B, L, C]
  - 更简洁: gate 直接对 T 个 token 操作, 无需 reshape
  - 共用 SparseFeedForwardNet 作为 expert

使用场景:
  - 最后 N 层替换普通 FFN, 增加模型容量
  - 100K+ 数据量时效果明显, 少量数据不建议用
  - 8 experts top-2 → 每 token 激活 2/8 = 25% expert 参数
"""

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic import SparseTensor
from .blocks import SparseSwiGLUFFN


__all__ = ['SparseMoEFFN']


class _AddAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        assert aux_loss.numel() == 1
        ctx.dtype = aux_loss.dtype
        ctx.requires_aux = aux_loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_loss = None
        if ctx.requires_aux:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


# =====================================================================
# MoE Gate
# =====================================================================

class SparseMoEGate(nn.Module):
    """
    Top-K softmax gate with load-balancing auxiliary loss.

    aux_loss = α · Σᵢ (Pᵢ · fᵢ)
    - Pᵢ: expert i 的平均 gate 概率
    - fᵢ: expert i 被选中的频率 × num_experts
    - 鼓励所有 expert 被均匀选中
    """
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = aux_loss_alpha
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, C] flat sparse tokens
        Returns:
            topk_idx:    [T, top_k]  选中的 expert 编号
            topk_weight: [T, top_k]  对应的 gate 权重
            aux_loss:    scalar      load-balancing loss
        """
        # Gate scores: [T, num_experts]
        logits = F.linear(x.float(), self.weight.float())
        scores = F.softmax(logits, dim=-1)

        # Top-K selection + renormalize
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # Aux loss (训练时)
        if self.training:
            # ce: 每个 expert 被选中的频率
            mask = F.one_hot(topk_idx.view(-1), num_classes=self.num_experts)
            ce = mask.float().mean(0)                       # [num_experts]
            fi = ce * self.num_experts                      # 归一化频率
            Pi = scores.mean(0)                             # [num_experts] 平均概率
            aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = torch.zeros(1, device=x.device)

        return topk_idx, topk_weight, aux_loss


# =====================================================================
# Sparse MoE FFN
# =====================================================================

class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts Feed-Forward Network.

    结构: Top-K 路由 experts + 1 共享 expert

    输出 = Σ (wₖ · Expertₖ(x))  +  SharedExpert(x)
           ─────────────────       ──────────────────
           路由部分 (稀疏)         共享部分 (所有 token)
    """
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由 experts
        self.gate = SparseMoEGate(channels, num_experts, top_k, aux_loss_alpha)
        self.experts = nn.ModuleList([
            SparseSwiGLUFFN(channels, mlp_ratio) for _ in range(num_experts)
        ])
        # 共享 expert: 所有 token 都走, 提供基线能力
        self.shared_expert = SparseSwiGLUFFN(channels, mlp_ratio)
        self.shared_expert_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: SparseTensor) -> SparseTensor:
        """
        Args:
            x: SparseTensor, feats [T, C]
        Returns:
            SparseTensor, feats [T, C]
        """
        identity = x
        feats = x.feats  # [T, C]
        T, C = feats.shape

        # Gate
        topk_idx, topk_weight, aux_loss = self.gate(feats)  # [T, top_k]

        # 路由部分: 对每个 expert 处理被分配到的 token
        y = torch.zeros_like(feats)
        for i, expert in enumerate(self.experts):
            # 找到被路由到 expert i 的 (token, slot) 对
            mask = (topk_idx == i)                          # [T, top_k] bool
            token_mask = mask.any(dim=-1)                   # [T]
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                # DDP 要求所有参数参与梯度计算
                # 用第一个 token 过 expert, 乘 0 → 零贡献但有梯度路径
                x_dummy = SparseTensor(
                    feats=x.feats[:1],
                    coords=x.coords[:1],
                    shape=torch.Size([1, C]),
                    layout=[slice(0, 1)],
                )
                y = y + expert(x_dummy).feats.sum() * 0
                continue

            # 提取这些 token, 过 expert
            x_sel = x.feats[token_indices]                  # [n, C]
            # 创建临时 SparseTensor 让 SparseFeedForwardNet 处理
            x_sparse = SparseTensor(
                feats=x_sel,
                coords=x.coords[token_indices],
                shape=torch.Size([1, C]),
                layout=[slice(0, len(token_indices))],
            )
            out_feats = expert(x_sparse).feats              # [n, C]

            # 加权累加 (一个 token 可能在多个 slot 选了同一 expert)
            w = topk_weight[token_indices]                  # [n, top_k]
            m = mask[token_indices]                         # [n, top_k] bool
            # 总权重 = 该 token 在选中 expert i 的所有 slot 的权重之和
            w_sum = (w * m.float()).sum(dim=-1, keepdim=True)  # [n, 1]
            y[token_indices] += out_feats * w_sum

        # 共享 expert (可学习缩放, 平衡路由/共享贡献)
        shared_out = self.shared_expert(identity).feats     # [T, C]
        y = y + shared_out * self.shared_expert_scale

        # 注入 aux_loss
        if self.training:
            y = _AddAuxLoss.apply(y, aux_loss)

        return x.replace(y)
