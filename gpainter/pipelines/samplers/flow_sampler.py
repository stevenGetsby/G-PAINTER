from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


def _apply_voxel_mask(z, voxel_mask, default_fill=None):
    """mask=0 的位置强制填充确定值.
    default_fill: 与 z 同形状的填充值, 默认为 1.0 (UDF 远场)."""
    if voxel_mask is None:
        return z
    if hasattr(z, 'feats'):
        if default_fill is not None:
            masked_feats = z.feats * voxel_mask + (1.0 - voxel_mask) * default_fill
        else:
            masked_feats = z.feats * voxel_mask + (1.0 - voxel_mask) * 1.0
        return z.replace(masked_feats)
    else:
        if default_fill is not None:
            return z * voxel_mask + (1.0 - voxel_mask) * default_fill
        else:
            return z * voxel_mask + (1.0 - voxel_mask) * 1.0


class FlowSampler(Sampler):
    """
    JiT flow-matching sampler (x-prediction).

    模型输出 pred_x (clean x_0), sampler 内部转为速度:
        v = (pred_x - z_t) / (1 - t)
    ODE: z_{t+dt} = z_t + dt * v.
    最后一步直接返回 pred_x 避免除零.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        if cond is not None and isinstance(cond, torch.Tensor):
            if cond.shape[0] == 1 and x_t.shape[0] > 1:
                cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    # ------------------------------------------------------------------
    def _get_pred_v(self, model, x_t, t_scalar, t_tensor, cond=None, **kwargs):
        """x-prediction → velocity: v = (pred_x - x_t) / (1-t)"""
        pred_x = self._inference_model(model, x_t, t_tensor, cond, **kwargs)
        denom = (1.0 - t_scalar)
        denom = max(denom, 0.05)
        pred_v = (pred_x - x_t) / denom
        return pred_x, pred_v

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_once(self, model, z_t, t, t_next, cond=None, **kwargs):
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device, dtype=z_t.dtype)
        pred_x, pred_v = self._get_pred_v(model, z_t, t, t_tensor, cond, **kwargs)

        # 最后一步直接返回 pred_x
        if t_next >= 1.0 - 1e-5:
            return edict(z_next=pred_x, pred_x=pred_x)

        dt = t_next - t
        z_next = z_t + dt * pred_v
        return edict(z_next=z_next, pred_x=pred_x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def heun_sample_once(self, model, z_t, t, t_next, cond=None, **kwargs):
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device, dtype=z_t.dtype)
        pred_x, pred_v = self._get_pred_v(model, z_t, t, t_tensor, cond, **kwargs)

        # 最后一步直接返回 pred_x
        if t_next >= 1.0 - 1e-5:
            return edict(z_next=pred_x, pred_x=pred_x)

        dt = t_next - t
        z_mid = z_t + dt * pred_v

        # Heun 修正前 mask z_mid, 保持和训练一致
        voxel_mask = kwargs.get('voxel_mask', None)
        default_fill = kwargs.get('default_fill', None)
        z_mid = _apply_voxel_mask(z_mid, voxel_mask, default_fill)

        # Heun 修正: 用 z_mid 处的速度做平均
        t_next_tensor = torch.full((z_t.shape[0],), t_next, device=z_t.device, dtype=z_t.dtype)
        _, pred_v_next = self._get_pred_v(model, z_mid, t_next, t_next_tensor, cond, **kwargs)
        z_next = z_t + dt * 0.5 * (pred_v + pred_v_next)

        return edict(z_next=z_next, pred_x=pred_x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        verbose: bool = True,
        use_heun: bool = False,
        **kwargs,
    ):
        # 提取 voxel_mask (如果有 submask, 上采样到 per-voxel)
        voxel_mask = kwargs.pop('voxel_mask', None)
        default_fill = kwargs.pop('default_fill', None)

        z_t = noise
        t_seq = np.linspace(0.0, 1.0, steps + 1)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        step_fn = self.heun_sample_once if use_heun else self.sample_once

        for t, t_next in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = step_fn(model, z_t, t, t_next, cond, **kwargs)
            z_t = _apply_voxel_mask(out.z_next, voxel_mask, default_fill)

        return edict(samples=z_t)


class FlowCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
