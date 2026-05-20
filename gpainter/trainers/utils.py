import torch.nn as nn


# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


# LR Schedulers
import math
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0

class WarmupCosineLRScheduler(LambdaLR):
    """
    Linear warmup -> cosine decay.
    - warmup_steps: 线性从 0 -> base_lr
    - total_steps: 总步数（到这一步时衰减到 min_lr/min_lr_ratio）
    - min_lr: 最小 lr（绝对值），会按 param_group 的 base_lr 转成 ratio
    - min_lr_ratio: 最小 lr 比例（相对 base_lr），二选一
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        min_lr_ratio: float | None = None,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if min_lr_ratio is not None and min_lr is not None and float(min_lr) != 0.0:
            raise ValueError("Use either min_lr or min_lr_ratio (not both).")

        base_lrs = [group["lr"] for group in optimizer.param_groups]

        if min_lr_ratio is None:
            min_lr = float(min_lr)
            ratios = [min_lr / max(lr, 1e-12) for lr in base_lrs]
        else:
            ratios = [float(min_lr_ratio) for _ in base_lrs]

        ratios = [min(max(r, 0.0), 1.0) for r in ratios]

        def make_lambda(r_min: float):
            def f(step: int) -> float:
                if self.warmup_steps > 0 and step < self.warmup_steps:
                    return float(step + 1) / float(self.warmup_steps)

                # warmup 结束后开始 cosine
                if self.total_steps <= self.warmup_steps:
                    return r_min

                progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1->0
                return r_min + (1.0 - r_min) * cosine
            return f

        super().__init__(optimizer, lr_lambda=[make_lambda(r) for r in ratios], last_epoch=last_epoch)