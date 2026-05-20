import warnings
import os
import json
import glob
import argparse
from easydict import EasyDict as edict
from contextlib import nullcontext, contextmanager

import torch
import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

from gpainter import models, datasets, trainers
from gpainter.utils.general_utils import dict_foreach, dict_reduce
from gpainter.utils import elastic_utils, grad_clip_utils
from gpainter.trainers.utils import WarmupCosineLRScheduler, LinearWarmupLRScheduler

# 设置日志
logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='dinov2')
warnings.filterwarnings('ignore', category=UserWarning, module='xformers')
warnings.filterwarnings('ignore', message='.*TBB threading layer.*')
warnings.filterwarnings('ignore', message='.*TORCH_CUDA_ARCH_LIST.*')
warnings.filterwarnings('ignore', message='.*Grad strides do not match bucket view strides.*')  # [优化] gradient_as_bucket_view 部分参数回退拷贝，安全忽略

# [优化 1] 启用 TF32 (针对 Ampere/Hopper 架构 GPU，如 H20/A100/H100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # [优化] Conv 算法自动调优
torch.set_float32_matmul_precision('high')  # [优化] TF32 全局启用
os.environ['SPCONV_ALGO'] = 'native'

def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


class AccelerateTrainerMixin:
    """
    Mixin class to adapt existing Trainers to Hugging Face Accelerate.
    Overrides critical methods: init_models_and_more, run_step, save, load.
    """
    def __init__(self, accelerator: Accelerator, *args, **kwargs):
        # Force disable legacy fp16 modes to avoid missing attribute errors in base class run()
        # The base Trainer.run() tries to access self.log_scale or self.scaler based on fp16_mode.
        # Since we rely on Accelerate for mixed precision, we disable these legacy paths.
        kwargs['fp16_mode'] = None

        self.accelerator = accelerator
        # 强制禁用原有的 DDP 检查和手动 DDP
        self.world_size = accelerator.num_processes
        self.rank = accelerator.process_index
        self.local_rank = accelerator.local_process_index
        self.is_master = accelerator.is_main_process

        # 调用父类初始化，但拦截特定的初始化逻辑
        super().__init__(*args, **kwargs)

        # 确保 dataloader 被 prepare (如果在 init 中已经创建)
        # [FIX] 不要使用 accelerator.prepare(dataloader)，因为 base.py 中的 ResumableSampler 已经处理了分布式切分。
        # 双重切分会导致某些 rank 数据为空，从而引发 NCCL 死锁 (Rank 0 等待 clip_grad, Rank N 等待 backward)。
        # if hasattr(self, 'dataloader'):
        #     self.dataloader = self.accelerator.prepare(self.dataloader)
        #     # 更新 iterator
        #     from gpainter.utils.data_utils import cycle
        #     self.data_iterator = cycle(self.dataloader)

        if self.is_master:
            logger.info("Accelerate Trainer Initialized.")

    @property
    def device(self):
        return self.accelerator.device

    def init_models_and_more(self, **kwargs):
        """
        Re-implementation of init_models_and_more using Accelerate.
        """
        # 1. 原始模型已经在 super().__init__ 之前的逻辑中被实例化 (通常在 main 中)
        # 或者在 init_models_and_more 之前 self.models 已经存在

        # 2. 构建 master params (用于优化器)
        # Accelerate 会处理精度，我们只需要收集参数
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        self.master_params = self.model_params

        # 3. 初始化优化器
        # [优化] 注入 fused=True，CUDA fused kernel 加速 optimizer step 10-20%
        opt_args = dict(self.optimizer_config['args'])
        if self.optimizer_config['name'] in ('AdamW', 'Adam') and torch.cuda.is_available():
            opt_args.setdefault('fused', True)

        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **opt_args)
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **opt_args)

        # 4. 初始化 LR Scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # 5. Elastic Controller (保持原逻辑)
        if self.elastic_controller_config is not None:
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # 6. Gradient Clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

        # 7. Accelerate Prepare
        # 将模型、优化器、调度器交给 accelerate
        # 注意：dataset 在 prepare_dataloader 中处理，或者在外部处理

        # 准备模型列表
        model_list = [self.models[name] for name in self.models]

        # [优化 2] 可选：使用 torch.compile 编译模型
        # model_list = [torch.compile(m) for m in model_list]

        objects_to_prepare = model_list + [self.optimizer]
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            objects_to_prepare.append(self.lr_scheduler)

        prepared_objects = self.accelerator.prepare(*objects_to_prepare)

        # 重新分配回 self
        # prepared_objects 顺序: models..., optimizer, [scheduler]
        for i, name in enumerate(self.models):
            self.models[name] = prepared_objects[i]

        self.optimizer = prepared_objects[len(self.models)]
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler = prepared_objects[len(self.models) + 1]

        # 设置 training_models 引用 (兼容 BasicTrainer 的用法)
        self.training_models = self.models

        # EMA 初始化 (仅 Master)
        if self.is_master:
            # 需要 unwrap 模型来获取干净的参数用于 EMA
            unwrapped_models = {name: self.accelerator.unwrap_model(model) for name, model in self.models.items()}
            master_params_raw = sum(
                [[p for p in model.parameters() if p.requires_grad] for model in unwrapped_models.values()]
            , [])
            import copy
            self.ema_params = [copy.deepcopy(master_params_raw) for _ in self.ema_rate]
            # [优化] 缓存 unwrapped 参数列表, 避免每步重建
            self._ema_source_params = master_params_raw

    def run_step(self, data_list):
        """
        Re-implementation of run_step using Accelerate.
        """
        step_log = {'loss': {}, 'status': {}}
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        losses = []
        statuses = []
        elastic_controller_logs = []

        # [优化] 移除手动的 zero_grad，Accelerate 的 accumulate 会处理
        # self.optimizer.zero_grad()

        # Loop over split batch (gradient accumulation within the batch)
        # 假设第一个模型是主模型，用于确定 no_sync 上下文
        main_model = list(self.training_models.values())[0]

        for i, mb_data in enumerate(data_list):
            # 处理 no_sync
            # Accelerate 的 prepare 返回的模型如果有 DDP，会自动处理 no_sync
            # 这里我们为了保持与 BasicTrainer 相同的 batch_split 逻辑，手动控制 sync
            # 如果不是最后一块数据，且是多卡环境，则使用 no_sync
            should_sync = (i == len(data_list) - 1)

            # [优化] 直接使用 accelerator.no_sync 上下文，而不是自己造轮子
            # 如果不需要同步，就进入 no_sync；否则进入 nullcontext
            ctx = self.accelerator.no_sync(main_model) if (not should_sync and self.world_size > 1) else nullcontext()

            with ctx, elastic_controller_context():
                # 计算 Loss
                loss, status = self.training_losses(**mb_data)
                l = loss['loss'] / len(data_list)

                # Backward
                self.accelerator.backward(l)

            # Log
            # [优化] 不要每一步都调用 .item()，这会阻塞 CPU/GPU 流水线！
            # 先 detach 放在 GPU 上，最后统一处理
            losses.append(dict_foreach(loss, lambda x: x.detach() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.detach() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())

        # Gradient Clip
        if self.grad_clip is not None:
            if self.accelerator.sync_gradients:
                # 1. Unscale Gradients
                # 当使用 AMP 时，梯度是 scaled 的。必须先 unscale 才能进行裁剪或自定义操作。
                # accelerator.clip_grad_norm_ 内部会自动 unscale，但不支持自定义 clipper 对象。
                # 所以我们显式调用 unscale_gradients。
                self.accelerator.unscale_gradients(self.optimizer)

                # 2. Clip Gradients
                if isinstance(self.grad_clip, (float, int)):
                    torch.nn.utils.clip_grad_norm_(self.model_params, self.grad_clip)
                else:
                    # 支持 AdaptiveGradClipper 等自定义对象
                    self.grad_clip(self.model_params)

        # Step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)  # [优化] set_to_none 省一次 memset + 降低内存峰值

        # LR Scheduler
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs aggregation (Same as BasicTrainer)
        # [优化] 统一转 CPU/item
        def to_cpu(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        # 注意：dict_reduce 需要修改以支持 Tensor 输入，或者我们在这里先转成 list of scalars
        # 简单起见，我们先在列表层面做转换
        losses_cpu = [dict_foreach(l, to_cpu) for l in losses]
        statuses_cpu = [dict_foreach(s, to_cpu) for s in statuses]

        step_log['loss'] = dict_reduce(losses_cpu, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses_cpu, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()

        # Update EMA (Only Master)
        if self.is_master:
            self.update_ema_accelerate()

        return step_log

    @torch.no_grad()  # [优化] 避免 EMA 操作被追踪梯度
    def update_ema_accelerate(self):
        """
        EMA update adapted for Accelerate.
        """
        # [优化] 使用缓存的参数列表, 不再每步 unwrap + 遍历
        for i, ema_rate in enumerate(self.ema_rate):
            for p, ema_p in zip(self._ema_source_params, self.ema_params[i]):
                ema_p.data.mul_(ema_rate).add_(p.data, alpha=1 - ema_rate)

    # [再次修复] 拦截 snapshot_inference，强制解包模型，并移除多余的同步
    # 注意：这里绝不能加 wait_for_everyone()。
    # 因为在 base.py 中，此方法被 dist_utils.master_first() 包裹，Rank N 已经在外部 Barrier 处等待。
    # 只要我们把模型解包（变成纯本地计算，不发 DDP 消息），Rank 0 就能顺利跑完并释放 Rank N。
    def snapshot_inference(self, *args, **kwargs):
        # 备份模型字典（注意：self.models 和 self.training_models 通常引用同一个对象）
        original_models = {}
        if hasattr(self, 'training_models'):
            original_models = self.training_models.copy()

        need_restore = False

        try:
            if self.is_master:
                # 仅主进程：解包模型 (Unwrap DDP)
                # 使用 unwrap_model 获取原始模型，彻底切断 DDP 通信
                for name, model in self.training_models.items():
                    self.training_models[name] = self.accelerator.unwrap_model(model)
                need_restore = True

            # 调用原始推理逻辑
            # 此时使用的是原生模型，Forward 过程不会触发任何集合通信
            if hasattr(super(), 'snapshot_inference'):
                super().snapshot_inference(*args, **kwargs)

        finally:
            # 恢复 DDP 模型引用，以便后续训练继续使用分布式模型
            if need_restore and original_models:
                self.training_models.clear()
                self.training_models.update(original_models)

            # 不要在此处添加 barrier，依靠外部 master_first 的退出机制进行同步

    def save(self):
        """
        Save checkpoint using Accelerate.
        NOTE: Trainer.run() 只在 rank0 调用 save()，这里不能做 wait_for_everyone()，否则会死锁。
        """
        if not self.is_master:
            return

        logger.info(f"Saving checkpoint at step {self.step}...")
        save_dir = os.path.join(self.output_dir, "ckpts")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save Models
        unwrapped_models = {name: self.accelerator.unwrap_model(model) for name, model in self.models.items()}
        for name, model in unwrapped_models.items():
            self.accelerator.save(model.state_dict(), os.path.join(save_dir, f'{name}_step{self.step:07d}.pt'))

        # 2. Save EMA (不要写回模型参数)
        for i, ema_rate in enumerate(self.ema_rate):
            ema_state_dicts = {}

            # 先拷一份普通 state_dict（detach+cpu），避免引用到活体参数
            for m_name, model in unwrapped_models.items():
                ema_state_dicts[m_name] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            # 用 ema_params 覆盖需要的参数项（注意：赋值，不要 copy_ 到 model 的 tensor 上）
            flat_keys = []
            for m_name, model in unwrapped_models.items():
                for p_name, p in model.named_parameters():
                    if p.requires_grad:
                        flat_keys.append((m_name, p_name))

            for idx, (m_name, p_name) in enumerate(flat_keys):
                ema_state_dicts[m_name][p_name] = self.ema_params[i][idx].detach().cpu()

            for m_name, ema_ckpt in ema_state_dicts.items():
                self.accelerator.save(ema_ckpt, os.path.join(save_dir, f"{m_name}_ema{ema_rate}_step{self.step:07d}.pt"))

        # 3. Save Misc (Optimizer, Scheduler, Step)
        misc_ckpt = {
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()

        self.accelerator.save(misc_ckpt, os.path.join(save_dir, f'misc_step{self.step:07d}.pt'))

        logger.info("Save Done.")

        # 为了完整的训练状态恢复（包括随机数种子等），建议同时使用 accelerate 自带的 save_state
        # self.accelerator.save_state(os.path.join(self.output_dir, 'accelerate_state'))

    def load(self, load_dir, step=0):
        """
        Load checkpoint.
        """
        self.accelerator.wait_for_everyone()
        logger.info(f"Loading checkpoint from step {step}...")

        # Load Models
        for name, model in self.models.items():
            ckpt_path = os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')
            model_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            # Unwrap to load
            unwrapped_model = self.accelerator.unwrap_model(model)
            _missing, _unexpected = unwrapped_model.load_state_dict(model_ckpt, strict=False)
            if _missing:
                logger.warning(f"  [{name}] missing keys (will use init): {_missing}")
            if _unexpected:
                logger.warning(f"  [{name}] unexpected keys (ignored): {_unexpected}")

        # Load Misc
        misc_path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
        misc_ckpt = torch.load(misc_path, map_location='cpu', weights_only=False)

        self.step = misc_ckpt['step']
        try:
            self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        except ValueError as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            logger.warning("This is expected if resuming from a checkpoint trained with a different strategy (e.g. legacy inflat_all vs accelerate). Optimizer state will be reset.")

        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            if 'lr_scheduler' in misc_ckpt:
                try:
                    self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
                except ValueError as e:
                    logger.warning(f"Failed to load lr_scheduler state: {e}")
                    logger.warning("LR scheduler state will be reset.")
            else:
                logger.warning("No lr_scheduler state in checkpoint, starting fresh.")

        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])

        # Re-init EMA params
        if self.is_master:
            # Similar to save logic, verify if we need to load EMA from disk or just init from current
            # For simplicity, if EMA files exist, load them.
            unwrapped_models = {name: self.accelerator.unwrap_model(model) for name, model in self.models.items()}
            for i, ema_rate in enumerate(self.ema_rate):
                for name, model in unwrapped_models.items():
                    path = os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt')
                    if os.path.exists(path):
                        logger.warning(f"EMA checkpoint found at {path} but loading implementation is pending logic verification.")
                        pass  # Loading EMA into memory logic is complex, skipping for brevity as it affects only master RAM

        logger.info("Load Done.")

    def check_ddp(self):
        # Override check_ddp to use accelerator
        pass


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step')
    parser.add_argument('--data_dir', type=str, default='', help='Training data directory (comma-separated for multiple)')
    parser.add_argument('--test_data_dir', type=str, default='', help='Test data directory for snapshot (comma-separated for multiple)')
    parser.add_argument('--auto_retry', type=int, default=1, help='Number of retries')
    parser.add_argument('--tryrun', action='store_true', help='Try run')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    # 添加混合精度参数，默认 fp16，但你可以指定 bf16
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help='Mixed precision mode')

    # 解析部分参数
    opt = parser.parse_args()
    # [优化] gradient_as_bucket_view=True: 大部分参数零拷贝, 少数非连续参数自动回退拷贝 (warning 已抑制)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, bucket_cap_mb=128, gradient_as_bucket_view=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        project_config=ProjectConfiguration(project_dir=opt.output_dir, automatic_checkpoint_naming=True),
        kwargs_handlers=[ddp_kwargs]  # 注入配置
    )

    # 2. 基础配置处理
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    config = json.load(open(opt.config, 'r'))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)

    # 3. 设置随机种子
    set_seed(42 + accelerator.process_index)

    # 4. 打印配置 (Main Process)
    if accelerator.is_main_process:
        print('\n\nConfig:')
        print('=' * 80)
        print(json.dumps(cfg.__dict__, indent=4))
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # 5. 加载数据
    # 数据目录优先级: config.dataset.args.roots > --data_dir
    dataset_args = dict(cfg.dataset.args)
    train_roots = dataset_args.pop('roots', None)
    if not train_roots and cfg.data_dir:
        train_roots = cfg.data_dir
    if not train_roots:
        raise ValueError("Must specify training data via --data_dir or config dataset.args.roots")
    dataset = getattr(datasets, cfg.dataset.name)(train_roots, **dataset_args)

    # 测试数据集 (用于 snapshot)
    test_dataset = None
    test_roots = None
    if hasattr(cfg, 'test_dataset') and cfg.test_dataset is not None:
        # config 中有独立的 test_dataset 定义
        test_ds_cfg = cfg.test_dataset
        test_ds_args = dict(test_ds_cfg.get('args', {}))
        test_roots = test_ds_args.pop('roots', None)
        if not test_roots and cfg.test_data_dir:
            test_roots = cfg.test_data_dir
        if test_roots:
            test_ds_name = test_ds_cfg.get('name', cfg.dataset.name)
            test_dataset = getattr(datasets, test_ds_name)(test_roots, **test_ds_args)
    elif cfg.test_data_dir:
        # 没有独立 test_dataset 配置, 但指定了 --test_data_dir
        test_dataset = getattr(datasets, cfg.dataset.name)(cfg.test_data_dir, **dataset_args)

    if accelerator.is_main_process:
        print(f'\n[Dataset] Train: {len(dataset)} samples')
        if test_dataset is not None:
            print(f'[Dataset] Test: {len(test_dataset)} samples')

    # 6. 构建模型
    # 移除 .cuda()，Accelerate 会处理
    # 关键修复：强制设置 use_fp16=False。
    # Accelerate 的 AMP (GradScaler) 需要模型参数保持为 FP32 (Master Weights)，
    # 如果模型自己转成了 FP16，GradScaler 在 unscale 时会报错 "Attempting to unscale FP16 gradients"。
    for name, model_cfg in cfg.models.items():
        if 'args' in model_cfg and 'use_fp16' in model_cfg.args:
            if model_cfg.args['use_fp16']:
                if accelerator.is_main_process:
                    logger.warning(f"Forcing use_fp16=False for model '{name}' to compatible with Accelerate AMP.")
                model_cfg.args['use_fp16'] = False

    model_dict = {
        name: getattr(models, model.name)(**model.args)
        for name, model in cfg.models.items()
    }

    # Model summary (Main Process) - same behavior as train.py
    if accelerator.is_main_process:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # 7. 动态构建 Trainer 类
    # 获取原始 Trainer 类
    OriginalTrainerClass = getattr(trainers, cfg.trainer.name)

    # 创建混入类
    class AcceleratedTrainer(AccelerateTrainerMixin, OriginalTrainerClass):
        pass

    # 8. 查找 Checkpoint
    cfg = find_ckpt(cfg)

    # 9. 实例化 Trainer
    # 注意：传入 accelerator 实例
    trainer = AcceleratedTrainer(
        accelerator=accelerator,
        models=model_dict,
        dataset=dataset,
        test_dataset=test_dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt
    )

    # 10. 开始训练
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == '__main__':
    main()