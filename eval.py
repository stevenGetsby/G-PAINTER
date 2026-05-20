#!/usr/bin/env python3
"""
GRAVER 数据集推理评估脚本.

从训练集随机抽取样本, 一次性运行所有适用模式, 输出到子文件夹方便对比:

  eval_output/
  ├── cond/           # 条件图片
  │   ├── 1.jpg
  │   └── ...
  ├── gt/             # GT → mesh (始终运行)
  │   ├── 1.ply
  │   ├── 1_normal.jpg
  │   └── ...
  ├── feats_only/     # GT coords + GT mask → 预测 feats → mesh
  │   ├── 1.ply
  │   ├── 1_normal.jpg
  │   └── ...
  └── mask_feats/     # GT coords → 预测 mask → 预测 feats → mesh
      ├── 1.ply
      ├── 1_normal.jpg
      └── ...

用法:
  python eval.py \\
      --data_root /mnt/data/yizhao/TRAIN \\
      --mask_ckpt_dir ../../ckpt/mask_test \\
      --feats_ckpt_dir ../../ckpt/feats_test \\
      --max_block_num 7000 --max_samples 200 \\
      --num_samples 5
"""
import argparse
import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image

from gpainter import models
from gpainter.datasets.block_feats import BlockFeats
from gpainter.dataset_toolkits.mesh2block import (
    BLOCK_DIM, BLOCK_GRID, BLOCK_FOLDER, COL_PREFIX,
)
from gpainter.modules.sparse.basic import SparseTensor
from gpainter.pipelines.samplers import FlowGuidanceIntervalSampler
from gpainter.trainers.flow_matching.mixins.image_conditioned import (
    ImageConditionedMixin as ImageCondHelper,
)


# =====================================================================
# Helpers
# =====================================================================

def _load_json(path: str) -> edict:
    with open(path, 'r') as f:
        return edict(json.load(f))


def _resolve_step(load_dir: str, ckpt: Union[str, int] = 'latest') -> int:
    if isinstance(ckpt, int):
        return ckpt
    if ckpt == 'latest':
        files = glob.glob(os.path.join(load_dir, 'ckpts', 'misc_*.pt'))
        if not files:
            raise FileNotFoundError(f'No checkpoint found under {load_dir}/ckpts')
        return max(int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files)
    return int(ckpt)


def _load_model(config_path: str, ckpt_dir: str, ckpt: str = 'latest',
                ema_rate: Optional[float] = 0.999, model_key: str = 'denoiser',
                device: str = 'cuda') -> torch.nn.Module:
    cfg = _load_json(config_path)
    model_cfg = cfg.models[model_key]
    model = getattr(models, model_cfg.name)(**model_cfg.args)
    step = _resolve_step(ckpt_dir, ckpt)
    if ema_rate is not None:
        ckpt_name = f'{model_key}_ema{ema_rate}_step{step:07d}.pt'
    else:
        ckpt_name = f'{model_key}_step{step:07d}.pt'
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name)
    if not os.path.exists(ckpt_path) and ema_rate is not None:
        ckpt_name = f'{model_key}_step{step:07d}.pt'
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    print(f'  Loading {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # strict=True 确保所有参数都加载, 不会静默跳过
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  ⚠ MISSING keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  ⚠ UNEXPECTED keys ({len(unexpected)}): {unexpected[:5]}...')
    if not missing and not unexpected:
        print(f'  ✓ All {len(state_dict)} keys loaded successfully')
    model.eval().to(device)
    return model


def _adapt_submask(submask: torch.Tensor, src_res: int, dst_res: int) -> torch.Tensor:
    """将 submask 从 src_res³ 上/下采样到 dst_res³ (nearest)."""
    if src_res == dst_res:
        return submask
    T = submask.shape[0]
    sub_3d = submask.reshape(T, 1, src_res, src_res, src_res)
    out_3d = F.interpolate(sub_3d, size=dst_res, mode='nearest')
    return out_3d.reshape(T, -1)


def _upsample_submask_to_voxel(submask: torch.Tensor, submask_res: int) -> torch.Tensor:
    """submask [T, R³] → voxel mask [T, BLOCK_DIM³]."""
    T = submask.shape[0]
    scale = BLOCK_DIM // submask_res
    sub_3d = submask.reshape(T, 1, submask_res, submask_res, submask_res)
    voxel_3d = F.interpolate(sub_3d, scale_factor=scale, mode='nearest')
    return voxel_3d.reshape(T, -1)


def _detect_submask_res(submask: torch.Tensor) -> int:
    """从 submask 维度推断分辨率: dim = res³ → res."""
    dim = submask.shape[1]
    res = round(dim ** (1.0 / 3.0))
    assert res ** 3 == dim, f"submask dim {dim} is not a perfect cube"
    return res


# =====================================================================
# Image encoder
# =====================================================================

class _ImageEncoder:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.helper = ImageCondHelper(image_cond_model='dinov2_vitl14_reg')
        self.helper._init_image_cond_model()
        self.helper.image_cond_model['model'] = (
            self.helper.image_cond_model['model'].to(device)
        )

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        img = np.array(image.convert('RGB')).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = self.helper.image_cond_model['transform'](img)
        feats = self.helper.image_cond_model['model'](img, is_training=True)['x_prenorm']
        return F.layer_norm(feats, feats.shape[-1:])


# =====================================================================
# Data loading
# =====================================================================

def _pick_random_indices(data_root: str, num: int, *,
                         max_block_num: int = 0, max_samples: int = 0,
                         seed: int = 42) -> List[int]:
    import pandas as pd
    metadata = pd.read_csv(os.path.join(data_root, 'metadata.csv'))
    metadata = metadata[metadata[f'{COL_PREFIX}_block_status'] == 'success']
    metadata = metadata[metadata['cond_rendered'].fillna(False).astype(bool)]
    if max_block_num > 0:
        metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] <= max_block_num]
    metadata = metadata.reset_index(drop=True)
    if max_samples > 0 and len(metadata) > max_samples:
        metadata = metadata.iloc[:max_samples]
    total = len(metadata)
    rng = np.random.RandomState(seed)
    num = min(num, total)
    indices = sorted(rng.choice(total, size=num, replace=False).tolist())
    print(f'Dataset: {total} samples, randomly picked {num}: {indices}')
    return indices


def load_sample(data_root: str, sample_idx: int, image_size: int = 518,
                max_block_num: int = 0, max_samples: int = 0):
    import pandas as pd
    metadata = pd.read_csv(os.path.join(data_root, 'metadata.csv'))
    metadata = metadata[metadata[f'{COL_PREFIX}_block_status'] == 'success']
    metadata = metadata[metadata['cond_rendered'].fillna(False).astype(bool)]
    if max_block_num > 0:
        metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] <= max_block_num]
    metadata = metadata.reset_index(drop=True)
    if max_samples > 0 and len(metadata) > max_samples:
        metadata = metadata.iloc[:max_samples]
    if sample_idx >= len(metadata):
        raise IndexError(f'sample_idx={sample_idx} >= {len(metadata)}')

    sha256 = metadata.at[sample_idx, 'sha256']
    local_path = metadata.at[sample_idx, 'local_path'] if 'local_path' in metadata.columns else ''

    # NPZ
    npz_path = os.path.join(data_root, BLOCK_FOLDER, f'{sha256}.npz')
    with np.load(npz_path) as data:
        coords = torch.from_numpy(data['coords']).int()
        raw = data['fine_feats']
        fine_feats = torch.from_numpy(
            raw.astype(np.float32) if raw.dtype == np.float16 else raw
        ).float()
        if 'submask' in data.files:
            submask = torch.from_numpy(data['submask'].astype(np.float32))
        else:
            submask = torch.ones(coords.shape[0], 64)

    # 条件图片 (与 ImageConditionedMixin 预处理一致)
    image_root = os.path.join(data_root, 'renders_cond', sha256)
    with open(os.path.join(image_root, 'transforms.json')) as f:
        transforms_meta = json.load(f)
    frame = transforms_meta['frames'][0]
    image_path = os.path.join(image_root, frame['file_path'])
    image = Image.open(image_path)

    alpha = np.array(image.getchannel(3))
    nz = alpha.nonzero()
    if nz[0].size == 0:
        h, w = alpha.shape
        bbox = [0, 0, w - 1, h - 1]
    else:
        bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * 1.2
    crop = [int(cx - hsize), int(cy - hsize), int(cx + hsize), int(cy + hsize)]
    image = image.crop(crop).resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha_ch = image.getchannel(3)
    image = image.convert('RGB')
    img_t = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    alpha_t = torch.tensor(np.array(alpha_ch)).float() / 255.0
    cond_pil = Image.fromarray(
        (img_t * alpha_t.unsqueeze(0)).permute(1, 2, 0).mul(255).clamp(0, 255)
        .byte().numpy()
    )

    # 原始 3D 文件路径 (metadata local_path 列)
    if local_path:
        raw_path = os.path.join(data_root, local_path)
    else:
        raw_path = ''

    return {
        'sha256': sha256,
        'coords': coords,
        'submask': submask,
        'fine_feats': fine_feats,
        'cond_image': cond_pil,
        'raw_path': raw_path,
    }


# =====================================================================
# Sampling — 与训练推理保持一致
# =====================================================================

@torch.no_grad()
def sample_mask(model, cond: torch.Tensor, coords: torch.Tensor, *,
                noise_scale: float = 1.0, cfg_strength: float = 3.0,
                cfg_interval: Tuple[float, float] = (0.1, 1.0),
                steps: int = 50, device: str = 'cuda') -> torch.Tensor:
    """GT coords → model(noise) → 预测 submask.

    维度由 model.token_dim 决定 (occ4=64, occ8=512), 不硬编码.
    阈值 0.5 与训练 train_iou 监控一致.
    """
    batch_coords = torch.cat([
        torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32),
        coords.to(device),
    ], dim=1)

    noise = SparseTensor(
        feats=torch.randn(batch_coords.shape[0], model.token_dim, device=device) * noise_scale,
        coords=batch_coords,
    )
    # 训练用 accelerate --mixed_precision bf16, 推理也用 autocast bf16 保持一致
    sampler = FlowGuidanceIntervalSampler()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        result = sampler.sample(
            model, noise=noise, cond=cond, neg_cond=torch.zeros_like(cond),
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
            steps=steps, verbose=True,
        )
    return (result.samples.feats.float() > 0.5).float()


@torch.no_grad()
def sample_feats(model, cond: torch.Tensor, coords: torch.Tensor,
                 submask: torch.Tensor, *, noise_scale: float = 2.0,
                 cfg_strength: float = 3.0,
                 cfg_interval: Tuple[float, float] = (0.1, 1.0),
                 steps: int = 50, device: str = 'cuda') -> torch.Tensor:
    """GT coords + submask → model(noise) → 预测 fine_feats.

    submask 必须已匹配 model.submask_resolution 维度;
    voxel_mask 从 submask 上采样到 BLOCK_DIM³, 用于 noise masking (与训练一致).
    """
    batch_coords = torch.cat([
        torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32),
        coords.to(device),
    ], dim=1)

    submask_d = submask.to(device)
    submask_res = _detect_submask_res(submask_d)
    voxel_mask = _upsample_submask_to_voxel(submask_d, submask_res)

    # noise masking: mask=0 位置填确定值 1.0 (与训练一致)
    noise_raw = torch.randn(batch_coords.shape[0], model.token_dim, device=device) * noise_scale
    noise_raw = noise_raw * voxel_mask + (1.0 - voxel_mask) * 1.0
    noise = SparseTensor(feats=noise_raw, coords=batch_coords)

    sampler = FlowGuidanceIntervalSampler()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        result = sampler.sample(
            model, noise=noise, cond=cond, neg_cond=torch.zeros_like(cond),
            submask=submask_d, voxel_mask=voxel_mask,
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
            steps=steps, verbose=True,
        )
    pred = result.samples.feats.float() * voxel_mask + (1.0 - voxel_mask) * 1.0
    return pred.clamp(0.0, 1.0)


# =====================================================================
# Mesh reconstruction + rendering
# =====================================================================

def _save_mesh_and_normal(coords_np, feats_np, mesh_path, normal_path):
    ok = BlockFeats.tokens_to_mesh(coords_np, feats_np, mesh_path, verbose=True)
    if ok:
        BlockFeats.render_normal_grid(
            mesh_path, normal_path, resolution=1024, radius=1.75, verbose=True,
        )
    return ok


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='GRAVER dataset inference evaluation')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./eval_output')

    parser.add_argument('--mask_ckpt_dir', type=str, default='')
    parser.add_argument('--feats_ckpt_dir', type=str, default='')
    parser.add_argument('--mask_config', type=str,
                        default='configs/flow_matching/block_mask.json')
    parser.add_argument('--feats_config', type=str,
                        default='configs/flow_matching/block_feats.json')
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--ema_rate', type=float, default=0.999)

    parser.add_argument('--max_block_num', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=0)

    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg_strength', type=float, default=3.0)
    parser.add_argument('--noise_scale_mask', type=float, default=1.0)
    parser.add_argument('--noise_scale_feats', type=float, default=2.0)
    parser.add_argument('--cfg_interval_min', type=float, default=0.1)
    parser.add_argument('--cfg_interval_max', type=float, default=1.0)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ema = args.ema_rate if args.ema_rate > 0 else None
    has_mask = bool(args.mask_ckpt_dir)
    has_feats = bool(args.feats_ckpt_dir)

    # ---- 创建输出目录 ----
    dirs = {}
    for d in ['cond', 'gt', 'raw_gt']:
        dirs[d] = os.path.join(args.output_dir, d)
        os.makedirs(dirs[d], exist_ok=True)
    if has_feats:
        dirs['feats_only'] = os.path.join(args.output_dir, 'feats_only')
        os.makedirs(dirs['feats_only'], exist_ok=True)
    if has_mask and has_feats:
        dirs['mask_feats'] = os.path.join(args.output_dir, 'mask_feats')
        os.makedirs(dirs['mask_feats'], exist_ok=True)

    # ---- 加载模型 ----
    mask_model = None
    feats_model = None

    if has_mask:
        mask_model = _load_model(
            args.mask_config, args.mask_ckpt_dir, args.ckpt, ema, device=args.device,
        )
    if has_feats:
        feats_model = _load_model(
            args.feats_config, args.feats_ckpt_dir, args.ckpt, ema, device=args.device,
        )

    encoder = None
    if has_mask or has_feats:
        print('Initializing DINOv2 image encoder ...')
        encoder = _ImageEncoder(device=args.device)

    # ---- 随机抽样 ----
    sample_indices = _pick_random_indices(
        args.data_root, args.num_samples,
        max_block_num=args.max_block_num, max_samples=args.max_samples,
        seed=args.seed,
    )

    # ---- 逐样本推理 ----
    metrics_lines = []

    for seq_num, idx in enumerate(sample_indices, start=1):
        print(f'\n{"=" * 60}')
        print(f'[{seq_num}/{len(sample_indices)}] dataset_idx={idx}')
        print(f'{"=" * 60}')

        sample = load_sample(
            args.data_root, idx,
            max_block_num=args.max_block_num, max_samples=args.max_samples,
        )
        n_blocks = sample['coords'].shape[0]
        print(f'  sha256={sample["sha256"]}, blocks={n_blocks}')

        # 条件图片
        sample['cond_image'].save(os.path.join(dirs['cond'], f'{seq_num}.jpg'))

        # 原始 3D 文件
        raw_src = sample['raw_path']
        if raw_src and os.path.exists(raw_src):
            ext = os.path.splitext(raw_src)[1]  # 保留原始扩展名 (.glb/.obj/...)
            shutil.copy2(raw_src, os.path.join(dirs['raw_gt'], f'{seq_num}{ext}'))
        else:
            print(f'  ⚠ raw file not found: {raw_src}')

        coords_np = sample['coords'].numpy().astype(np.int32)
        gt_submask = sample['submask']
        gt_feats = sample['fine_feats']
        gt_submask_res = _detect_submask_res(gt_submask)

        # ---- GT → mesh ----
        print(f'\n  [gt] tokens_to_mesh ...')
        _save_mesh_and_normal(
            coords_np, gt_feats.numpy().astype(np.float32),
            os.path.join(dirs['gt'], f'{seq_num}.ply'),
            os.path.join(dirs['gt'], f'{seq_num}_normal.jpg'),
        )

        cond = None
        if encoder is not None:
            cond = encoder.encode(sample['cond_image'])

        # ---- feats_only: GT coords + GT mask → predict feats → mesh ----
        if has_feats:
            print(f'\n  [feats_only] sampling feats (GT mask) ...')
            feats_submask_res = feats_model.submask_resolution
            if feats_submask_res > 0 and gt_submask_res != feats_submask_res:
                submask_for_feats = _adapt_submask(
                    gt_submask, gt_submask_res, feats_submask_res,
                )
                print(f'    submask adapted: {gt_submask_res}³→{feats_submask_res}³')
            else:
                submask_for_feats = gt_submask

            pred_feats = sample_feats(
                feats_model, cond, sample['coords'], submask_for_feats,
                noise_scale=args.noise_scale_feats,
                cfg_strength=args.cfg_strength,
                cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
                steps=args.steps, device=args.device,
            )
            _save_mesh_and_normal(
                coords_np, pred_feats.cpu().numpy().astype(np.float32),
                os.path.join(dirs['feats_only'], f'{seq_num}.ply'),
                os.path.join(dirs['feats_only'], f'{seq_num}_normal.jpg'),
            )

        # ---- mask_feats: GT coords → predict mask → predict feats → mesh ----
        if has_mask and has_feats:
            print(f'\n  [mask_feats] stage 2: sampling submask ...')
            pred_mask = sample_mask(
                mask_model, cond, sample['coords'],
                noise_scale=args.noise_scale_mask,
                cfg_strength=args.cfg_strength,
                cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
                steps=args.steps, device=args.device,
            )
            pred_mask_res = mask_model.resolution

            # 与 GT submask 对比指标 (在统一分辨率下)
            if gt_submask_res != pred_mask_res:
                gt_mask_cmp = _adapt_submask(
                    gt_submask, gt_submask_res, pred_mask_res,
                ).to(args.device)
            else:
                gt_mask_cmp = gt_submask.to(args.device)
            pred_b = (pred_mask > 0.5).float()
            gt_b = (gt_mask_cmp > 0.5).float()
            tp = (pred_b * gt_b).sum().item()
            fp = (pred_b * (1 - gt_b)).sum().item()
            fn = ((1 - pred_b) * gt_b).sum().item()
            iou = tp / max(tp + fp + fn, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            line = (f'  sample {seq_num}: IoU={iou:.4f}  '
                    f'Precision={prec:.4f}  Recall={rec:.4f}')
            print(line)
            metrics_lines.append(line)

            # 适配 submask 到 feats model 期望分辨率
            feats_submask_res = feats_model.submask_resolution
            if feats_submask_res > 0 and pred_mask_res != feats_submask_res:
                submask_for_feats = _adapt_submask(
                    pred_mask, pred_mask_res, feats_submask_res,
                )
            else:
                submask_for_feats = pred_mask

            print(f'  [mask_feats] stage 3: sampling feats (predicted mask) ...')
            pred_feats = sample_feats(
                feats_model, cond, sample['coords'], submask_for_feats,
                noise_scale=args.noise_scale_feats,
                cfg_strength=args.cfg_strength,
                cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
                steps=args.steps, device=args.device,
            )
            _save_mesh_and_normal(
                coords_np, pred_feats.cpu().numpy().astype(np.float32),
                os.path.join(dirs['mask_feats'], f'{seq_num}.ply'),
                os.path.join(dirs['mask_feats'], f'{seq_num}_normal.jpg'),
            )

    # 保存 mask 指标汇总
    if metrics_lines:
        metrics_path = os.path.join(args.output_dir, 'mask_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write('\n'.join(metrics_lines) + '\n')
        print(f'\nMask metrics saved: {metrics_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
