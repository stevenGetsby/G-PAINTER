#!/usr/bin/env python3
"""
PBR overfit evaluation: load checkpoint, run inference on all training samples,
render pred vs GT side-by-side.

Usage:
  python eval_pbr.py \
      --config configs/pbr_overfit30.json \
      --ckpt_dir /cfs/yizhao/ckpt/pbr_overfit30 \
      --ckpt 10000 \
      --output_dir /cfs/yizhao/ckpt/pbr_overfit30/eval_all \
      --device cuda
"""
import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpainter import models, datasets
from gpainter.modules.sparse.basic import SparseTensor
from gpainter.pipelines import samplers
from gpainter.trainers.flow_matching.mixins.image_conditioned import (
    ImageConditionedMixin as ImageCondHelper,
)

# PBR constants (from pbr_matching.py / block_pbr_dense.py)
PBR_RES = 8
PBR_CHANNELS = 6
PBR_TOKEN_DIM = PBR_RES ** 3 * PBR_CHANNELS  # 3072
FIXED_DEFAULT_PBR = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], dtype=torch.float32)


def expand_mask(pbr_mask):
    """[T, 512] -> [T, 3072]"""
    return pbr_mask.unsqueeze(-1).expand(-1, -1, PBR_CHANNELS).reshape(pbr_mask.shape[0], -1)


def make_default_fill(T, device):
    single = FIXED_DEFAULT_PBR.to(device).repeat(PBR_RES ** 3)
    return single.unsqueeze(0).expand(T, -1)


def load_config(path):
    with open(path) as f:
        return edict(json.load(f))


def load_model(config_path, ckpt_dir, ckpt='latest', ema_rate=0.999, device='cuda'):
    cfg = load_config(config_path)
    model_cfg = cfg.models.denoiser
    model = getattr(models, model_cfg.name)(**model_cfg.args)

    if ckpt == 'latest':
        files = glob.glob(os.path.join(ckpt_dir, 'ckpts', 'misc_*.pt'))
        step = max(int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files)
    else:
        step = int(ckpt)

    if ema_rate:
        ckpt_name = f'denoiser_ema{ema_rate}_step{step:07d}.pt'
    else:
        ckpt_name = f'denoiser_step{step:07d}.pt'
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name)
    if not os.path.exists(ckpt_path):
        ckpt_name = f'denoiser_step{step:07d}.pt'
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name)

    print(f'Loading {ckpt_path}')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  Missing: {missing[:5]}')
    if unexpected:
        print(f'  Unexpected: {unexpected[:5]}')
    print(f'  Loaded {len(sd)} keys (step {step})')
    model.eval().to(device)
    return model, step


def init_image_encoder(device='cuda'):
    helper = ImageCondHelper(image_cond_model='dinov2_vitl14_reg')
    helper._init_image_cond_model()
    helper.image_cond_model['model'] = helper.image_cond_model['model'].to(device)
    return helper


@torch.no_grad()
def encode_image(helper, img_tensor, device='cuda'):
    """img_tensor: [B, 3, H, W] float [0,1]"""
    img = helper.image_cond_model['transform'](img_tensor.to(device))
    feats = helper.image_cond_model['model'](img, is_training=True)['x_prenorm']
    return F.layer_norm(feats, feats.shape[-1:])


@torch.no_grad()
def run_inference(model, data, image_encoder, device='cuda',
                  noise_scale=2.0, cfg_strength=1.5, cfg_interval=(0.0, 1.0),
                  steps=50):
    """Run PBR flow matching inference on a single batch."""
    for k, v in list(data.items()):
        if hasattr(v, 'cuda'):
            data[k] = v.to(device)

    x_0 = data['x_0']
    pbr_mask = data['pbr_mask'].to(device)
    fine_feats = data['fine_feats']

    voxel_mask = expand_mask(pbr_mask)
    default_fill = make_default_fill(x_0.feats.shape[0], device)

    # Noise
    noise_raw = noise_scale * torch.randn_like(x_0.feats)
    noise_raw = noise_raw * voxel_mask + (1.0 - voxel_mask) * default_fill
    noise = x_0.replace(noise_raw)

    # Condition
    cond_imgs = data['cond']  # [B, 3, H, W]
    cond_is_precomputed = data.get('cond_is_precomputed', None)
    if cond_is_precomputed is not None and all(cond_is_precomputed):
        cond = cond_imgs.to(device)
    else:
        cond = encode_image(image_encoder, cond_imgs, device)
    neg_cond = torch.zeros_like(cond)

    sampler = samplers.FlowGuidanceIntervalSampler()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        res = sampler.sample(
            model, noise=noise, cond=cond, neg_cond=neg_cond,
            submask=pbr_mask, voxel_mask=voxel_mask,
            default_fill=default_fill,
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
            steps=steps, use_heun=False, verbose=True,
        )
        pred = res.samples

    pred_feats = pred.feats * voxel_mask + (1.0 - voxel_mask) * default_fill
    gt_feats = x_0.feats * voxel_mask + (1.0 - voxel_mask) * default_fill

    return {
        'coords': x_0.coords.cpu(),
        'layout': x_0.layout,
        'pred_pbr': pred_feats.cpu().float().numpy(),
        'gt_pbr': gt_feats.cpu().float().numpy(),
        'fine_feats': fine_feats.cpu().float().numpy(),
        'pbr_mask': pbr_mask.cpu().numpy(),
    }


def render_sample(coords_np, fine_feats_np, pbr_np, output_path, device='cuda'):
    """Render PBR-shaded views for a single sample."""
    from gpainter.dataset_toolkits.mesh2pbrblock import extract_voxels, MC_THRESHOLD, SAMPLE_RES
    from gpainter.renderers import render_block_pbr_shaded
    import cubvh

    all_c, all_l = extract_voxels(coords_np, fine_feats_np)
    if len(all_c) == 0:
        return False

    v_mc, f_mc = cubvh.sparse_marching_cubes(all_c.cuda(), all_l.cuda(), MC_THRESHOLD)
    nf = f_mc.shape[0]

    MAX_FACES = 2000000
    TARGET_FACES = 500000
    if nf > MAX_FACES:
        import cumesh
        m = cumesh.CuMesh()
        m.init(v_mc.float(), f_mc.int())
        m.simplify(TARGET_FACES)
        v_mc, f_mc = m.read()

    verts = (v_mc.float() / SAMPLE_RES - 0.5).cpu().numpy()
    faces = f_mc.int().cpu().numpy()
    del v_mc, f_mc
    torch.cuda.empty_cache()

    # Upsample 8->16
    n = pbr_np.shape[0]
    t = torch.from_numpy(pbr_np.reshape(n, 8, 8, 8, 6)).permute(0, 4, 1, 2, 3).float()
    t16 = F.interpolate(t, size=16, mode='trilinear', align_corners=False)
    pbr_16 = t16.permute(0, 2, 3, 4, 1).numpy()

    views = render_block_pbr_shaded(verts, faces, coords_np, pbr_16,
                                     resolution=512, num_views=4, device=device)
    N, C, H, W = views.shape
    strip = torch.cat([views[i] for i in range(N)], dim=2)
    img = (strip.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path, quality=95)
    del views, strip
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg_strength', type=float, default=1.5)
    parser.add_argument('--noise_scale', type=float, default=2.0)
    parser.add_argument('--render_gpu', type=str, default='0',
                        help='GPU for rendering subprocess')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, step = load_model(args.config, args.ckpt_dir, args.ckpt, device=args.device)
    print(f'Model loaded at step {step}')

    # Load dataset
    cfg = load_config(args.config)
    ds_cfg = cfg.dataset
    ds_cls = getattr(datasets, ds_cfg.name)
    dataset = ds_cls(**ds_cfg.args)
    print(f'Dataset: {len(dataset)} samples')

    # Image encoder
    image_encoder = init_image_encoder(args.device)

    # Inference loop: one sample at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=getattr(dataset, 'collate_fn', None))

    details_dir = os.path.join(args.output_dir, 'details')
    os.makedirs(details_dir, exist_ok=True)

    all_results = []
    for idx, data in enumerate(dataloader):
        print(f'\n=== Sample {idx}/{len(dataset)} ===')

        result = run_inference(
            model, data, image_encoder, device=args.device,
            noise_scale=args.noise_scale, cfg_strength=args.cfg_strength,
            steps=args.steps,
        )

        sl = result['layout'][0]
        coords_np = result['coords'][sl][:, 1:].cpu().numpy().astype(np.int32)
        pred_pbr = np.clip(result['pred_pbr'][sl], 0, 1)
        gt_pbr = np.clip(result['gt_pbr'][sl], 0, 1)
        fine_np = result['fine_feats'][sl]
        mask_np = result['pbr_mask'][sl]

        # Save NPZ
        npz_path = os.path.join(details_dir, f'sample_{idx:03d}.npz')
        np.savez_compressed(npz_path,
            coords=coords_np,
            fine_feats=fine_np.astype(np.float16),
            pbr_mask=mask_np.astype(np.uint8),
            pred_pbr=pred_pbr.astype(np.float16),
            gt_pbr=gt_pbr.astype(np.float16),
        )

        # Render pred and gt
        for tag, pbr in [('pred', pred_pbr), ('gt', gt_pbr)]:
            out_path = os.path.join(details_dir, f'sample_{idx:03d}_{tag}.jpg')
            try:
                ok = render_sample(coords_np, fine_np, pbr, out_path, device=args.device)
                if ok:
                    print(f'  Rendered {tag}')
                else:
                    print(f'  Empty mesh for {tag}')
            except Exception as e:
                print(f'  Render failed ({tag}): {e}')
                torch.cuda.empty_cache()

        all_results.append(idx)

    # Compose overview grids
    num = len(all_results)
    for tag in ['pred', 'gt']:
        imgs = []
        for i in range(num):
            p = os.path.join(details_dir, f'sample_{i:03d}_{tag}.jpg')
            if os.path.exists(p):
                imgs.append(Image.open(p))
        if not imgs:
            continue
        w, h = imgs[0].size
        cols = min(4, len(imgs))
        rows = (len(imgs) + cols - 1) // cols
        grid = Image.new('RGB', (w * cols, h * rows), (30, 30, 30))
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            grid.paste(im, (c * w, r * h))
        grid.save(os.path.join(args.output_dir, f'overview_{tag}.jpg'), quality=90)
        print(f'Overview {tag}: {len(imgs)} samples')

    print(f'\nDone! Results in {args.output_dir}')


if __name__ == '__main__':
    main()
