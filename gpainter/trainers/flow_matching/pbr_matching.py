"""
PBR flow matching trainer v2 — aligned with GRAVER feats_matching methodology.

Design:
  - Input: coords [N,3], pbr_mask [N,512] (8³ surface mask), cond (DINOv2)
  - Output: pbr_feats [N,3072] = 8³×6ch PBR in [0,1]
  - Mask conditioning: pbr_mask fed to model as submask (like GRAVER Stage 3)
  - Loss: v-loss with surface_weight on surface voxels
  - Complexity weighting: per-block PBR color variance reweighting
  - TV regularization: spatial smoothness on predicted PBR
  - Snapshot: ODE sample → render block PBR on GT mesh
"""
from typing import *
import os
import copy
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...pipelines import samplers
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin

PBR_CHANNELS = 6
PBR_RES = 8
PBR_TOKEN_DIM = PBR_RES ** 3 * PBR_CHANNELS  # 3072


# Fixed default PBR: [bc(3), metal(1), rough(1), alpha(1)]
FIXED_DEFAULT_PBR = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], dtype=torch.float32)


class PBRFlowTrainer(FlowMatchingTrainer):
    """Flow matching trainer for dense block PBR prediction (v2: GRAVER-aligned)."""

    def __init__(
        self,
        *args,
        noise_scale: float = 2.0,
        surface_weight: float = 8.0,
        complexity_boost: float = 2.0,
        loss_type: str = "v_loss",
        cond_noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale
        self.surface_weight = surface_weight
        self.complexity_boost = complexity_boost
        self.loss_type = loss_type
        self.cond_noise_std = cond_noise_std

        print(f"[PBRFlowTrainer-v2] loss_type={loss_type}, "
              f"noise_scale={noise_scale}, surface_weight={surface_weight}, "
              f"complexity_boost={complexity_boost}")

    # ---- Dataloader ----

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu,
        )
        num_gpus = max(torch.cuda.device_count(), 1)
        cores_per_gpu = os.cpu_count() // num_gpus
        num_workers = max(1, min(cores_per_gpu, 16))

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split,
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    # ---- SparseTensor diffuse helpers ----

    @staticmethod
    def _expand_t(t, layout, T):
        counts = torch.tensor([sl.stop - sl.start for sl in layout],
                              device=t.device, dtype=torch.long)
        return t.repeat_interleave(counts).unsqueeze(1)

    def diffuse(self, x_0, t, noise=None):
        if isinstance(x_0, sp.SparseTensor):
            if noise is None:
                noise = x_0.replace(torch.randn_like(x_0.feats))
            t_tok = self._expand_t(t, x_0.layout, x_0.feats.shape[0])
            x_t = t_tok * x_0.feats + (1 - t_tok) * noise.feats
            return x_0.replace(x_t)
        return super().diffuse(x_0, t, noise=noise)

    # ---- Default PBR fill helpers ----

    @staticmethod
    def _make_default_fill(T, device):
        """Fixed default fill [T, 3072] — same value for all samples."""
        single = FIXED_DEFAULT_PBR.to(device).repeat(PBR_RES ** 3)  # [3072]
        return single.unsqueeze(0).expand(T, -1)  # [T, 3072]

    @staticmethod
    def _expand_mask(pbr_mask):
        """[T, 512] → [T, 3072] by repeating each voxel mask for 6 channels."""
        return pbr_mask.unsqueeze(-1).expand(-1, -1, PBR_CHANNELS).reshape(
            pbr_mask.shape[0], -1)

    # ---- Complexity reweighting (PBR-adapted from GRAVER feats) ----

    @torch.no_grad()
    def _pbr_complexity(
        self,
        gt_feats: torch.Tensor,
        pbr_mask: torch.Tensor,
        coords: torch.Tensor,
        layout: List[slice],
        boost: float,
    ) -> torch.Tensor:
        """
        Per-block complexity based on PBR color variance within surface voxels
        + neighborhood color gradient across adjacent blocks.

        Blocks with complex textures (patterns, color transitions) get higher weight.
        Output: [T] weights in [1, 1+boost].
        """
        T = gt_feats.shape[0]
        R = PBR_RES
        device = gt_feats.device

        # 1. Per-block color variance: reshape [T, 512*6] → [T, 512, 6], take RGB channels
        pbr_6ch = gt_feats.reshape(T, R**3, PBR_CHANNELS)  # [T, 512, 6]
        rgb = pbr_6ch[:, :, :3]                              # [T, 512, 3]
        mask_flat = pbr_mask                                  # [T, 512]

        # Weighted color variance on surface voxels
        surface_count = mask_flat.sum(dim=1).clamp(min=1)     # [T]
        rgb_mean = (rgb * mask_flat.unsqueeze(-1)).sum(dim=1) / surface_count.unsqueeze(-1)  # [T, 3]
        color_var = ((rgb - rgb_mean.unsqueeze(1)).pow(2) * mask_flat.unsqueeze(-1)).sum(dim=(1, 2))
        color_var = color_var / (surface_count * 3).clamp(min=1)  # [T]

        # 2. Neighborhood color gradient (6-connected, GPU vectorized)
        offsets = torch.tensor(
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
            device=device, dtype=torch.long,
        )

        complexity = color_var.clone()

        for sl in layout:
            bc = coords[sl][:, 1:].long()  # [N_b, 3]
            cv = color_var[sl]
            N_b = bc.shape[0]
            if N_b <= 1:
                continue

            P1, P2 = 100003, 1009
            keys = bc[:, 0] * P1 + bc[:, 1] * P2 + bc[:, 2]

            nb_coords = bc.unsqueeze(1) + offsets.unsqueeze(0)  # [N_b, 6, 3]
            nb_keys = nb_coords[:, :, 0] * P1 + nb_coords[:, :, 1] * P2 + nb_coords[:, :, 2]

            sorted_keys, sort_idx = keys.sort()
            flat_nb = nb_keys.reshape(-1)
            pos = torch.searchsorted(sorted_keys, flat_nb).clamp(max=N_b - 1)
            matched = sorted_keys[pos] == flat_nb
            nb_idx = sort_idx[pos].reshape(N_b, 6)
            valid = matched.reshape(N_b, 6)

            # Mean color per block: [N_b, 3]
            block_mean_rgb = (rgb[sl] * mask_flat[sl].unsqueeze(-1)).sum(dim=1) / surface_count[sl].unsqueeze(-1)

            nb_mean = block_mean_rgb[nb_idx]  # [N_b, 6, 3]
            color_diff = (block_mean_rgb.unsqueeze(1) - nb_mean).pow(2).sum(dim=-1)  # [N_b, 6]
            color_diff = color_diff * valid.float()

            n_valid = valid.float().sum(dim=1).clamp(min=1)
            mean_diff = color_diff.sum(dim=1) / n_valid

            # Combine: self variance + neighborhood gradient
            complexity[sl] = cv + mean_diff

        # Normalize → [1, 1+boost]
        cmax, cmin = complexity.max(), complexity.min()
        if cmax > cmin:
            complexity = (complexity - cmin) / (cmax - cmin)
        else:
            complexity.zero_()

        return 1.0 + boost * complexity

    # ---- Training losses ----

    def training_losses(self, x_0=None, cond=None, pbr_mask=None, **kwargs):
        assert x_0 is not None
        B = len(x_0.layout)
        T = x_0.feats.shape[0]
        device = x_0.device
        terms = edict()

        cond = self.get_cond(cond, **kwargs)
        if self.cond_noise_std > 0:
            cond = cond + self.cond_noise_std * torch.randn_like(cond)

        with torch.no_grad():
            pbr_mask = pbr_mask.to(device)
            voxel_mask = self._expand_mask(pbr_mask)         # [T, 3072]
            default_fill = self._make_default_fill(T, device)  # [T, 3072]

            # Mask GT: surface = GT, non-surface = default_pbr
            x_masked_feats = x_0.feats * voxel_mask + (1.0 - voxel_mask) * default_fill
            x_masked = x_0.replace(x_masked_feats)

            # Noise: surface = randn, non-surface = 0 (will be masked to default after diffuse)
            noise_raw = self.noise_scale * torch.randn_like(x_0.feats)
            noise_raw = noise_raw * voxel_mask  # non-surface noise = 0
            noise = x_0.replace(noise_raw)

            t = self.sample_t(B).to(device).float()
            x_t = self.diffuse(x_masked, t, noise=noise)

            # Enforce non-surface = default after diffusion
            x_t_feats = x_t.feats * voxel_mask + (1.0 - voxel_mask) * default_fill
            x_t = x_t.replace(x_t_feats)

        # Forward: model receives pbr_mask as submask conditioning
        pred = self.training_models["denoiser"](x_t, t, cond, submask=pbr_mask)

        # Loss computation
        x_residual = pred.feats - x_masked.feats
        t_tok = self._expand_t(t, x_0.layout, T)

        if self.loss_type == "v_loss":
            denom = (1 - t_tok).clamp(min=0.05)
            diff = (x_residual / denom) ** 2
        else:
            diff = x_residual ** 2

        # Mask: only compute loss on surface voxels
        diff = diff * voxel_mask

        # --- Per-block loss with surface weighting ---
        surface_count = voxel_mask.sum(dim=1).clamp(min=1)   # [T]
        block_loss = diff.sum(dim=1) / surface_count          # [T]
        block_loss = block_loss * self.surface_weight

        # --- Complexity reweighting (like GRAVER feats) ---
        with torch.no_grad():
            complexity_w = self._pbr_complexity(
                x_masked.feats, pbr_mask, x_0.coords, x_0.layout,
                self.complexity_boost,
            )
        block_loss = block_loss * complexity_w

        # Per-sample normalization
        per_sample = torch.stack([block_loss[sl].mean() for sl in x_0.layout])
        terms["flow_loss"] = per_sample.mean()
        terms["loss"] = terms["flow_loss"]
        terms["complexity_avg"] = complexity_w.mean()

        # Monitor by t bucket
        with torch.no_grad():
            batch_loss = per_sample
            for lo, hi in [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
                m = (t >= lo) & (t < hi)
                if m.any():
                    terms[f"loss_t{lo:.1f}_{hi:.1f}"] = batch_loss[m].mean()

        return terms, {}

    # ---- Snapshot: sample PBR → render on GT mesh ----

    @torch.no_grad()
    def snapshot(self, num_samples=4, batch_size=1, steps=50,
                 cfg_strength=1.5, cfg_interval=(0.0, 1.0), **kwargs):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

        snapshot_dir = os.path.join(self.output_dir, "samples", f"step{self.step:07d}")
        if self.is_master:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f"\n[PBR Snapshot] Step {self.step}: {num_samples} samples")

        if self.world_size > 1:
            dist.barrier()

        model_states = {n: m.training for n, m in self.models.items()}
        for m in self.models.values():
            m.eval()

        try:
            snap_dataset = (self.test_dataset if hasattr(self, 'test_dataset')
                            and self.test_dataset else copy.deepcopy(self.dataset))
            dataloader = DataLoader(
                snap_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, collate_fn=getattr(snap_dataset, 'collate_fn', None),
            )

            samples_per_rank = int(np.ceil(num_samples / self.world_size))
            my_start = self.rank * samples_per_rank
            my_end = min(my_start + samples_per_rank, num_samples)
            my_count = max(0, my_end - my_start)

            sampler = self.get_sampler()
            model = getattr(self.models["denoiser"], "module", self.models["denoiser"])
            use_amp = hasattr(self, 'accelerator') and self.accelerator.mixed_precision != 'no'

            details_dir = os.path.join(snapshot_dir, "details")
            os.makedirs(details_dir, exist_ok=True)

            data_iter = iter(dataloader)
            saved = 0

            while saved < my_count:
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    data = next(data_iter)

                for k, v in list(data.items()):
                    if hasattr(v, 'cuda'):
                        data[k] = v.cuda()

                x_0 = data.pop('x_0', None)
                pbr_mask = data.pop('pbr_mask', None).cuda()
                fine_feats = data.pop('fine_feats', None)
                data.pop('default_pbr', None)  # not used anymore

                voxel_mask = self._expand_mask(pbr_mask)
                default_fill = self._make_default_fill(x_0.feats.shape[0], x_0.device)

                # Initial noise: surface = randn, non-surface = 0
                # _apply_voxel_mask in sampler will fill non-surface with default_fill each step
                noise_raw = self.noise_scale * torch.randn_like(x_0.feats)
                noise_raw = noise_raw * voxel_mask  # non-surface = 0, will be filled by mask
                # Apply mask once to set initial non-surface = default
                noise_raw = noise_raw * voxel_mask + (1.0 - voxel_mask) * default_fill
                noise = x_0.replace(noise_raw)

                args = self.get_inference_cond(**data)
                args['submask'] = pbr_mask
                args['voxel_mask'] = voxel_mask
                args['default_fill'] = default_fill

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    res = sampler.sample(
                        model, noise=noise, **args,
                        steps=steps, cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        use_heun=False, verbose=False,
                    )
                    pred = res.samples

                # Post-process: non-surface = default
                pred_feats = pred.feats * voxel_mask + (1.0 - voxel_mask) * default_fill
                pred = pred.replace(pred_feats)

                actual_batch = len(x_0.layout)
                for b in range(min(actual_batch, my_count - saved)):
                    sl = x_0.layout[b]
                    global_idx = my_start + saved

                    # Extract per-block data
                    coords_np = x_0.coords[sl][:, 1:].cpu().numpy()   # [N, 3]
                    pred_pbr = pred.feats[sl].cpu().float().numpy()    # [N, 3072]
                    # Mask GT same as pred: non-surface = FIXED_DEFAULT_PBR
                    gt_feats_sl = x_0.feats[sl]
                    vm_sl = voxel_mask[sl]
                    df_sl = default_fill[sl]
                    gt_pbr = (gt_feats_sl * vm_sl + (1.0 - vm_sl) * df_sl).cpu().float().numpy()
                    gt_fine = fine_feats[sl].cpu().float().numpy()     # [N, 4096]
                    mask_np = pbr_mask[sl].cpu().numpy()               # [N, 512]

                    # Save NPZ (no GPU rendering — avoids CUDA crash)
                    npz_path = os.path.join(details_dir, f"sample_{global_idx:03d}.npz")
                    np.savez_compressed(npz_path,
                        coords=coords_np.astype(np.int32),
                        fine_feats=gt_fine.astype(np.float16),
                        pbr_mask=mask_np.astype(np.uint8),
                        pred_pbr=np.clip(pred_pbr, 0, 1).astype(np.float16),
                        gt_pbr=np.clip(gt_pbr, 0, 1).astype(np.float16),
                    )

                    saved += 1
                    print(f"  [Rank {self.rank}] Saved sample {saved}/{my_count}")

            if self.world_size > 1:
                dist.barrier()

            # Offline render in subprocess (won't crash training if it fails)
            if self.is_master:
                self._offline_render(snapshot_dir, num_samples)

        finally:
            for n, m in self.models.items():
                m.train(model_states[n])

    @staticmethod
    def _offline_render(snapshot_dir, num_samples):
        """Render saved NPZ samples in a subprocess — CUDA errors won't kill training."""
        import subprocess
        details_dir = os.path.join(snapshot_dir, "details")
        # Find project root dynamically (directory containing gpainter/)
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
        script = f'''
import sys, os, glob, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "{_project_root}")
from gpainter.dataset_toolkits.mesh2pbrblock import extract_voxels, MC_THRESHOLD, SAMPLE_RES
from gpainter.renderers import render_block_pbr_shaded
from PIL import Image
import cubvh, cumesh

MAX_RENDER_FACES = 2000000
TARGET_FACES = 500000

def upsample_8_to_16(flat):
    n = flat.shape[0]
    t = torch.from_numpy(flat.reshape(n,8,8,8,6)).permute(0,4,1,2,3).float()
    t16 = F.interpolate(t, size=16, mode='trilinear', align_corners=False)
    return t16.permute(0,2,3,4,1).numpy()

details = "{details_dir}"
for npz_path in sorted(glob.glob(os.path.join(details, "*.npz"))):
    try:
        d = np.load(npz_path)
        coords, fine, pred, gt = d["coords"], d["fine_feats"].astype(np.float32), d["pred_pbr"].astype(np.float32), d["gt_pbr"].astype(np.float32)
        mask = d["pbr_mask"]
        all_c, all_l = extract_voxels(coords, fine)
        if len(all_c) == 0: continue
        v_mc, f_mc = cubvh.sparse_marching_cubes(all_c.cuda(), all_l.cuda(), MC_THRESHOLD)
        nv, nf = v_mc.shape[0], f_mc.shape[0]
        del all_c, all_l
        if nf > MAX_RENDER_FACES:
            print(f"  Decimating {{nv}}v {{nf}}f -> {{TARGET_FACES}}f")
            m = cumesh.CuMesh(); m.init(v_mc.float(), f_mc.int()); m.simplify(TARGET_FACES)
            v_mc, f_mc = m.read()
            print(f"  After: {{v_mc.shape[0]}}v {{f_mc.shape[0]}}f")
        verts = (v_mc.float() / SAMPLE_RES - 0.5).cpu().numpy()
        faces = f_mc.int().cpu().numpy()
        del v_mc, f_mc; torch.cuda.empty_cache()
        base = os.path.splitext(os.path.basename(npz_path))[0]
        for tag, pbr in [("pred", upsample_8_to_16(pred)), ("gt", upsample_8_to_16(gt))]:
            views = render_block_pbr_shaded(verts, faces, coords, pbr, resolution=384, num_views=4, device="cuda")
            N, C, H, W = views.shape
            strip = torch.cat([views[i] for i in range(N)], dim=2)
            img = (strip.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(details, f"{{base}}_{{tag}}.jpg"), quality=95)
            del views, strip; torch.cuda.empty_cache()
        print(f"  Rendered {{base}}")
    except Exception as e:
        print(f"  Render fail {{npz_path}}: {{e}}")
        torch.cuda.empty_cache()

# Compose overview
for tag in ["pred", "gt"]:
    imgs = []
    for i in range({num_samples}):
        p = os.path.join(details, f"sample_{{i:03d}}_{{tag}}.jpg")
        if os.path.exists(p): imgs.append(Image.open(p))
    if not imgs: continue
    w, h = imgs[0].size
    cols = min(4, len(imgs)); rows = (len(imgs)+cols-1)//cols
    grid = Image.new("RGB", (w*cols, h*rows), (30,30,30))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); grid.paste(im, (c*w, r*h))
    grid.save(os.path.join("{snapshot_dir}", f"overview_{{tag}}.jpg"), quality=90)
    print(f"  Overview {{tag}}: {{len(imgs)}} samples")
'''
        try:
            import sys
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, '-c', script],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('RENDER_GPU', '5')},
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print(f"  [Render subprocess] exit={result.returncode}")
                if result.stderr:
                    print(result.stderr[-500:])
        except subprocess.TimeoutExpired:
            print("  [Render subprocess] timed out (300s)")
        except Exception as e:
            print(f"  [Render subprocess] error: {e}")


# ---- CFG / ImageConditioned variants ----

class PBRFlowCFGTrainer(ClassifierFreeGuidanceMixin, PBRFlowTrainer):
    pass


class ImageConditionedPBRFlowCFGTrainer(
    ImageConditionedMixin, PBRFlowCFGTrainer,
):
    def get_sampler(self, **kwargs):
        return samplers.FlowGuidanceIntervalSampler(**kwargs)
