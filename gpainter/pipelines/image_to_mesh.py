from typing import *
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import utils3d.torch
from easydict import EasyDict as edict
from PIL import Image

from .base import Pipeline
from . import samplers
from .. import models
from ..modules import sparse as sp
from ..dataset_toolkits.mesh2block import BLOCK_DIM, BLOCK_GRID, SUBMASK_RES
from ..datasets.block_feats import BlockFeats
from ..renderers import OctreeRenderer
from ..representations.octree import DfsOctree as Octree
from ..trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin


def _load_json(path: str) -> edict:
    with open(path, 'r') as f:
        return edict(json.load(f))


def _resolve_step(load_dir: str, ckpt: Union[str, int]) -> int:
    if isinstance(ckpt, int):
        return ckpt
    if ckpt == 'latest':
        files = glob.glob(os.path.join(load_dir, 'ckpts', 'misc_*.pt'))
        if not files:
            raise FileNotFoundError(f'No checkpoint found under {load_dir}/ckpts')
        return max(int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files)
    return int(ckpt)


def _resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


class GraverImageToMeshPipeline(Pipeline):
    def __init__(
        self,
        models: Optional[Dict[str, torch.nn.Module]] = None,
        samplers_dict: Optional[Dict[str, Any]] = None,
        stage_params: Optional[Dict[str, Dict[str, Any]]] = None,
        image_cond_model: str = 'dinov2_vitl14_reg',
        voxel_mode: bool = False,
    ):
        super().__init__(models)
        self.samplers = samplers_dict or {}
        self.stage_params = stage_params or {}
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None
        self.voxel_mode = voxel_mode  # True when using VoxelFlowModel

    @classmethod
    def from_config(cls, config_path: str) -> 'GraverImageToMeshPipeline':
        config_path = os.path.abspath(config_path)
        config_dir = os.path.dirname(config_path)
        cfg = _load_json(config_path)
        args = cfg.args

        built_models = {}
        built_samplers = {}
        stage_params = {}

        # Detect voxel mode: 'voxel' key replaces 'coords' + 'mask'
        voxel_mode = 'voxel' in args.stages

        if voxel_mode:
            stages_to_load = ['voxel', 'feats']
        else:
            stages_to_load = ['coords', 'mask', 'feats']

        for stage_name in stages_to_load:
            stage_cfg = edict(args.stages[stage_name])
            train_cfg_path = _resolve_path(config_dir, stage_cfg.config)
            train_cfg = _load_json(train_cfg_path)
            model_key = stage_cfg.get('model_key', 'denoiser')
            model_cfg = train_cfg.models[model_key]
            model = getattr(models, model_cfg.name)(**model_cfg.args)

            load_dir = _resolve_path(config_dir, stage_cfg.load_dir)
            if not os.path.isdir(load_dir):
                raise FileNotFoundError(f'Checkpoint directory not found: {load_dir}')
            step = _resolve_step(load_dir, stage_cfg.get('ckpt', 'latest'))
            ema_rate = stage_cfg.get('ema_rate', None)
            ckpt_name = f'{model_key}_step{step:07d}.pt'
            if ema_rate is not None:
                ckpt_name = f'{model_key}_ema{ema_rate}_step{step:07d}.pt'
            ckpt_path = os.path.join(load_dir, 'ckpts', ckpt_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'Model checkpoint not found: {ckpt_path}')

            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            built_models[f'{stage_name}_model'] = model

            sampler_cfg = edict(stage_cfg.sampler)
            built_samplers[stage_name] = getattr(samplers, sampler_cfg.name)(**sampler_cfg.get('args', {}))

            trainer_args = dict(train_cfg.trainer.args) if 'trainer' in train_cfg and 'args' in train_cfg.trainer else {}
            stage_params[stage_name] = {
                'noise_scale': stage_cfg.get('noise_scale', trainer_args.get('noise_scale', 1.0)),
                'sampler_params': dict(sampler_cfg.get('params', {})),
                'threshold': stage_cfg.get('threshold', 0.5),
            }

        pipeline = cls(
            models=built_models,
            samplers_dict=built_samplers,
            stage_params=stage_params,
            image_cond_model=args.get('image_cond_model', 'dinov2_vitl14_reg'),
            voxel_mode=voxel_mode,
        )
        pipeline._config = cfg
        return pipeline

    @classmethod
    def from_pretrained(cls, path: str) -> 'GraverImageToMeshPipeline':
        config_path = path
        if os.path.isdir(path):
            config_path = os.path.join(path, 'pipeline.json')
        return cls.from_config(config_path)

    def _init_image_cond_model(self):
        helper = ImageConditionedMixin(image_cond_model=self.image_cond_model_name)
        helper._init_image_cond_model()
        self.image_cond_model = helper.image_cond_model

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            return next(model.parameters()).device
        return torch.device('cpu')

    def to(self, device: Union[str, torch.device]) -> 'GraverImageToMeshPipeline':
        device = torch.device(device)
        for model in self.models.values():
            model.to(device)
        if self.image_cond_model is not None:
            self.image_cond_model['model'] = self.image_cond_model['model'].to(device)
        return self

    def preprocess_image(self, input_image: Image.Image) -> Image.Image:
        from .image_to_3d import GraverImageTo3DPipeline
        helper = GraverImageTo3DPipeline()
        helper.rembg_session = getattr(self, 'rembg_session', None)
        image = helper.preprocess_image(input_image)
        self.rembg_session = helper.rembg_session
        return image

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if self.image_cond_model is None:
            self._init_image_cond_model()

        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB')).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        if image.ndim != 4:
            raise ValueError('Image tensor must be [B, C, H, W].')

        image = image.to(self.device)
        image = self.image_cond_model['transform'](image)
        feats = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        return F.layer_norm(feats, feats.shape[-1:])

    @staticmethod
    def _dense_to_coords(occ: torch.Tensor, threshold: float) -> torch.Tensor:
        coords = torch.nonzero(occ[0, 0] > threshold, as_tuple=False).int()
        if coords.numel() == 0:
            raise RuntimeError('Stage 1 predicted no active blocks. Lower the threshold or check the checkpoint.')
        batch = torch.zeros(coords.shape[0], 1, device=coords.device, dtype=torch.int32)
        return torch.cat([batch, coords], dim=1)

    @staticmethod
    def _upsample_submask(submask: torch.Tensor) -> torch.Tensor:
        t = submask.shape[0]
        r = SUBMASK_RES
        scale = BLOCK_DIM // r
        sub_3d = submask.reshape(t, 1, r, r, r)
        voxel_3d = F.interpolate(sub_3d, scale_factor=scale, mode='nearest')
        return voxel_3d.reshape(t, -1)

    @staticmethod
    def _build_global_mask_volume(coords: torch.Tensor, submask: torch.Tensor) -> torch.Tensor:
        """Scatter per-block occ4 submask into a global support volume [1,1,256,256,256]."""
        res = BLOCK_GRID * SUBMASK_RES
        device = submask.device
        volume = torch.zeros(1, 1, res, res, res, device=device)
        coords_xyz = coords[:, 1:].long()
        submask_3d = submask.reshape(-1, SUBMASK_RES, SUBMASK_RES, SUBMASK_RES)
        for i in range(coords_xyz.shape[0]):
            x, y, z = coords_xyz[i]
            xs = x * SUBMASK_RES
            ys = y * SUBMASK_RES
            zs = z * SUBMASK_RES
            volume[0, 0, xs:xs + SUBMASK_RES, ys:ys + SUBMASK_RES, zs:zs + SUBMASK_RES] = torch.maximum(
                volume[0, 0, xs:xs + SUBMASK_RES, ys:ys + SUBMASK_RES, zs:zs + SUBMASK_RES],
                submask_3d[i],
            )
        return volume

    @staticmethod
    def _render_volume_vis(volume: torch.Tensor, path: str, resolution: int) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.pi / 8
        yaws = [y + yaws_offset for y in yaws]
        pitches = [-np.pi / 9, np.pi / 10, -np.pi / 12, np.pi / 8]

        exts, ints = [], []
        for yaw, pitch in zip(yaws, pitches):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ], device=volume.device).float() * 2
            fov = torch.deg2rad(torch.tensor(30.0, device=volume.device))
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.zeros(3, device=volume.device).float(),
                torch.tensor([0, 0, 1], device=volume.device).float(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        occ = (volume[0, 0] > 0.5)
        coords = torch.nonzero(occ, as_tuple=False)
        if coords.numel() == 0:
            blank = Image.new('RGB', (1024, 1024), (0, 0, 0))
            blank.save(path)
            return path

        representation = Octree(
            depth=10,
            aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
            device=volume.device,
            primitive='voxel',
            sh_degree=0,
            primitive_config={'solid': True},
        )
        representation.position = coords.float() / resolution
        representation.depth = torch.full(
            (representation.position.shape[0], 1),
            int(math.log2(resolution)),
            dtype=torch.uint8,
            device=volume.device,
        )

        image = torch.zeros(3, 1024, 1024, device=volume.device)
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
            r, c = divmod(j, 2)
            image[:, r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] = res['color']

        out = (image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(out).save(path)
        return path

    @staticmethod
    def _derive_output_paths(mesh_path: Optional[str]) -> Dict[str, Optional[str]]:
        if mesh_path is None:
            return {
                'mesh_path': None,
                'coords_vis_path': None,
                'mask_vis_path': None,
                'normal_path': None,
            }
        root, ext = os.path.splitext(mesh_path)
        if ext.lower() != '.ply':
            mesh_path = root + '.ply'
            root = root
        return {
            'mesh_path': mesh_path,
            'coords_vis_path': root + '_coords.jpg',
            'mask_vis_path': root + '_mask.jpg',
            'normal_path': root + '_normal.jpg',
        }

    @torch.no_grad()
    def sample_coords(self, cond: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.models['coords_model']
        params = self.stage_params['coords']
        noise = torch.randn(1, 1, BLOCK_GRID, BLOCK_GRID, BLOCK_GRID, device=self.device) * params['noise_scale']
        sampler_params = dict(params['sampler_params'])
        result = self.samplers['coords'].sample(
            model,
            noise=noise,
            cond=cond,
            neg_cond=torch.zeros_like(cond),
            verbose=verbose,
            **sampler_params,
        )
        occ = result.samples.clamp(0.0, 1.0)
        coords = self._dense_to_coords(occ, params['threshold'])
        return occ, coords

    @torch.no_grad()
    def sample_voxel(
        self, cond: torch.Tensor, verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VoxelFlowModel: ODE → 64³ occupancy, then extra forward → submask.
        Returns (occ, coords, submask).
        """
        model = self.models['voxel_model']
        params = self.stage_params['voxel']
        noise = torch.randn(1, 1, BLOCK_GRID, BLOCK_GRID, BLOCK_GRID, device=self.device) * params['noise_scale']
        sampler_params = dict(params['sampler_params'])

        # ODE: model returns only occupancy during sampling (return_submask=False)
        result = self.samplers['voxel'].sample(
            model,
            noise=noise,
            cond=cond,
            neg_cond=torch.zeros_like(cond),
            verbose=verbose,
            **sampler_params,
        )
        occ = result.samples.clamp(0.0, 1.0)
        coords = self._dense_to_coords(occ, params['threshold'])

        # Extra forward at t=1 with denoised occ → read submask head
        t_final = torch.ones(1, device=self.device)
        _, sm_feats, _ = model(occ, t_final, cond, return_submask=True)
        sm_pred = model.extract_submask_at_coords(sm_feats, coords)
        submask = (sm_pred > 0.0).float()   # threshold submask logits

        return occ, coords, submask

    @torch.no_grad()
    def sample_mask(self, cond: torch.Tensor, coords: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        model = self.models['mask_model']
        params = self.stage_params['mask']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], model.token_dim, device=self.device) * params['noise_scale'],
            coords=coords,
        )
        sampler_params = dict(params['sampler_params'])
        result = self.samplers['mask'].sample(
            model,
            noise=noise,
            cond=cond,
            neg_cond=torch.zeros_like(cond),
            verbose=verbose,
            **sampler_params,
        )
        return (result.samples.feats > 0.5).float()

    @torch.no_grad()
    def sample_feats(self, cond: torch.Tensor, coords: torch.Tensor, submask: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        model = self.models['feats_model']
        params = self.stage_params['feats']
        voxel_mask = self._upsample_submask(submask)

        noise_raw = torch.randn(coords.shape[0], model.token_dim, device=self.device) * params['noise_scale']
        noise_raw = noise_raw * voxel_mask + (1.0 - voxel_mask) * 1.0
        noise = sp.SparseTensor(feats=noise_raw, coords=coords)

        sampler_params = dict(params['sampler_params'])
        result = self.samplers['feats'].sample(
            model,
            noise=noise,
            cond=cond,
            neg_cond=torch.zeros_like(cond),
            submask=submask,
            voxel_mask=voxel_mask,
            verbose=verbose,
            **sampler_params,
        )
        pred = result.samples.feats * voxel_mask + (1.0 - voxel_mask) * 1.0
        return pred.clamp(0.0, 1.0)

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        *,
        preprocess: bool = True,
        mesh_path: Optional[str] = None,
        save_npz_path: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if preprocess:
            image = self.preprocess_image(image)
        cond = self.encode_image(image)

        if self.voxel_mode:
            # VoxelFlowModel: one model for both coords and submask
            occ, coords, submask = self.sample_voxel(cond, verbose=verbose)
        else:
            # Separate Stage 1 + Stage 2
            occ, coords = self.sample_coords(cond, verbose=verbose)
            submask = self.sample_mask(cond, coords, verbose=verbose)
        fine_feats = self.sample_feats(cond, coords, submask, verbose=verbose)

        output_paths = self._derive_output_paths(mesh_path)
        coords_vis_path = None
        mask_vis_path = None
        normal_path = None

        if output_paths['coords_vis_path'] is not None:
            coords_vis_path = self._render_volume_vis((occ > 0.5).float(), output_paths['coords_vis_path'], BLOCK_GRID)

        if output_paths['mask_vis_path'] is not None:
            global_mask = self._build_global_mask_volume(coords, submask)
            mask_vis_path = self._render_volume_vis(global_mask, output_paths['mask_vis_path'], BLOCK_GRID * SUBMASK_RES)

        coords_np = coords[:, 1:].detach().cpu().numpy().astype(np.int32)
        submask_np = submask.detach().cpu().numpy().astype(np.float32)
        fine_np = fine_feats.detach().cpu().numpy().astype(np.float32)

        if save_npz_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_npz_path)), exist_ok=True)
            np.savez_compressed(
                save_npz_path,
                coords=coords_np,
                submask=submask_np,
                fine_feats=fine_np,
            )

        mesh_ok = None
        if output_paths['mesh_path'] is not None:
            mesh_ok = BlockFeats.tokens_to_mesh(coords_np, fine_np, output_paths['mesh_path'], verbose=verbose)
            if mesh_ok:
                normal_path = output_paths['normal_path']
                BlockFeats.render_normal_grid(output_paths['mesh_path'], normal_path, resolution=1024, radius=1.75, verbose=verbose)

        return {
            'occupancy': occ.detach().cpu(),
            'coords': coords_np,
            'submask': submask_np,
            'fine_feats': fine_np,
            'mesh_path': output_paths['mesh_path'],
            'mesh_ok': mesh_ok,
            'coords_vis_path': coords_vis_path,
            'mask_vis_path': mask_vis_path,
            'normal_path': normal_path,
        }