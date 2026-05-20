import json
import os
import time
from typing import *
import numpy as np
import torch
from PIL import Image
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin, PrecomputedImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from ..dataset_toolkits.mesh2block import (
    BLOCK_GRID, BLOCK_INNER, BLOCK_DIM,
    SAMPLE_RES, MC_THRESHOLD,
    COL_PREFIX, BLOCK_FOLDER, SUBMASK_DIM,
)


class BlockFeats(StandardDatasetBase):

    def __init__(
        self,
        roots: str,
        *,
        max_block_num: int = 15000,
        min_block_num: int = 0,
        min_aesthetic_score: float = 5.0,
        max_samples: int = 0,
    ):
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.min_aesthetic_score = min_aesthetic_score
        self.max_samples = max_samples

        super().__init__(roots)
        self.filter_existing_instances(
            lambda root, instance: os.path.exists(os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')),
            stat_name='npz exists',
        )

        if self.max_samples > 0 and len(self.instances) > self.max_samples:
            self.instances = self.instances[:self.max_samples]
            self.metadata = self.metadata.iloc[:self.max_samples]

        self.loads = [
            max(1, int(self.metadata.at[i, f'{COL_PREFIX}_num_blocks']))
            for i in range(len(self.instances))
        ]
        if self.max_samples > 0:
            print(f'  [Dataset] max_samples={self.max_samples}, actual={len(self.instances)}')

    def filter_metadata(self, metadata):
        stats = {}

        metadata = metadata[metadata[f'{COL_PREFIX}_block_status'] == "success"]
        stats['block successed:'] = len(metadata)

        metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] <= self.max_block_num]
        stats[f'block num <= {self.max_block_num}:'] = len(metadata)

        if self.min_block_num > 0:
            metadata = metadata[metadata[f'{COL_PREFIX}_num_blocks'] >= self.min_block_num]
            stats[f'block num >= {self.min_block_num}:'] = len(metadata)

        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, 'render', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0

        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

    def _get_geo(self, root, instance):
        verts, face = utils3d.io.read_ply(os.path.join(root, 'render', instance, 'mesh.ply'))
        mesh = {
            "vertices": torch.from_numpy(verts),
            "faces": torch.from_numpy(face),
        }
        return mesh

    def get_instance(self, root, instance):
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')

        with np.load(npz_path) as data:
            coords = torch.from_numpy(data['coords']).int()
            raw = data['fine_feats']
            if raw.dtype == np.float16:
                fine_feats = torch.from_numpy(raw.astype(np.float32))
            else:
                fine_feats = torch.from_numpy(raw).float()

            # per-block sub-mask (兼容旧数据: 没有则全 1)
            if 'submask' in data.files:
                submask = torch.from_numpy(data['submask'].astype(np.float32))
            else:
                submask = torch.ones(coords.shape[0], SUBMASK_DIM)

        return {
            'coords': coords,
            'fine_feats': fine_feats,
            'submask': submask,
        }

    # ------------------------------------------------------------------
    # Snapshot 可视化: tokens → mesh → 法线图
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_laplacian_smooth(vertices, faces, iterations=1, lam=0.5):
        """GPU Laplacian 平滑 (uniform weights)."""
        v = vertices.clone()
        f = faces.long()
        for _ in range(iterations):
            e01 = torch.stack([f[:, 0], f[:, 1]], dim=1)
            e12 = torch.stack([f[:, 1], f[:, 2]], dim=1)
            e20 = torch.stack([f[:, 2], f[:, 0]], dim=1)
            edges = torch.cat([e01, e12, e20, e01.flip(1), e12.flip(1), e20.flip(1)], dim=0)
            src, dst = edges[:, 0], edges[:, 1]
            neighbor_sum = torch.zeros_like(v)
            neighbor_cnt = torch.zeros(v.shape[0], 1, device=v.device)
            neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), v[src])
            neighbor_cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=v.device))
            neighbor_cnt = neighbor_cnt.clamp(min=1)
            v = v + lam * (neighbor_sum / neighbor_cnt - v)
        return v, faces

    @staticmethod
    def _gpu_bilateral_smooth(vertices, faces, iterations=2, lam=0.4, sigma_n=0.3):
        """
        GPU Bilateral Mesh Smoothing — 在去噪的同时保留锐利边缘.

        原理: 在 Laplacian 位移上乘以法线相似性权重.
        法线相似 → 大权重 → 正常平滑 (去噪)
        法线不同 → 小权重 → 几乎不动 (保边)

        sigma_n: 法线差异敏感度. 越小越保边 (推荐 0.2~0.5)
        """
        v = vertices.clone()
        f = faces.long()

        # 构建边连接 (一次性)
        e01 = torch.stack([f[:, 0], f[:, 1]], dim=1)
        e12 = torch.stack([f[:, 1], f[:, 2]], dim=1)
        e20 = torch.stack([f[:, 2], f[:, 0]], dim=1)
        edges = torch.cat([e01, e12, e20, e01.flip(1), e12.flip(1), e20.flip(1)], dim=0)
        src, dst = edges[:, 0], edges[:, 1]

        for _ in range(iterations):
            # 计算顶点法线 (面积加权)
            v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            fn = torch.cross(v1 - v0, v2 - v0, dim=1)
            vn = torch.zeros_like(v)
            for i in range(3):
                vn.scatter_add_(0, f[:, i].unsqueeze(1).expand(-1, 3), fn)
            vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-8)

            # 法线相似性 bilateral weight
            n_dot = (vn[src] * vn[dst]).sum(dim=1)           # cos similarity [-1, 1]
            w = torch.exp((n_dot - 1.0) / (sigma_n ** 2))    # 法线越近 w→1, 差别大 w→0

            # 加权 Laplacian
            weighted_sum = torch.zeros_like(v)
            weight_total = torch.zeros(v.shape[0], 1, device=v.device)
            weighted_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), v[src] * w.unsqueeze(1))
            weight_total.scatter_add_(0, dst.unsqueeze(1), w.unsqueeze(1))
            weight_total = weight_total.clamp(min=1e-8)

            v = v + lam * (weighted_sum / weight_total - v)

        return v, faces

    @staticmethod
    def tokens_to_mesh(coords_np, tokens_np, output_path, verbose=True):
        """
        将 block tokens 重建为 mesh 并保存 — 全 GPU 流水线.
        coords_np: [N, 3] or [N, 4] block 坐标 (int)
        tokens_np: [N, BLOCK_DIM³] UDF 值 (float, 已 clip 到 [0,1])
        output_path: 保存路径 (.ply)
        """
        import cubvh
        import cumesh
        import trimesh

        device = torch.device("cuda")
        t0 = time.time()

        # ---- 全部搬上 GPU ----
        coords = torch.as_tensor(coords_np, dtype=torch.long, device=device)
        if coords.shape[1] == 4:
            coords = coords[:, 1:]

        n = coords.shape[0]
        F_full = torch.as_tensor(tokens_np, dtype=torch.float32, device=device).reshape(
            n, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM,
        )
        F_core = F_full.contiguous()

        # ---- 边界角点平均 (GPU scatter) ----
        max_dim = BLOCK_GRID * BLOCK_INNER + BLOCK_DIM
        S1, S2 = max_dim * max_dim, max_dim

        ci_arr = torch.arange(BLOCK_DIM, device=device)
        ci, cj, ck = torch.meshgrid(ci_arr, ci_arr, ci_arr, indexing="ij")
        local_key = (ci * S1 + cj * S2 + ck).reshape(-1)

        bx = coords[:, 0] * BLOCK_INNER
        by = coords[:, 1] * BLOCK_INNER
        bz = coords[:, 2] * BLOCK_INNER
        base_key = bx * S1 + by * S2 + bz

        flat_key = (base_key[:, None] + local_key[None, :]).reshape(-1)
        flat_val = F_core.reshape(-1)

        uniq, inverse = torch.unique(flat_key, return_inverse=True)
        num_uniq = uniq.shape[0]
        val_sum = torch.zeros(num_uniq, dtype=torch.float64, device=device)
        val_cnt = torch.zeros(num_uniq, dtype=torch.float64, device=device)
        val_sum.scatter_add_(0, inverse, flat_val.double())
        val_cnt.scatter_add_(0, inverse, torch.ones_like(flat_val, dtype=torch.float64))
        val_avg = (val_sum / val_cnt).float()

        F_core = val_avg[inverse].reshape(n, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)
        del flat_key, flat_val, val_sum, val_cnt, val_avg, inverse, uniq, F_full

        # ---- 提取 8 角点, 筛选有效 voxel (GPU) ----
        vx, vy, vz = torch.meshgrid(
            torch.arange(BLOCK_INNER, device=device),
            torch.arange(BLOCK_INNER, device=device),
            torch.arange(BLOCK_INNER, device=device),
            indexing="ij",
        )
        local_vox = torch.stack([vx, vy, vz], dim=-1).reshape(-1, 3)
        lx, ly, lz = local_vox[:, 0], local_vox[:, 1], local_vox[:, 2]

        all_logits = torch.stack([
            F_core[:, lx,   ly,   lz  ], F_core[:, lx+1, ly,   lz  ],
            F_core[:, lx+1, ly+1, lz  ], F_core[:, lx,   ly+1, lz  ],
            F_core[:, lx,   ly,   lz+1], F_core[:, lx+1, ly,   lz+1],
            F_core[:, lx+1, ly+1, lz+1], F_core[:, lx,   ly+1, lz+1],
        ], dim=-1)

        all_global = coords[:, None, :] * BLOCK_INNER + local_vox[None, :, :]

        mc_thr = 1.0 * MC_THRESHOLD

        # ---- UDF Sharpening: 加陡表面处梯度 → MC 提取更锐利 ----
        # power > 1 把靠近 0 的值压得更低, 靠近 mc_thr 的值不变
        # 效果: 表面处零交叉变陡 → 三角面更贴合真实表面
        SHARPEN_ALPHA = 1.5
        normalized = (all_logits / mc_thr).clamp(min=0)
        all_logits = mc_thr * torch.where(
            normalized <= 1.0,
            normalized.pow(SHARPEN_ALPHA),
            1.0 + (normalized - 1.0) * SHARPEN_ALPHA,
        )

        valid = (torch.isfinite(all_logits).all(dim=-1)
                 & (all_logits.min(dim=-1).values < (mc_thr + 0.03)))

        valid_coords = all_global[valid].long()
        valid_logits = all_logits[valid].float()
        del all_logits, all_global, F_core
        torch.cuda.empty_cache()

        if valid_coords.shape[0] == 0:
            if verbose:
                print(f"  [tokens_to_mesh] No valid voxels for {output_path}")
            return False

        # ---- Marching Cubes (GPU) ----
        t_mc = time.time()
        try:
            v, f = cubvh.sparse_marching_cubes(valid_coords, valid_logits, mc_thr)
        except Exception as e:
            if verbose:
                print(f"  [tokens_to_mesh] MC error: {e}")
            del valid_coords, valid_logits
            torch.cuda.empty_cache()
            return False
        del valid_coords, valid_logits
        if verbose:
            print(f"  MC: {time.time()-t_mc:.2f}s, {f.shape[0]:,} faces")

        v = v.float() / SAMPLE_RES - 0.5

        # ---- CuMesh GPU 清理 + 减面 ----
        t_clean = time.time()
        cm = cumesh.CuMesh()
        cm.init(v.contiguous(), f.int().contiguous())
        del v, f
        cm.remove_duplicate_faces()
        cm.remove_degenerate_faces()
        cm.remove_small_connected_components(min_area=1e-6)

        TARGET_FACES = 5_000_000
        cur_faces = cm.num_faces
        if cur_faces > TARGET_FACES:
            cm.simplify(TARGET_FACES, verbose=False)
            if verbose:
                print(f"  Decimated: {cur_faces:,} -> {cm.num_faces:,} ({time.time()-t_clean:.2f}s)")

        new_v, new_f = cm.read()

        # ---- GPU Bilateral smoothing (保边去噪) ----
        new_v, new_f = BlockFeats._gpu_bilateral_smooth(new_v, new_f, iterations=3, lam=0.4, sigma_n=0.3)

        # ---- 导出 (唯一 CPU 步骤: I/O) ----
        mesh_out = trimesh.Trimesh(
            vertices=new_v.cpu().numpy(),
            faces=new_f.cpu().numpy(),
            process=False,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mesh_out.export(output_path)
        if verbose:
            print(f"  Total: {time.time()-t0:.2f}s -> {output_path}")
        return True

    @staticmethod
    def render_normal_grid(ply_path, output_path, resolution=512, radius=1.75, verbose=True):
        """
        用 nvdiffrast 渲染 4 视角法线图拼成 2x2 grid.
        """
        from ..renderers.normal_render import render_random_normals_grid
        try:
            render_random_normals_grid(
                ply_path, output_path,
                resolution=resolution, radius=radius,
                target_faces=5000000,
                verbose=verbose,
            )
            return True
        except Exception as e:
            if verbose:
                print(f"  [render] error: {e}")
            return False

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            coords_list = []
            fine_list = []
            submask_list = []
            layout = []
            start = 0

            for i, b in enumerate(sub_batch):
                n_blocks = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n_blocks, 1), i, dtype=torch.int32),
                    b['coords'],
                ], dim=-1))
                fine_list.append(b['fine_feats'])
                if 'submask' in b:
                    submask_list.append(b['submask'])
                layout.append(slice(start, start + n_blocks))
                start += n_blocks

            coords = torch.cat(coords_list, dim=0)
            fine_feats = torch.cat(fine_list, dim=0)

            pack['x_f'] = SparseTensor(
                coords=coords,
                feats=fine_feats,
                shape=torch.Size([len(group), fine_feats.shape[1]]),
                layout=layout,
            )

            # per-block sub-mask: [T, 8], 和 feats 同维度对齐
            if submask_list:
                pack['submask'] = torch.cat(submask_list, dim=0)

            exclude_keys = {'coords', 'fine_feats', 'submask'}
            other_keys = [k for k in sub_batch[0].keys() if k not in exclude_keys]

            for k in other_keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class TextConditionedBlockFeats(TextConditionedMixin, BlockFeats):
    """Text conditioned block feats dataset"""
    pass


class ImageConditionedBlockFeats(ImageConditionedMixin, BlockFeats):
    """Image conditioned block feats dataset"""
    pass


class PrecomputedImageConditionedBlockFeats(PrecomputedImageConditionedMixin, BlockFeats):
    """Image conditioned block feats dataset with precomputed DINOv2 features"""
    pass
