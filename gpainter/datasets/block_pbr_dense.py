"""
G-PAINTER dataset: 8³ block PBR for sparse flow matching.

NPZ fields used:
    coords       [N, 3]    int32    — block coordinates in 64³ grid
    fine_feats   [N, 4096] float16  — UDF per block (for snapshot mesh)
    pbr_8_feats  [N, 3072] uint8    — 8³×6 PBR, quantized [0,255]
    submask      [N, 512]  uint8    — 8³ surface mask (from UDF thresholding)
"""
import os
import functools
import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .components import StandardDatasetBase, ImageConditionedMixin

PBR_RES = 8
PBR_CHANNELS = 6
PBR_TOKEN_DIM = PBR_RES ** 3 * PBR_CHANNELS   # 3072

# Folder name under each data root
BLOCK_FOLDER = "pbr_blocks_dense"


def _filter_meta(metadata, max_block_num, min_block_num):
    stats = {}
    col = 'pbr_dense_num_blocks'
    if col not in metadata.columns:
        col = 'num_blocks'
    if col not in metadata.columns:
        # no block count column — return all
        stats['all'] = len(metadata)
        return metadata, stats
    metadata = metadata[metadata[col] <= max_block_num]
    stats[f'blocks<={max_block_num}'] = len(metadata)
    if min_block_num > 0:
        metadata = metadata[metadata[col] >= min_block_num]
        stats[f'blocks>={min_block_num}'] = len(metadata)
    return metadata, stats


class BlockPBRDense(StandardDatasetBase):
    """Dense 16³×6 PBR block dataset for flow matching training.
    Uses metadata.csv (sha256 as instance key) + pbr_blocks_dense/ NPZ."""

    def __init__(self, roots, *, max_block_num=15000, min_block_num=0,
                 max_samples=0, **kwargs):
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.max_samples = max_samples
        super().__init__(roots, **kwargs)
        # Only keep instances that have a PBR NPZ
        self.filter_existing_instances(
            lambda root, inst: os.path.exists(
                os.path.join(root, BLOCK_FOLDER, f'{inst}.npz')),
            stat_name='pbr_npz_exists',
        )
        if self.max_samples > 0 and len(self.instances) > self.max_samples:
            self.instances = self.instances[:self.max_samples]
            self.metadata = self.metadata.iloc[:self.max_samples]
        # Loads for balanced sampling
        self.loads = [1] * len(self.instances)
        print(f"[BlockPBRDense] {len(self.instances)} instances")

    def filter_metadata(self, metadata):
        return _filter_meta(metadata, self.max_block_num, self.min_block_num)

    def get_instance(self, root, instance):
        npz_path = os.path.join(root, BLOCK_FOLDER, f'{instance}.npz')
        with np.load(npz_path) as data:
            coords = torch.from_numpy(data['coords'].astype(np.int32))           # [N, 3]
            fine_feats = torch.from_numpy(data['fine_feats'].astype(np.float32))  # [N, 4096]

            # 8³ PBR
            raw = data['pbr_8_feats']                                              # [N, 3072] uint8
            pbr = torch.from_numpy(raw.astype(np.float32))
            if raw.dtype == np.uint8:
                pbr = pbr / 255.0
            pbr = pbr.reshape(-1, PBR_TOKEN_DIM)                                  # [N, 3072]

            # 8³ mask (submask ≡ pbr_8_mask, use submask directly)
            pbr_mask = torch.from_numpy(data['submask'].astype(np.float32))        # [N, 512]

        return {
            'coords': coords,
            'fine_feats': fine_feats,
            'pbr_mask': pbr_mask,        # [N, 512]  — 8³
            'pbr_feats': pbr,            # [N, 3072] — 8³×6
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            groups = [list(range(len(batch)))]
        else:
            groups = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size,
            )
        packs = []
        for group in groups:
            sub = [batch[i] for i in group]
            coords_list, feats_list, mask_list, fine_list, layout = [], [], [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([
                    torch.full((n, 1), i, dtype=torch.int32), b['coords'],
                ], dim=-1))
                feats_list.append(b['pbr_feats'])
                mask_list.append(b['pbr_mask'])
                fine_list.append(b['fine_feats'])
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list, 0)
            feats = torch.cat(feats_list, 0)
            masks = torch.cat(mask_list, 0)
            fines = torch.cat(fine_list, 0)

            pack = {
                'x_0': SparseTensor(
                    coords=coords, feats=feats,
                    shape=torch.Size([len(group), PBR_TOKEN_DIM]),
                    layout=layout,
                ),
                'pbr_mask': masks,          # [T, 512] — 8³
                'fine_feats': fines,        # [T, 4096] — UDF, for snapshot mesh
            }

            # Forward per-sample keys (cond images, etc.)
            skip = {'coords', 'pbr_feats', 'pbr_mask', 'fine_feats'}
            for k in sub[0]:
                if k in skip:
                    continue
                if isinstance(sub[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub])
                elif isinstance(sub[0][k], list):
                    pack[k] = sum([b[k] for b in sub], [])
                else:
                    pack[k] = [b[k] for b in sub]
            packs.append(pack)

        return packs[0] if split_size is None else packs


class ImageConditionedBlockPBRDense(ImageConditionedMixin, BlockPBRDense):
    pass
