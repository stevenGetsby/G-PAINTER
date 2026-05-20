from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str or list): paths to the dataset.
            - str: single path or comma-separated paths
            - list: list of paths
    """

    def __init__(self,
        roots: Union[str, List[str]],
    ):
        super().__init__()
        if isinstance(roots, list):
            self.roots = roots
        else:
            self.roots = [r.strip() for r in roots.split(',') if r.strip()]
        self.instances = []
        all_metadata = []

        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            all_metadata.append(metadata)

        # 整数 index, 与 self.instances 一一对齐 (不去重, 支持不同 root 同 sha256 不同数据)
        self.metadata = pd.concat(all_metadata, ignore_index=True)

    def filter_existing_instances(self, exists_fn, stat_name: str = 'Existing files'):
        keep_indices = []
        counts = {os.path.basename(root): 0 for root in self.roots}
        for i, (root, instance) in enumerate(self.instances):
            if exists_fn(root, instance):
                keep_indices.append(i)
                counts[os.path.basename(root)] += 1

        if len(keep_indices) == len(self.instances):
            for key, count in counts.items():
                self._stats[key][stat_name] = count
            return

        self.instances = [self.instances[i] for i in keep_indices]
        self.metadata = self.metadata.iloc[keep_indices].reset_index(drop=True)
        for key, count in counts.items():
            self._stats[key][stat_name] = count

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for i, (root, sha256) in enumerate(self.instances):
            self.captions[(root, sha256)] = json.loads(self.metadata.at[i, 'captions'])

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[(root, instance)])
        pack['cond'] = text
        return pack


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        # Fix: Use astype(bool) to avoid FutureWarning
        metadata['cond_rendered'] = metadata['cond_rendered'].fillna(False).astype(bool)
        metadata = metadata[metadata['cond_rendered']]

        stats['Cond rendered'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        nz = np.array(alpha).nonzero()
        if nz[0].size == 0:
            # alpha 全透明: 用整张图
            h, w = alpha.shape
            bbox = [0, 0, w - 1, h - 1]
        else:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image

        return pack


class PrecomputedImageConditionedMixin:
    """加载离线编码的 DINOv2 特征, 跳过在线图片加载和模型前向.

    目录结构:
      {root}/dinov2_cond/{sha256}/{view:03d}.pt  →  [1369, 1024] float32

    训练时随机选一个视角的 .pt 文件加载, 返回 pack['cond'] = [1369, 1024].
    标记 pack['cond_is_precomputed'] = True, 让 trainer 跳过 encode_image.
    """
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size       # 保持接口一致
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        # 需要 cond_rendered (原始图片存在) 且 dinov2_cond_encoded
        metadata['cond_rendered'] = metadata['cond_rendered'].fillna(False).astype(bool)
        metadata = metadata[metadata['cond_rendered']]
        stats['Cond rendered'] = len(metadata)

        if 'dinov2_cond_encoded' in metadata.columns:
            metadata['dinov2_cond_encoded'] = metadata['dinov2_cond_encoded'].fillna(False).astype(bool)
            metadata = metadata[metadata['dinov2_cond_encoded']]
            stats['DINOv2 encoded'] = len(metadata)

        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        feat_dir = os.path.join(root, 'dinov2_cond', instance)
        pt_files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.pt')])
        assert len(pt_files) > 0, f"No .pt files in {feat_dir}"

        view = np.random.randint(len(pt_files))
        feat_path = os.path.join(feat_dir, pt_files[view])
        cond_feat = torch.load(feat_path, map_location='cpu', weights_only=True)  # [1369, 1024]

        pack['cond'] = cond_feat
        pack['cond_is_precomputed'] = True
        return pack
