#!/usr/bin/env python3
"""
Convert eval NPZ (pred/gt PBR) → GLB meshes with full PBR textures.

Uses pbr2mesh: marching cubes → simplify → xatlas UV → texture bake → GLB
with baseColor + metallicRoughness + alpha textures.

Usage:
  python npz2glb.py \
      --input_dir /cfs/yizhao/ckpt/pbr_overfit30/eval_all/details \
      --output_dir /cfs/yizhao/ckpt/pbr_overfit30/eval_all/glb
"""
import argparse
import glob
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpainter.dataset_toolkits.mesh2pbrblock import pbr2mesh


def convert_eval_npz(npz_path, output_path, pbr_key='pred_pbr',
                     tex_size=1024, max_faces=300000, verbose=True):
    """Convert eval NPZ to GLB via pbr2mesh with full PBR textures."""
    data = np.load(npz_path)
    coords = data['coords']
    fine_feats = data['fine_feats'].astype(np.float32)
    pbr_raw = data[pbr_key].astype(np.float32)

    # pbr2mesh expects 'pbr_feats' — save temp NPZ with correct field name
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
        np.savez_compressed(tmp_path,
            coords=coords.astype(np.int32),
            fine_feats=fine_feats.astype(np.float32),
            pbr_feats=pbr_raw.astype(np.float32),
            submask=data['pbr_mask'].astype(np.float32) if 'pbr_mask' in data.files
                    else np.ones((len(coords), 512), dtype=np.float32),
        )

    try:
        result = pbr2mesh(tmp_path, output_path,
                         tex_size=tex_size, max_faces=max_faces,
                         fast=False, verbose=verbose)
        return result is not None
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tex_size', type=int, default=1024)
    parser.add_argument('--max_faces', type=int, default=300000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npz')))
    print(f'Found {len(npz_files)} NPZ files', flush=True)

    for npz_path in npz_files:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        print(f'\n{"="*60}\n{base}\n{"="*60}', flush=True)

        for tag, key in [('pred', 'pred_pbr'), ('gt', 'gt_pbr')]:
            out_path = os.path.join(args.output_dir, f'{base}_{tag}.glb')
            try:
                ok = convert_eval_npz(npz_path, out_path, pbr_key=key,
                                      tex_size=args.tex_size, max_faces=args.max_faces)
                if ok:
                    print(f'  {tag} OK', flush=True)
            except Exception as e:
                print(f'  {tag} FAILED: {e}', flush=True)

    print(f'\nDone! GLBs in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
