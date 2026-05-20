#!/usr/bin/env python3
"""
Convert eval NPZ (pred/gt PBR) → GLB meshes with PBR vertex colors.

Usage:
  python npz2glb.py \
      --input_dir /cfs/yizhao/ckpt/pbr_overfit30/eval_all/details \
      --output_dir /cfs/yizhao/ckpt/pbr_overfit30/eval_all/glb \
      --device cuda
"""
import argparse
import glob
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpainter.dataset_toolkits.mesh2pbrblock import extract_voxels, MC_THRESHOLD, SAMPLE_RES


def npz_to_glb(npz_path, output_path, pbr_key='pred_pbr', device='cuda', max_faces=300000):
    """NPZ with coords + fine_feats + pred/gt pbr → vertex-color GLB."""
    import cubvh
    import cumesh
    import trimesh

    data = np.load(npz_path)
    coords = data['coords']
    fine_feats = data['fine_feats'].astype(np.float32)
    pbr_raw = data[pbr_key].astype(np.float32)

    n_blocks = len(coords)

    # Upsample 8³→16³ PBR
    t = torch.from_numpy(pbr_raw.reshape(n_blocks, 8, 8, 8, 6)).permute(0, 4, 1, 2, 3).float()
    t16 = F.interpolate(t, size=16, mode='trilinear', align_corners=False)
    pbr_16 = t16.permute(0, 2, 3, 4, 1)  # [N, 16, 16, 16, 6] tensor

    # Marching cubes
    all_c, all_l = extract_voxels(coords, fine_feats)
    if len(all_c) == 0:
        print(f"  Empty mesh: {npz_path}")
        return False

    v_mc, f_mc = cubvh.sparse_marching_cubes(all_c.to(device), all_l.to(device), MC_THRESHOLD)
    v_mc = v_mc.float() / SAMPLE_RES - 0.5
    f_mc = f_mc.int()

    # Remove small fragments
    v_np_raw = v_mc.cpu().numpy()
    f_np_raw = f_mc.cpu().numpy().astype(np.int32)
    raw_mesh = trimesh.Trimesh(vertices=v_np_raw, faces=f_np_raw, process=False)
    components = raw_mesh.split(only_watertight=False)
    if len(components) > 1:
        largest = max(components, key=lambda m: len(m.faces))
        v_mc = torch.from_numpy(largest.vertices.astype(np.float32)).to(device)
        f_mc = torch.from_numpy(largest.faces.astype(np.int32)).to(device)

    # Simplify
    cu = cumesh.CuMesh()
    cu.init(v_mc, f_mc)
    if f_mc.shape[0] > max_faces:
        cu.simplify(max_faces)
    v_s, f_s = cu.read()
    v_np = v_s.cpu().numpy().astype(np.float32)
    f_np = f_s.cpu().numpy().astype(np.int32)

    # Sample PBR at vertices using the same trilinear logic as the renderer
    from gpainter.dataset_toolkits.mesh2block import BLOCK_GRID, BLOCK_INNER, BLOCK_DIM
    SAMPLE_RES_LOCAL = BLOCK_GRID * BLOCK_INNER  # 960

    dense_pbr = pbr_16.to(device)  # [N, 16, 16, 16, 6]
    coords_t = torch.from_numpy(coords).long().to(device)

    # Build block LUT (same as renderer)
    block_lut = torch.full((BLOCK_GRID ** 3,), -1, device=device, dtype=torch.long)
    block_keys = coords_t[:, 0] * (BLOCK_GRID ** 2) + coords_t[:, 1] * BLOCK_GRID + coords_t[:, 2]
    block_lut[block_keys] = torch.arange(coords_t.shape[0], device=device)
    default_pbr = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32)

    # Trilinear sample at vertices (same as _sample_dense_block_pbr)
    pts = torch.from_numpy(v_np).float().to(device).clamp(-0.5, 0.5 - 1e-6)
    max_grid = SAMPLE_RES_LOCAL - 1e-6
    grid = ((pts + 0.5) * SAMPLE_RES_LOCAL).clamp(0.0, max_grid)

    block = torch.floor(grid / BLOCK_INNER).long().clamp(0, BLOCK_GRID - 1)
    local = grid - block.float() * BLOCK_INNER
    lower = torch.floor(local).long().clamp(0, BLOCK_DIM - 1)
    upper = (lower + 1).clamp(max=BLOCK_DIM - 1)
    weight = local - lower.float()

    block_key = block[:, 0] * (BLOCK_GRID ** 2) + block[:, 1] * BLOCK_GRID + block[:, 2]
    block_idx = block_lut[block_key]

    v_pbr = default_pbr.expand(len(v_np), -1).clone()
    valid = block_idx >= 0
    if valid.any():
        bid = block_idx[valid]
        lo = lower[valid]
        hi = upper[valid]
        wx, wy, wz = weight[valid].unbind(dim=1)

        def corner(ix, iy, iz):
            return dense_pbr[bid, ix, iy, iz]

        c000 = corner(lo[:, 0], lo[:, 1], lo[:, 2])
        c001 = corner(lo[:, 0], lo[:, 1], hi[:, 2])
        c010 = corner(lo[:, 0], hi[:, 1], lo[:, 2])
        c011 = corner(lo[:, 0], hi[:, 1], hi[:, 2])
        c100 = corner(hi[:, 0], lo[:, 1], lo[:, 2])
        c101 = corner(hi[:, 0], lo[:, 1], hi[:, 2])
        c110 = corner(hi[:, 0], hi[:, 1], lo[:, 2])
        c111 = corner(hi[:, 0], hi[:, 1], hi[:, 2])

        wx0 = (1.0 - wx).unsqueeze(1)
        wy0 = (1.0 - wy).unsqueeze(1)
        wz0 = (1.0 - wz).unsqueeze(1)
        wx1 = wx.unsqueeze(1)
        wy1 = wy.unsqueeze(1)
        wz1 = wz.unsqueeze(1)

        sampled = (
            c000 * wx0 * wy0 * wz0 + c001 * wx0 * wy0 * wz1
            + c010 * wx0 * wy1 * wz0 + c011 * wx0 * wy1 * wz1
            + c100 * wx1 * wy0 * wz0 + c101 * wx1 * wy0 * wz1
            + c110 * wx1 * wy1 * wz0 + c111 * wx1 * wy1 * wz1
        )
        v_pbr[valid] = sampled

    v_pbr = v_pbr.cpu().numpy()
    v_pbr = np.clip(v_pbr, 0, 1)

    # Vertex colors: use albedo (first 3 channels), apply gamma decode → simple shading → gamma encode
    # to match the rendered appearance more closely
    albedo = v_pbr[:, :3]

    # Vertex colors (RGBA)
    vc = np.ones((len(v_np), 4), dtype=np.uint8) * 255
    vc[:, :3] = (albedo * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, vertex_colors=vc, process=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_faces', type=int, default=300000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npz')))
    print(f'Found {len(npz_files)} NPZ files')

    for npz_path in npz_files:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        print(f'\n=== {base} ===')

        # Pred GLB
        pred_path = os.path.join(args.output_dir, f'{base}_pred.glb')
        try:
            ok = npz_to_glb(npz_path, pred_path, pbr_key='pred_pbr',
                           device=args.device, max_faces=args.max_faces)
            if ok:
                print(f'  pred -> {pred_path}')
        except Exception as e:
            print(f'  pred FAILED: {e}')

        # GT GLB
        gt_path = os.path.join(args.output_dir, f'{base}_gt.glb')
        try:
            ok = npz_to_glb(npz_path, gt_path, pbr_key='gt_pbr',
                           device=args.device, max_faces=args.max_faces)
            if ok:
                print(f'  gt -> {gt_path}')
        except Exception as e:
            print(f'  gt FAILED: {e}')

    print(f'\nDone! GLBs in {args.output_dir}')


if __name__ == '__main__':
    main()
