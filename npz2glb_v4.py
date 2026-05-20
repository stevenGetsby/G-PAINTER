#!/usr/bin/env python3
"""Export eval PBR NPZ to GLB with render-like per-texel PBR baking.

This avoids the blocky artifacts from vertex-only PBR baking. For each UV texel,
it interpolates the 3D surface position and samples the dense block PBR volume
using the same trilinear lookup used by the renderer.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpainter.dataset_toolkits.mesh2pbrblock import extract_voxels, MC_THRESHOLD
from gpainter.dataset_toolkits.mesh2block import BLOCK_DIM, BLOCK_GRID, BLOCK_INNER, SAMPLE_RES

PBR_CHANNELS = 6
DEFAULT_PBR = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], dtype=torch.float32)


def upsample_pbr_8_to_16(pbr_raw):
    n_blocks = pbr_raw.shape[0]
    tensor = torch.from_numpy(pbr_raw.reshape(n_blocks, 8, 8, 8, 6)).permute(0, 4, 1, 2, 3).float()
    dense = F.interpolate(tensor, size=16, mode="trilinear", align_corners=False)
    return dense.permute(0, 2, 3, 4, 1).contiguous()


def build_block_lut(coords, device):
    coords_t = torch.from_numpy(coords.astype(np.int64)).to(device)
    lut = torch.full((BLOCK_GRID ** 3,), -1, device=device, dtype=torch.long)
    keys = coords_t[:, 0] * (BLOCK_GRID ** 2) + coords_t[:, 1] * BLOCK_GRID + coords_t[:, 2]
    lut[keys] = torch.arange(coords_t.shape[0], device=device)
    return lut


@torch.no_grad()
def sample_dense_pbr(positions_np, block_lut, dense_pbr, device="cuda", chunk_size=200000):
    if len(positions_np) == 0:
        return np.zeros((0, PBR_CHANNELS), dtype=np.float32)

    outputs = []
    default_pbr = DEFAULT_PBR.to(device)
    max_grid = SAMPLE_RES - 1e-6

    for start in range(0, len(positions_np), chunk_size):
        end = min(start + chunk_size, len(positions_np))
        pts = torch.from_numpy(positions_np[start:end]).float().to(device).clamp(-0.5, 0.5 - 1e-6)
        grid = ((pts + 0.5) * SAMPLE_RES).clamp(0.0, max_grid)

        block = torch.floor(grid / BLOCK_INNER).long().clamp(0, BLOCK_GRID - 1)
        local = grid - block.float() * BLOCK_INNER
        lower = torch.floor(local).long().clamp(0, BLOCK_DIM - 1)
        upper = (lower + 1).clamp(max=BLOCK_DIM - 1)
        weight = local - lower.float()

        block_key = block[:, 0] * (BLOCK_GRID ** 2) + block[:, 1] * BLOCK_GRID + block[:, 2]
        block_idx = block_lut[block_key]

        sampled = default_pbr.expand(end - start, -1).clone()
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

            sampled_valid = (
                c000 * wx0 * wy0 * wz0 + c001 * wx0 * wy0 * wz1
                + c010 * wx0 * wy1 * wz0 + c011 * wx0 * wy1 * wz1
                + c100 * wx1 * wy0 * wz0 + c101 * wx1 * wy0 * wz1
                + c110 * wx1 * wy1 * wz0 + c111 * wx1 * wy1 * wz1
            )
            sampled[valid] = sampled_valid

        outputs.append(sampled.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def make_mesh(coords, fine_feats, max_faces, device):
    import cubvh
    import cumesh
    import trimesh

    all_c, all_l = extract_voxels(coords, fine_feats)
    if len(all_c) == 0:
        raise RuntimeError("empty mesh")

    verts, faces = cubvh.sparse_marching_cubes(all_c.to(device), all_l.to(device), MC_THRESHOLD)
    verts = verts.float() / SAMPLE_RES - 0.5
    faces = faces.int()

    raw = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy().astype(np.int32), process=False)
    parts = raw.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(parts, key=lambda mesh: len(mesh.faces))
        verts = torch.from_numpy(largest.vertices.astype(np.float32)).to(device)
        faces = torch.from_numpy(largest.faces.astype(np.int32)).to(device)

    cu = cumesh.CuMesh()
    cu.init(verts, faces)
    if faces.shape[0] > max_faces:
        cu.simplify(max_faces)
    verts, faces = cu.read()
    return verts.cpu().numpy().astype(np.float32), faces.cpu().numpy().astype(np.int32)


def bake_texture(v_out, f_out, uv_out, coords, dense_pbr, tex_size, device, batch_faces=2000):
    block_lut = build_block_lut(coords, device)
    dense_pbr = dense_pbr.to(device).float()

    uv_px = uv_out * (tex_size - 1)
    texture = np.zeros((tex_size, tex_size, PBR_CHANNELS), dtype=np.float64)
    weight_sum = np.zeros((tex_size, tex_size), dtype=np.float64)

    for face_start in range(0, len(f_out), batch_faces):
        face_end = min(face_start + batch_faces, len(f_out))
        pix_x_parts, pix_y_parts, pos_parts = [], [], []

        for face in f_out[face_start:face_end]:
            i0, i1, i2 = face
            p0, p1, p2 = uv_px[i0], uv_px[i1], uv_px[i2]
            xmin = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
            xmax = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), tex_size - 1)
            ymin = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
            ymax = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), tex_size - 1)
            if xmin > xmax or ymin > ymax:
                continue

            xs = np.arange(xmin, xmax + 1) + 0.5
            ys = np.arange(ymin, ymax + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            px = gx.ravel()
            py = gy.ravel()

            denom = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if abs(denom) < 1e-10:
                continue
            inv = 1.0 / denom
            w0 = ((p1[0] - px) * (p2[1] - py) - (p2[0] - px) * (p1[1] - py)) * inv
            w1 = ((p2[0] - px) * (p0[1] - py) - (p0[0] - px) * (p2[1] - py)) * inv
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.001) & (w1 >= -0.001) & (w2 >= -0.001)
            if not inside.any():
                continue

            ix = gx.ravel()[inside].astype(np.int32)
            iy = gy.ravel()[inside].astype(np.int32)
            pos = (
                w0[inside, None] * v_out[i0]
                + w1[inside, None] * v_out[i1]
                + w2[inside, None] * v_out[i2]
            ).astype(np.float32)
            pix_x_parts.append(ix)
            pix_y_parts.append(iy)
            pos_parts.append(pos)

        if not pos_parts:
            continue

        ix_all = np.concatenate(pix_x_parts)
        iy_all = np.concatenate(pix_y_parts)
        pos_all = np.concatenate(pos_parts)
        vals = sample_dense_pbr(pos_all, block_lut, dense_pbr, device=device)
        np.add.at(texture, (iy_all, ix_all), vals)
        np.add.at(weight_sum, (iy_all, ix_all), 1.0)

        if face_start % (batch_faces * 10) == 0:
            filled = int((weight_sum > 0).sum())
            print(f"  bake faces {face_end}/{len(f_out)} filled={filled}/{tex_size * tex_size}", flush=True)

    mask = weight_sum > 0
    texture[mask] /= weight_sum[mask, None]
    return np.clip(texture, 0.0, 1.0), mask


def export_glb(v_out, f_out, uv_out, texture, mask, output_path):
    import trimesh

    mask_inv = (~mask).astype(np.uint8)
    base_color = np.clip(texture[:, :, :3] * 255, 0, 255).astype(np.uint8)
    metallic_tex = np.clip(texture[:, :, 3] * 255, 0, 255).astype(np.uint8)
    roughness_tex = np.clip(texture[:, :, 4] * 255, 0, 255).astype(np.uint8)
    alpha_tex = np.clip(texture[:, :, 5] * 255, 0, 255).astype(np.uint8)

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic_tex = cv2.inpaint(metallic_tex, mask_inv, 1, cv2.INPAINT_TELEA)
    roughness_tex = cv2.inpaint(roughness_tex, mask_inv, 1, cv2.INPAINT_TELEA)
    alpha_tex = cv2.inpaint(alpha_tex, mask_inv, 1, cv2.INPAINT_TELEA)

    alpha_mode = "BLEND" if (alpha_tex < 250).any() else "OPAQUE"
    base_img = Image.fromarray(np.concatenate([base_color, alpha_tex[:, :, None]], axis=-1))
    mr_img = Image.fromarray(np.stack([np.zeros_like(metallic_tex), roughness_tex, metallic_tex], axis=-1))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )

    uv_export = uv_out.copy()
    uv_export[:, 1] = 1.0 - uv_export[:, 1]

    normals = np.zeros_like(v_out)
    for face in f_out:
        i0, i1, i2 = face
        fn = np.cross(v_out[i1] - v_out[i0], v_out[i2] - v_out[i0])
        normals[i0] += fn
        normals[i1] += fn
        normals[i2] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    mesh = trimesh.Trimesh(
        vertices=v_out,
        faces=f_out,
        vertex_normals=normals,
        visual=trimesh.visual.TextureVisuals(uv=uv_export, material=material),
        process=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)


def convert_one(npz_path, output_path, key, tex_size, max_faces, device):
    import xatlas

    started = time.time()
    data = np.load(npz_path)
    coords = data["coords"].astype(np.int32)
    fine_feats = data["fine_feats"].astype(np.float32)
    pbr = data[key].astype(np.float32)
    dense_pbr = upsample_pbr_8_to_16(pbr)

    print(f"mesh {os.path.basename(npz_path)} {key}", flush=True)
    verts, faces = make_mesh(coords, fine_feats, max_faces=max_faces, device=device)
    print(f"  mesh verts={len(verts)} faces={len(faces)}", flush=True)

    vmapping, uv_faces, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    v_out = verts[vmapping]
    f_out = uv_faces.astype(np.int32)
    uv_out = uvs.astype(np.float32)
    print(f"  uv verts={len(v_out)} faces={len(f_out)}", flush=True)

    texture, mask = bake_texture(v_out, f_out, uv_out, coords, dense_pbr, tex_size, device=device)
    print(f"  filled={mask.sum()}/{tex_size * tex_size}", flush=True)
    export_glb(v_out, f_out, uv_out, texture, mask, output_path)
    print(f"  saved {output_path} ({time.time() - started:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--tex_size", type=int, default=1024)
    parser.add_argument("--max_faces", type=int, default=300000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_npz = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if args.samples:
        wanted = set(args.samples)
        all_npz = [p for p in all_npz if os.path.splitext(os.path.basename(p))[0] in wanted]
    print(f"Found {len(all_npz)} NPZ", flush=True)

    for npz_path in all_npz:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        for tag, key in [("pred", "pred_pbr"), ("gt", "gt_pbr")]:
            out = os.path.join(args.output_dir, f"{base}_{tag}.glb")
            convert_one(npz_path, out, key, args.tex_size, args.max_faces, args.device)


if __name__ == "__main__":
    main()
