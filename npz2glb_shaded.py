#!/usr/bin/env python3
"""Export eval PBR NPZ to visually baked GLB.

This path is meant for inspection/viewer parity with eval render images. It
bakes the same studio GGX/ACES/gamma look into baseColor and marks the material
as KHR_materials_unlit so generic GLB viewers do not relight the asset.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npz2glb_v4 import (  # noqa: E402
    PBR_CHANNELS,
    build_block_lut,
    make_mesh,
    sample_dense_pbr,
    upsample_pbr_8_to_16,
)
from gpainter.renderers.pbr_renderer import (  # noqa: E402
    _AMBIENT,
    _aces,
    _get_mv,
    _ggx_brdf,
    _studio_lights,
)

GLB_JSON = 0x4E4F534A
UNLIT_EXT = "KHR_materials_unlit"


def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


@torch.no_grad()
def shade_pbr(pbr_np, normal_np, pos_np, device, view_mode, elevation, azimuth, radius):
    if len(pbr_np) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pbr = torch.from_numpy(pbr_np).float().to(device)
    normal = F.normalize(torch.from_numpy(normal_np).float().to(device), dim=-1).unsqueeze(1)
    pos = torch.from_numpy(pos_np).float().to(device)

    albedo = pbr[:, :3].clamp(0, 1).pow(2.2).unsqueeze(1)
    metallic = pbr[:, 3:4].clamp(0, 1).unsqueeze(1)
    roughness = pbr[:, 4:5].clamp(0.04, 1).unsqueeze(1)

    if view_mode == "camera":
        mv = torch.from_numpy(_get_mv(elevation, azimuth, radius)).float().to(device)
        cam_pos = -mv[:3, :3].T @ mv[:3, 3]
        view_dir = F.normalize(cam_pos.unsqueeze(0) - pos, dim=-1).unsqueeze(1)
    else:
        view_dir = normal

    ndotv = (normal * view_dir).sum(-1, keepdim=True)
    normal = torch.where(ndotv < 0, -normal, normal)

    color = torch.zeros_like(albedo)
    for light_dir, light_col in _studio_lights(device):
        color = color + _ggx_brdf(normal, view_dir, light_dir, albedo, roughness, metallic, light_col)
    color = color + _AMBIENT * albedo
    color = _aces(color).clamp(0, 1).pow(1.0 / 2.2)
    return color.squeeze(1).cpu().numpy().astype(np.float32)


def bake_shaded_texture(
    v_out,
    f_out,
    n_out,
    uv_out,
    coords,
    dense_pbr,
    tex_size,
    device,
    view_mode,
    elevation,
    azimuth,
    radius,
    batch_faces=2000,
):
    block_lut = build_block_lut(coords, device)
    dense_pbr = dense_pbr.to(device).float()

    uv_px = uv_out * (tex_size - 1)
    color_tex = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    alpha_tex = np.zeros((tex_size, tex_size), dtype=np.float64)
    weight_sum = np.zeros((tex_size, tex_size), dtype=np.float64)

    for face_start in range(0, len(f_out), batch_faces):
        face_end = min(face_start + batch_faces, len(f_out))
        pix_x_parts, pix_y_parts, pos_parts, normal_parts = [], [], [], []

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

            w0i = w0[inside]
            w1i = w1[inside]
            w2i = w2[inside]
            ix = gx.ravel()[inside].astype(np.int32)
            iy = gy.ravel()[inside].astype(np.int32)
            pos = (w0i[:, None] * v_out[i0] + w1i[:, None] * v_out[i1] + w2i[:, None] * v_out[i2]).astype(np.float32)
            normal = (w0i[:, None] * n_out[i0] + w1i[:, None] * n_out[i1] + w2i[:, None] * n_out[i2]).astype(np.float32)

            pix_x_parts.append(ix)
            pix_y_parts.append(iy)
            pos_parts.append(pos)
            normal_parts.append(normal)

        if not pos_parts:
            continue

        ix_all = np.concatenate(pix_x_parts)
        iy_all = np.concatenate(pix_y_parts)
        pos_all = np.concatenate(pos_parts)
        normal_all = np.concatenate(normal_parts)
        normal_all = normal_all / np.maximum(np.linalg.norm(normal_all, axis=1, keepdims=True), 1e-8)

        pbr = sample_dense_pbr(pos_all, block_lut, dense_pbr, device=device)
        shaded = shade_pbr(pbr, normal_all, pos_all, device, view_mode, elevation, azimuth, radius)

        for channel in range(3):
            np.add.at(color_tex[:, :, channel], (iy_all, ix_all), shaded[:, channel])
        np.add.at(alpha_tex, (iy_all, ix_all), pbr[:, 5].clip(0, 1))
        np.add.at(weight_sum, (iy_all, ix_all), 1.0)

        if face_start % (batch_faces * 10) == 0:
            filled = int((weight_sum > 0).sum())
            print(f"  shaded bake faces {face_end}/{len(f_out)} filled={filled}/{tex_size * tex_size}", flush=True)

    mask = weight_sum > 0
    color_tex[mask] /= weight_sum[mask, None]
    alpha_tex[mask] /= weight_sum[mask]
    return np.clip(color_tex, 0.0, 1.0), np.clip(alpha_tex, 0.0, 1.0), mask


def patch_glb_unlit(path):
    with open(path, "rb") as file:
        magic, version, _ = struct.unpack("<4sII", file.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"not a GLB2 file: {path}")
        chunks = []
        while True:
            header = file.read(8)
            if not header:
                break
            chunk_len, chunk_type = struct.unpack("<II", header)
            chunks.append((chunk_type, file.read(chunk_len)))

    if not chunks or chunks[0][0] != GLB_JSON:
        raise ValueError(f"first GLB chunk is not JSON: {path}")

    gltf = json.loads(chunks[0][1].decode("utf-8").rstrip(" \t\r\n\x00"))
    used = gltf.setdefault("extensionsUsed", [])
    if UNLIT_EXT not in used:
        used.append(UNLIT_EXT)
    for material in gltf.get("materials", []):
        material.setdefault("extensions", {})[UNLIT_EXT] = {}
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    new_chunks = [(GLB_JSON, json_bytes)] + chunks[1:]
    total_len = 12 + sum(8 + len(data) for _, data in new_chunks)

    with open(path, "wb") as file:
        file.write(struct.pack("<4sII", b"glTF", 2, total_len))
        for chunk_type, data in new_chunks:
            file.write(struct.pack("<II", len(data), chunk_type))
            file.write(data)


def export_shaded_glb(v_out, f_out, uv_out, shaded, alpha, mask, output_path):
    import trimesh

    mask_inv = (~mask).astype(np.uint8)
    base_color = np.clip(shaded * 255, 0, 255).astype(np.uint8)
    alpha_tex = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    alpha_tex = cv2.inpaint(alpha_tex, mask_inv, 1, cv2.INPAINT_TELEA)

    base_img = Image.fromarray(np.concatenate([base_color, alpha_tex[:, :, None]], axis=-1))
    alpha_mode = "BLEND" if (alpha_tex < 250).any() else "OPAQUE"
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )

    uv_export = uv_out.copy()
    uv_export[:, 1] = 1.0 - uv_export[:, 1]
    mesh = trimesh.Trimesh(
        vertices=v_out,
        faces=f_out,
        visual=trimesh.visual.TextureVisuals(uv=uv_export, material=material),
        process=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)
    patch_glb_unlit(output_path)


def convert_one(npz_path, output_path, key, tex_size, max_faces, device, view_mode, elevation, azimuth, radius):
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
    normals = compute_vertex_normals(verts, faces)

    vmapping, uv_faces, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    v_out = verts[vmapping]
    n_out = normals[vmapping]
    f_out = uv_faces.astype(np.int32)
    uv_out = uvs.astype(np.float32)
    print(f"  uv verts={len(v_out)} faces={len(f_out)}", flush=True)

    shaded, alpha, mask = bake_shaded_texture(
        v_out,
        f_out,
        n_out,
        uv_out,
        coords,
        dense_pbr,
        tex_size,
        device=device,
        view_mode=view_mode,
        elevation=elevation,
        azimuth=azimuth,
        radius=radius,
    )
    print(f"  filled={mask.sum()}/{tex_size * tex_size}", flush=True)
    export_shaded_glb(v_out, f_out, uv_out, shaded, alpha, mask, output_path)
    print(f"  saved {output_path} ({time.time() - started:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--tex_size", type=int, default=1024)
    parser.add_argument("--max_faces", type=int, default=300000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keys", nargs="*", default=["pred"], choices=["pred", "gt"])
    parser.add_argument("--view_mode", default="camera", choices=["normal", "camera"])
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_npz = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if args.samples:
        wanted = set(args.samples)
        all_npz = [path for path in all_npz if os.path.splitext(os.path.basename(path))[0] in wanted]
    print(f"Found {len(all_npz)} NPZ", flush=True)

    key_map = {"pred": "pred_pbr", "gt": "gt_pbr"}
    for npz_path in all_npz:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        for tag in args.keys:
            out = os.path.join(args.output_dir, f"{base}_{tag}_shaded.glb")
            convert_one(
                npz_path,
                out,
                key_map[tag],
                args.tex_size,
                args.max_faces,
                args.device,
                args.view_mode,
                args.elevation,
                args.azimuth,
                args.radius,
            )


if __name__ == "__main__":
    main()