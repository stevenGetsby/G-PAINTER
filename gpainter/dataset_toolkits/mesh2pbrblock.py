# mesh2pbrblock: GLB → dense 16³ block PBR (encode / decode / roundtrip).
# Channel order (TRELLIS.2): base_color(3) | metallic(1) | roughness(1) | alpha(1).

import argparse
import hashlib
import os
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from gpainter.renderers import render_block_pbr_shaded, render_block_pbr_views, render_mesh_textured, render_pbr_views
    from gpainter.dataset_toolkits.mesh2block import (
        BLOCK_DIM,
        BLOCK_GRID,
        BLOCK_INNER,
        MAX_QUERY_PTS,
        MC_THRESHOLD,
        PBR_CHANNELS,
        SAMPLE_RES,
        SUBMASK_DIM,
        SUBMASK_RES,
        SUBMASK_STRIDE,
        SURFACE_THRESHOLD,
        TRUNCATION,
        VOXEL_SIZE,
    )
else:
    from ..renderers import render_block_pbr_shaded, render_block_pbr_views, render_mesh_textured, render_pbr_views
    from .mesh2block import (
        BLOCK_DIM,
        BLOCK_GRID,
        BLOCK_INNER,
        MAX_QUERY_PTS,
        MC_THRESHOLD,
        PBR_CHANNELS,
        SAMPLE_RES,
        SUBMASK_DIM,
        SUBMASK_RES,
        SUBMASK_STRIDE,
        SURFACE_THRESHOLD,
        TRUNCATION,
        VOXEL_SIZE,
    )



# ---- mesh loading, PBR sampling, decode (inlined from mesh2pbr) ----

# ===================== Texture sampling =====================

def _bilinear_sample_np(tex_img, uv):
    """PIL Image + [N,2] UV → [N,C] float32.  Handles REPEAT wrap."""
    arr = np.array(tex_img).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    if arr.ndim == 2:
        arr = arr[:, :, None]

    # REPEAT wrap: fmod then clamp
    u = (uv[:, 0] % 1.0) * (W - 1)
    v = ((1.0 - uv[:, 1]) % 1.0) * (H - 1)

    u0 = np.floor(u).astype(int).clip(0, W - 1)
    v0 = np.floor(v).astype(int).clip(0, H - 1)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    fu = (u - u0)[:, None]
    fv = (v - v0)[:, None]

    return (arr[v0, u0] * (1 - fu) * (1 - fv) +
            arr[v0, u1] * fu * (1 - fv) +
            arr[v1, u0] * (1 - fu) * fv +
            arr[v1, u1] * fu * fv).astype(np.float32)


def sample_pbr_at_surface(pts, mesh_data, bvh, faces_t, device):
    """
    Per-point PBR sampling with per-submesh material handling.
    BVH → face_id → submesh → UV interpolation → texture sample.
    Uses per-point UV interpolation for maximum texture detail.
    """
    N = pts.shape[0]
    pbr = np.zeros((N, 6), dtype=np.float32)
    pbr[:, :3] = 0.5; pbr[:, 3] = 0.5; pbr[:, 5] = 1.0

    dist, fids_raw, uvw = bvh.unsigned_distance(pts, return_uvw=True)
    fids = fids_raw.long().cpu().numpy()
    bary = uvw.cpu().numpy()
    n_faces = len(mesh_data['faces'])
    fids = np.clip(fids, 0, n_faces - 1)

    face_uvs = mesh_data.get('face_uvs')        # [F, 3, 2]
    face_submesh = mesh_data.get('face_submesh') # [F] int
    submeshes = mesh_data.get('submeshes', [])   # list of dicts

    if face_uvs is not None and submeshes:
        # Per-point UV interpolation (preserves texture detail)
        fuv = face_uvs[fids]  # [N, 3, 2]
        iuv = (bary[:, 0:1] * fuv[:, 0] +
               bary[:, 1:2] * fuv[:, 1] +
               bary[:, 2:3] * fuv[:, 2])  # [N, 2]

        sub_ids = face_submesh[fids]  # [N]
        for si, sm in enumerate(submeshes):
            mask = sub_ids == si
            if not mask.any():
                continue
            uv_sel = iuv[mask]
            if sm.get('albedo_tex') is not None:
                vals = _bilinear_sample_np(sm['albedo_tex'], uv_sel)
                pbr[mask, :3] = vals[:, :3] * sm['bc_factor'][None, :]
            else:
                pbr[mask, :3] = sm['bc_factor'][None, :]
            if sm.get('rough_tex') is not None:
                vals = _bilinear_sample_np(sm['rough_tex'], uv_sel)
                pbr[mask, 3] = vals[:, 0] * sm['r_factor']
            else:
                pbr[mask, 3] = sm['r_factor']
            if sm.get('metal_tex') is not None:
                vals = _bilinear_sample_np(sm['metal_tex'], uv_sel)
                pbr[mask, 4] = vals[:, 0] * sm['m_factor']
            else:
                pbr[mask, 4] = sm['m_factor']
            if sm.get('alpha_tex') is not None:
                vals = _bilinear_sample_np(sm['alpha_tex'], uv_sel)
                pbr[mask, 5] = vals[:, 0] * sm['a_factor']
            else:
                pbr[mask, 5] = sm['a_factor']

    return pbr


# ===================== Mesh loading =====================

def load_mesh(path, verbose=True):
    """
    Load mesh with per-submesh materials preserved.
    Stores: per-face UV, per-face submesh ID, per-submesh textures.
    Enables per-point UV interpolation at sample time.
    """
    import trimesh

    scene = trimesh.load(path, process=False, force='scene')
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry={'main': scene})

    all_verts, all_faces, all_face_uvs = [], [], []
    all_face_submesh = []
    submeshes = []
    vert_offset = 0

    # Traverse scene graph to get correct transforms for each geometry instance
    # scene.graph.to_flattened() gives {node_name: {'geometry': geom_name, 'matrix': 4x4}}
    instances = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        instances.append((node_name, geom_name, transform, geom))

    if not instances:
        # Fallback: iterate geometry directly
        for name, geom in scene.geometry.items():
            instances.append((name, name, np.eye(4), geom))

    skipped_geometries = []
    for _, (node_name, geom_name, tf, geom) in enumerate(instances):
        if not isinstance(geom, trimesh.Trimesh) or not hasattr(geom, 'faces'):
            skipped_geometries.append((geom_name, type(geom).__name__))
            continue

        v = geom.vertices.astype(np.float32)
        # Apply scene graph transform
        v_h = np.hstack([v, np.ones((len(v), 1), dtype=np.float32)])
        v = (v_h @ tf.T)[:, :3].astype(np.float32)

        f_local = geom.faces.astype(np.int32)
        n_f = len(f_local)
        if len(v) == 0 or n_f == 0:
            skipped_geometries.append((geom_name, 'empty_mesh'))
            continue
        submesh_id = len(submeshes)

        # UV
        uv = getattr(geom.visual, 'uv', None)
        if uv is not None:
            fuv = uv[f_local].astype(np.float32)  # [F_local, 3, 2]
        else:
            fuv = np.zeros((n_f, 3, 2), dtype=np.float32)

        # Material
        mat = getattr(geom.visual, 'material', None)
        sm = {'bc_factor': np.array([1., 1., 1.], dtype=np.float32),
              'r_factor': 0.5, 'm_factor': 0.0, 'a_factor': 1.0,
              'alpha_mode': 'OPAQUE', 'alpha_cutoff': 0.5, 'double_sided': True,
              'albedo_tex': None, 'rough_tex': None, 'metal_tex': None, 'alpha_tex': None}

        if mat is not None:
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                t = mat.baseColorTexture
                sm['albedo_tex'] = t.convert('RGB') if isinstance(t, Image.Image) else Image.fromarray(np.array(t)[:, :, :3])
                try:
                    if isinstance(t, Image.Image) and t.mode == 'RGBA':
                        sm['alpha_tex'] = t.getchannel(3)
                    elif not isinstance(t, Image.Image) and np.array(t).shape[2] >= 4:
                        sm['alpha_tex'] = Image.fromarray(np.array(t)[:, :, 3])
                except Exception:
                    pass
            if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None:
                fct = np.array(mat.baseColorFactor, dtype=np.float32)
                # Normalize 0-255 to 0-1 if needed
                if fct.max() > 1.0:
                    fct = fct / 255.0
                if len(fct) >= 3: sm['bc_factor'] = fct[:3]
                if len(fct) >= 4: sm['a_factor'] = float(fct[3])
            if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                mr_img = mat.metallicRoughnessTexture
                # Convert palette/grayscale to RGB
                if isinstance(mr_img, Image.Image):
                    mr_img = mr_img.convert('RGB')
                mr_arr = np.array(mr_img)
                if mr_arr.ndim == 3 and mr_arr.shape[2] >= 2:
                    sm['rough_tex'] = Image.fromarray(mr_arr[:, :, 1])
                    sm['metal_tex'] = Image.fromarray(mr_arr[:, :, 2] if mr_arr.shape[2] >= 3 else mr_arr[:, :, 0])
                    sm['r_factor'], sm['m_factor'] = 1.0, 1.0
            if hasattr(mat, 'roughnessFactor') and mat.roughnessFactor is not None:
                sm['r_factor'] = float(mat.roughnessFactor)
            if hasattr(mat, 'metallicFactor') and mat.metallicFactor is not None:
                sm['m_factor'] = float(mat.metallicFactor)
            if hasattr(mat, 'alphaMode') and mat.alphaMode is not None:
                sm['alpha_mode'] = str(mat.alphaMode).upper()
            if hasattr(mat, 'alphaCutoff') and mat.alphaCutoff is not None:
                sm['alpha_cutoff'] = float(mat.alphaCutoff)
            if hasattr(mat, 'doubleSided') and mat.doubleSided is not None:
                sm['double_sided'] = bool(mat.doubleSided)

        # Pre-bake per-vertex PBR by sampling texture at each vertex UV
        # This avoids UV seam issues during BVH point sampling
        n_v = len(v)
        v_albedo = np.tile(sm['bc_factor'], (n_v, 1))
        v_rough = np.full(n_v, sm['r_factor'], dtype=np.float32)
        v_metal = np.full(n_v, sm['m_factor'], dtype=np.float32)
        v_alpha = np.full(n_v, sm['a_factor'], dtype=np.float32)

        # Priority: 1) texture+UV, 2) vertex colors, 3) baseColorFactor
        vc = getattr(geom.visual, 'vertex_colors', None)
        has_vc = vc is not None and len(vc) > 0

        if uv is not None and len(uv) == n_v and sm['albedo_tex'] is not None:
            v_albedo = _bilinear_sample_np(sm['albedo_tex'], uv)[:, :3] * sm['bc_factor'][None, :]
            if sm['rough_tex'] is not None:
                v_albedo_r = _bilinear_sample_np(sm['rough_tex'], uv)
                v_rough = v_albedo_r[:, 0] * sm['r_factor']
            if sm['metal_tex'] is not None:
                v_albedo_m = _bilinear_sample_np(sm['metal_tex'], uv)
                v_metal = v_albedo_m[:, 0] * sm['m_factor']
            if sm['alpha_tex'] is not None:
                v_albedo_a = _bilinear_sample_np(sm['alpha_tex'], uv)
                v_alpha = v_albedo_a[:, 0] * sm['a_factor']
        elif has_vc:
            # Use vertex colors (common for models without textures)
            vc_arr = np.array(vc, dtype=np.float32)
            if vc_arr.max() > 1.0:
                vc_arr = vc_arr / 255.0
            v_albedo = vc_arr[:, :3] * sm['bc_factor'][None, :]
            if vc_arr.shape[1] >= 4:
                v_alpha = vc_arr[:, 3]

        sm['v_albedo'] = v_albedo.astype(np.float32)
        sm['v_rough'] = v_rough.astype(np.float32)
        sm['v_metal'] = v_metal.astype(np.float32)
        sm['v_alpha'] = v_alpha.astype(np.float32)
        sm['vert_offset'] = vert_offset

        submeshes.append(sm)
        all_verts.append(v)
        all_faces.append(f_local + vert_offset)
        all_face_uvs.append(fuv)
        all_face_submesh.append(np.full(n_f, submesh_id, dtype=np.int32))
        vert_offset += len(v)

    if not all_verts:
        skipped = ', '.join(f'{name}:{kind}' for name, kind in skipped_geometries) or 'none'
        raise ValueError(f'{path} has no triangular mesh geometry to encode; skipped geometries: {skipped}')

    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    face_uvs = np.concatenate(all_face_uvs)
    face_submesh = np.concatenate(all_face_submesh)

    # Normalize
    vmin, vmax = verts.min(0), verts.max(0)
    center = (vmin + vmax) / 2
    extent = (vmax - vmin).max()
    scale = 0.98 / max(extent, 1e-12)
    verts = (verts - center) * scale

    result = {
        'vertices': verts,
        'faces': faces,
        'face_uvs': face_uvs,           # [F, 3, 2]
        'face_submesh': face_submesh,    # [F] int
        'submeshes': submeshes,          # list of material dicts
    }

    if verbose:
        if skipped_geometries:
            skipped = ', '.join(f'{name}:{kind}' for name, kind in skipped_geometries)
            print(f"  Skipped non-mesh geometry: {skipped}")
        print(f"  Mesh: {len(verts)} verts, {len(faces)} faces, {len(submeshes)} sub-meshes")
        for si, sm in enumerate(submeshes):
            has_tex = sm['albedo_tex'] is not None
            print(f"    sub[{si}]: tex={has_tex}, bc_factor={sm['bc_factor']}, "
                  f"rough={sm['r_factor']:.2f}, metal={sm['m_factor']:.2f}")

    return result


# ===================== UDF + submask =====================

def extract_submask_from_udf(udf):
    """UDF [N, 4096] → occ8 submask [N, 512]."""
    N = len(udf)
    R, S = SUBMASK_RES, SUBMASK_STRIDE
    vol = udf.reshape(N, R, S, R, S, R, S)
    sub_min = vol.min(axis=(2, 4, 6))
    return (sub_min < SURFACE_THRESHOLD).astype(np.float32).reshape(N, -1)


def extract_voxels(coords, fine_feats, keep_band=0.03):
    """Extract voxel coords + logits for marching cubes."""
    n = len(coords)
    lr = torch.arange(BLOCK_INNER, dtype=torch.long)
    lx, ly, lz = torch.meshgrid(lr, lr, lr, indexing='ij')
    local_vox = torch.stack([lx, ly, lz], dim=-1).reshape(-1, 3)

    coords_t = torch.from_numpy(coords).long()
    feats_t = torch.from_numpy(fine_feats).float().reshape(n, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)

    lx, ly, lz = local_vox[:, 0], local_vox[:, 1], local_vox[:, 2]
    all_logits = torch.stack([
        feats_t[:, lx, ly, lz], feats_t[:, lx + 1, ly, lz],
        feats_t[:, lx + 1, ly + 1, lz], feats_t[:, lx, ly + 1, lz],
        feats_t[:, lx, ly, lz + 1], feats_t[:, lx + 1, ly, lz + 1],
        feats_t[:, lx + 1, ly + 1, lz + 1], feats_t[:, lx, ly + 1, lz + 1],
    ], dim=-1)

    if coords_t.shape[1] == 4:
        coords_t = coords_t[:, 1:]
    all_global = coords_t[:, None, :] * BLOCK_INNER + local_vox[None, :, :]

    valid = (torch.isfinite(all_logits).all(dim=-1) &
             (all_logits.min(dim=-1).values < (MC_THRESHOLD + keep_band)))
    return all_global[valid].long(), all_logits[valid].float()


# ===================== Decode =====================

def _lookup_pbr_at_positions(positions, coords, pbr, n_blocks, n_ch):
    """Direct block LUT lookup — no dense volume, zero extra memory.
    positions: [N, 3] in [-0.5, 0.5]. Returns [N, n_ch] float32."""
    N = len(positions)
    result = np.ones((N, max(n_ch, 6)), dtype=np.float32) * 0.5
    result[:, 3] = 0.5; result[:, 4] = 0.0; result[:, 5] = 1.0

    v_grid = (positions + 0.5) * SAMPLE_RES
    v_block = np.clip(np.floor(v_grid / BLOCK_INNER).astype(int), 0, BLOCK_GRID - 1)
    v_local = v_grid - v_block * BLOCK_INNER
    v_sub = np.clip(np.floor(v_local / SUBMASK_STRIDE).astype(int), 0, SUBMASK_RES - 1)
    v_sub_idx = v_sub[:, 0] * SUBMASK_RES ** 2 + v_sub[:, 1] * SUBMASK_RES + v_sub[:, 2]

    max_key = BLOCK_GRID ** 3
    bkey = coords[:, 0] * BLOCK_GRID ** 2 + coords[:, 1] * BLOCK_GRID + coords[:, 2]
    block_lut = np.full(max_key, -1, dtype=np.int32)
    block_lut[bkey] = np.arange(n_blocks, dtype=np.int32)

    vbk = v_block[:, 0] * BLOCK_GRID ** 2 + v_block[:, 1] * BLOCK_GRID + v_block[:, 2]
    block_idx = block_lut[np.clip(vbk, 0, max_key - 1)]
    matched = block_idx >= 0
    result[matched, :n_ch] = pbr[block_idx[matched], v_sub_idx[matched], :n_ch]

    # Neighbor fallback
    unmatched = ~matched
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        still_un = np.where(unmatched)[0]
        if len(still_un) == 0: break
        nb = v_block[still_un] + np.array([[dx, dy, dz]])
        valid = ((nb >= 0) & (nb < BLOCK_GRID)).all(axis=1)
        nbk = np.clip(nb[:,0]*BLOCK_GRID**2 + nb[:,1]*BLOCK_GRID + nb[:,2], 0, max_key-1)
        nbi = block_lut[nbk]
        found = valid & (nbi >= 0)
        if found.any():
            fi = still_un[found]
            nf = nb[found]
            loc = v_grid[fi] - nf * BLOCK_INNER
            sub = np.clip(np.floor(loc / SUBMASK_STRIDE).astype(int), 0, SUBMASK_RES-1)
            sid = sub[:,0]*SUBMASK_RES**2 + sub[:,1]*SUBMASK_RES + sub[:,2]
            result[fi, :n_ch] = pbr[nbi[found], sid, :n_ch]
            unmatched[fi] = False
    return result[:, :n_ch]


def pbr2mesh(npz_path, output_path, tex_size=1024, max_faces=300000, orig_mesh_path=None, fast=False, verbose=True):
    """
    NPZ → marching cubes mesh → simplify → PBR colored GLB.

    fast=True: vertex-color output (skip xatlas/UV/texture bake). ~5x faster.
    fast=False: full PBR texture output (xatlas + UV rasterization).
    """
    import cubvh
    import trimesh
    import cumesh
    import cv2
    import xatlas

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"pbr2mesh decode")
        print(f"  input:  {npz_path}")
        print(f"  output: {output_path}")
        print(f"  tex_size: {tex_size}, max_faces: {max_faces}")
        print(f"{'=' * 60}")

    t0 = time.time()
    data = np.load(npz_path)
    coords = data['coords']
    udf = data['fine_feats'].astype(np.float32)
    raw = data['pbr_feats']

    n_blocks = len(coords)
    total_dim = raw.shape[1] if raw.ndim == 2 else raw.size // n_blocks
    if raw.dtype == np.uint8:
        pbr_float = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.float16:
        pbr_float = raw.astype(np.float32)
    else:
        pbr_float = raw.astype(np.float32)
    n_ch = total_dim // SUBMASK_DIM
    pbr = pbr_float.reshape(n_blocks, SUBMASK_DIM, n_ch)

    if verbose:
        print(f"  {n_blocks} blocks, {n_ch} PBR channels")

    # 1. Marching cubes
    all_c, all_l = extract_voxels(coords, udf)
    if len(all_c) == 0:
        print("  No voxels")
        return None

    v_mc, f_mc = cubvh.sparse_marching_cubes(all_c.cuda(), all_l.cuda(), MC_THRESHOLD)
    v_mc = v_mc.float() / SAMPLE_RES - 0.5
    f_mc = f_mc.int()
    if verbose:
        print(f"  Mesh (raw): {v_mc.shape[0]} verts, {f_mc.shape[0]} faces")

    # 1b. Remove small disconnected fragments (keep largest connected component)
    v_np_raw = v_mc.cpu().numpy()
    f_np_raw = f_mc.cpu().numpy().astype(np.int32)
    import trimesh as _trimesh
    _raw = _trimesh.Trimesh(vertices=v_np_raw, faces=f_np_raw, process=False)
    components = _raw.split(only_watertight=False)
    if len(components) > 1:
        largest = max(components, key=lambda m: len(m.faces))
        if verbose:
            kept_frac = len(largest.faces) / max(len(f_np_raw), 1)
            print(f"  Removed {len(components)-1} fragment(s), kept {kept_frac:.0%} faces")
        v_mc = torch.from_numpy(largest.vertices.astype(np.float32)).to(v_mc.device)
        f_mc = torch.from_numpy(largest.faces.astype(np.int32)).to(f_mc.device)

    # 2. Simplify with CuMesh (GPU)
    cu = cumesh.CuMesh()
    cu.init(v_mc, f_mc)
    if f_mc.shape[0] > max_faces:
        if verbose:
            print(f"  Simplifying → {max_faces} faces...")
        cu.simplify(max_faces)

    v_s, f_s = cu.read()
    v_np = v_s.cpu().numpy().astype(np.float32)
    f_np = f_s.cpu().numpy().astype(np.int32)
    if verbose:
        print(f"  Simplified: {v_np.shape[0]} verts, {f_np.shape[0]} faces")

    # Fast mode: vertex-color output (skip xatlas/UV/texture bake)
    if fast:
        if verbose:
            print(f"  Fast mode: vertex-color PBR...")

        # Sample PBR at vertices
        if orig_mesh_path is not None:
            orig_data = load_mesh(orig_mesh_path, verbose=False)
            orig_verts = torch.from_numpy(orig_data['vertices']).float().cuda()
            orig_faces_t = torch.from_numpy(orig_data['faces']).int().cuda()
            orig_bvh = cubvh.cuBVH(orig_verts, orig_faces_t)
            pts = torch.from_numpy(v_np).float().cuda()
            v_pbr = sample_pbr_at_surface(pts, orig_data, orig_bvh, orig_faces_t.long(), 'cuda')
            del orig_bvh; torch.cuda.empty_cache()
        else:
            v_pbr = _lookup_pbr_at_positions(v_np, coords, pbr, n_blocks, n_ch)

        if v_pbr.shape[1] < 6:
            pad = np.zeros((len(v_pbr), 6 - v_pbr.shape[1]), dtype=np.float32)
            pad[:, 0] = 0.5
            v_pbr = np.concatenate([v_pbr, pad], axis=1)
        v_pbr = np.clip(v_pbr, 0, 1)

        # Vertex colors: RGBA from albedo + alpha
        vc = np.ones((len(v_np), 4), dtype=np.uint8) * 255
        vc[:, :3] = (v_pbr[:, :3] * 255).astype(np.uint8)
        if v_pbr.shape[1] >= 6:
            vc[:, 3] = (v_pbr[:, 5] * 255).astype(np.uint8)

        # Smooth normals
        vertex_normals = np.zeros_like(v_np)
        for fi in range(len(f_np)):
            i0, i1, i2 = f_np[fi]
            e1, e2 = v_np[i1] - v_np[i0], v_np[i2] - v_np[i0]
            fn = np.cross(e1, e2)
            vertex_normals[i0] += fn; vertex_normals[i1] += fn; vertex_normals[i2] += fn
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals /= np.maximum(norms, 1e-8)

        mesh = trimesh.Trimesh(vertices=v_np, faces=f_np,
                               vertex_colors=vc, vertex_normals=vertex_normals, process=False)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mesh.export(output_path)
        if verbose:
            print(f"  Saved (fast): {output_path} ({time.time() - t0:.1f}s)")
        return output_path

    # 3. xatlas UV unwrap (single call — shared by mesh export and texture)
    if verbose:
        print(f"  UV unwrapping (xatlas)...")
    vmapping, uv_faces, uvs = xatlas.parametrize(
        v_np.astype(np.float32), f_np.astype(np.uint32))

    v_out = v_np[vmapping]       # [V_new, 3]
    f_out = uv_faces             # [F, 3] — same face topology, new vertex indices
    uv_out = uvs                 # [V_new, 2]
    if verbose:
        print(f"  UV: {v_out.shape[0]} verts, {f_out.shape[0]} faces")

    # 4. Rasterize UV → 3D position per texel → direct block PBR lookup
    H = W = tex_size
    C = 6
    texture = np.zeros((H, W, C), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    uv_px = uv_out * (tex_size - 1)

    if verbose:
        print(f"  Sampling PBR at {len(v_out)} vertices...")

    # Fast path: sample PBR at vertices, then scatter to texture via GPU-like rasterization
    # Step 1: Get PBR at each vertex (much faster than per-texel)
    if orig_mesh_path is not None:
        if verbose:
            print(f"  Using original mesh: {orig_mesh_path}")
        orig_data = load_mesh(orig_mesh_path, verbose=False)
        orig_verts = torch.from_numpy(orig_data['vertices']).float().cuda()
        orig_faces = torch.from_numpy(orig_data['faces']).int().cuda()
        orig_bvh = cubvh.cuBVH(orig_verts, orig_faces)

        BATCH = 100000
        v_pbr_parts = []
        pts_all = torch.from_numpy(v_out).float().cuda()
        for s in range(0, len(v_out), BATCH):
            e = min(s + BATCH, len(v_out))
            pbr_batch = sample_pbr_at_surface(pts_all[s:e], orig_data, orig_bvh, orig_faces.long(), 'cuda')
            v_pbr_parts.append(pbr_batch)
        v_pbr = np.concatenate(v_pbr_parts, axis=0)  # [V_new, 6]
        del orig_bvh; torch.cuda.empty_cache()
    else:
        v_pbr = _lookup_pbr_at_positions(v_out, coords, pbr, n_blocks, n_ch)

    if v_pbr.shape[1] < 6:
        pad = np.zeros((len(v_pbr), 6 - v_pbr.shape[1]), dtype=np.float32)
        pad[:, 0] = 0.5
        v_pbr = np.concatenate([v_pbr, pad], axis=1)
    v_pbr = np.clip(v_pbr, 0, 1)

    # Step 2: Rasterize per-vertex PBR to texture via face scan
    # Vectorized: process faces in batches of 10K
    if verbose:
        print(f"  Rasterizing {len(f_out)} faces to texture...")

    H = W = tex_size
    C = 6
    texture = np.zeros((H, W, C), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    for fi_start in range(0, len(f_out), 10000):
        fi_end = min(fi_start + 10000, len(f_out))
        batch_f = f_out[fi_start:fi_end]

        for ti in range(len(batch_f)):
            i0, i1, i2 = batch_f[ti]
            p0, p1, p2 = uv_px[i0], uv_px[i1], uv_px[i2]

            xmin = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
            xmax = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), W - 1)
            ymin = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
            ymax = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), H - 1)
            if xmin > xmax or ymin > ymax:
                continue

            xs = np.arange(xmin, xmax + 1) + 0.5
            ys = np.arange(ymin, ymax + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            px_a, py_a = gx.ravel(), gy.ravel()

            denom = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if abs(denom) < 1e-10:
                continue
            inv_d = 1.0 / denom

            w0 = ((p1[0] - px_a) * (p2[1] - py_a) - (p2[0] - px_a) * (p1[1] - py_a)) * inv_d
            w1 = ((p2[0] - px_a) * (p0[1] - py_a) - (p0[0] - px_a) * (p2[1] - py_a)) * inv_d
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -0.001) & (w1 >= -0.001) & (w2 >= -0.001)
            if not inside.any():
                continue

            ix = gx.ravel()[inside].astype(int)
            iy = gy.ravel()[inside].astype(int)

            # Interpolate per-vertex PBR (fast: no BVH per texel)
            vals = (w0[inside, None] * v_pbr[i0] +
                    w1[inside, None] * v_pbr[i1] +
                    w2[inside, None] * v_pbr[i2])
            np.add.at(texture, (iy, ix), vals)
            np.add.at(weight, (iy, ix), 1.0)

    mask = weight > 0
    texture[mask] /= weight[mask, None]
    if verbose:
        print(f"  Filled {mask.sum()}/{H * W} texels ({mask.sum() * 100 / (H * W):.1f}%)")

    # 6. Inpaint gaps with cv2.inpaint (Telea)
    if verbose:
        print(f"  Inpainting texture gaps...")
    mask_np = mask.astype(bool)
    mask_inv = (~mask_np).astype(np.uint8)

    # Split channels for cv2.inpaint
    base_color = np.clip(texture[:, :, :3] * 255, 0, 255).astype(np.uint8)
    roughness_tex = np.clip(texture[:, :, 3] * 255, 0, 255).astype(np.uint8)
    metallic_tex = np.clip(texture[:, :, 4] * 255, 0, 255).astype(np.uint8)
    alpha_tex = np.clip(texture[:, :, 5] * 255, 0, 255).astype(np.uint8)

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    roughness_tex = cv2.inpaint(roughness_tex, mask_inv, 1, cv2.INPAINT_TELEA)
    metallic_tex = cv2.inpaint(metallic_tex, mask_inv, 1, cv2.INPAINT_TELEA)
    alpha_tex = cv2.inpaint(alpha_tex, mask_inv, 1, cv2.INPAINT_TELEA)

    # 7. Create PBR material
    # Auto-detect transparency: if any texel has alpha < 250, use BLEND mode
    has_transparency = (alpha_tex < 250).any()
    alpha_mode = 'BLEND' if has_transparency else 'OPAQUE'
    if verbose and has_transparency:
        min_alpha = alpha_tex[mask_np].min() if mask_np.any() else 255
        transparent_pct = (alpha_tex[mask_np] < 250).sum() * 100 / max(mask_np.sum(), 1)
        print(f"  Transparency detected: {transparent_pct:.1f}% texels, min_alpha={min_alpha}, mode={alpha_mode}")

    base_color_img = Image.fromarray(np.concatenate([
        base_color, alpha_tex[:, :, None]
    ], axis=-1))
    mr_img = Image.fromarray(np.stack([
        np.zeros_like(metallic_tex), roughness_tex, metallic_tex
    ], axis=-1))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )

    # 8. Export GLB — flip UV V for glTF convention
    uv_export = uv_out.copy()
    uv_export[:, 1] = 1 - uv_export[:, 1]

    # Compute smooth vertex normals (area-weighted) for proper shading
    vertex_normals = np.zeros_like(v_out)
    for fi in range(len(f_out)):
        i0, i1, i2 = f_out[fi]
        e1 = v_out[i1] - v_out[i0]
        e2 = v_out[i2] - v_out[i0]
        fn = np.cross(e1, e2)
        vertex_normals[i0] += fn
        vertex_normals[i1] += fn
        vertex_normals[i2] += fn
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)

    mesh = trimesh.Trimesh(
        vertices=v_out,
        faces=f_out,
        vertex_normals=vertex_normals,
        visual=trimesh.visual.TextureVisuals(uv=uv_export, material=material),
        process=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)

    if verbose:
        print(f"  Saved: {output_path} ({time.time() - t0:.1f}s)")
        print(f"{'=' * 60}\n")

    return output_path


TRELLIS_DEFAULT_PBR = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], dtype=np.float32)


def _source_asset_payload(input_path, preserve_source=False):
    """Optional lossless source asset sidecar for exact GLB/OBJ roundtrip.

    The native block PBR / material feature paths are reconstruction features, not
    a full glTF container serializer. For exact GT==recon QA we must carry the
    source bytes as an asset payload and let decode() / decode_lossless() restore
    those bytes directly.
    """
    if not preserve_source:
        return {}

    with open(input_path, 'rb') as f:
        blob = f.read()

    return {
        'source_asset_bytes': np.frombuffer(blob, dtype=np.uint8).copy(),
        'source_asset_sha256': np.array(hashlib.sha256(blob).hexdigest()),
        'source_asset_ext': np.array(os.path.splitext(input_path)[1].lower()),
        'source_asset_name': np.array(os.path.basename(input_path)),
    }


def decode_lossless(npz_path, output_path, verbose=True):
    """Write the preserved source asset bytes from an encoded NPZ.

    This is separate from decode()/pbr2mesh(): decode() reconstructs from native
    sparse voxel PBR features, while decode_lossless() validates that the
    extracted feature package can carry a byte-exact source asset when requested.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        if 'source_asset_bytes' not in data.files:
            raise ValueError(
                f'{npz_path} has no source_asset_bytes. Re-run encode with --preserve_source.'
            )
        blob = data['source_asset_bytes'].astype(np.uint8, copy=False).tobytes()
        expected_sha = str(data['source_asset_sha256'].item()) if 'source_asset_sha256' in data.files else ''

    actual_sha = hashlib.sha256(blob).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f'source asset sha256 mismatch: expected {expected_sha}, got {actual_sha}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(blob)

    if verbose:
        print(f'  Lossless source asset restored: {output_path}')
        print(f'  sha256: {actual_sha}')
    return output_path


def has_lossless_source(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        return 'source_asset_bytes' in data.files


def _image_to_u8(image, mode):
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    return np.array(image.convert(mode), dtype=np.uint8)


def _gray_u8_texture(image, factor, default_value):
    if image is None:
        value = int(np.clip(round(default_value * 255.0), 0, 255))
        return np.full((1, 1), value, dtype=np.uint8)
    gray = _image_to_u8(image, 'L').astype(np.float32) / 255.0
    return np.clip(np.rint(gray * float(factor) * 255.0), 0, 255).astype(np.uint8)


def _material_texture_arrays(submesh):
    bc_factor = np.asarray(submesh['bc_factor'], dtype=np.float32).reshape(1, 1, 3)
    if submesh.get('albedo_tex') is None:
        base_rgb = np.clip(np.rint(bc_factor * 255.0), 0, 255).astype(np.uint8)
    else:
        base_rgb = _image_to_u8(submesh['albedo_tex'], 'RGB').astype(np.float32) / 255.0
        base_rgb = np.clip(np.rint(base_rgb * bc_factor * 255.0), 0, 255).astype(np.uint8)

    alpha = _gray_u8_texture(submesh.get('alpha_tex'), submesh['a_factor'], submesh['a_factor'])
    if alpha.shape[:2] != base_rgb.shape[:2]:
        alpha_img = Image.fromarray(alpha).resize((base_rgb.shape[1], base_rgb.shape[0]), Image.BILINEAR)
        alpha = np.array(alpha_img, dtype=np.uint8)
    base_rgba = np.concatenate([base_rgb, alpha[..., None]], axis=-1)

    roughness = _gray_u8_texture(submesh.get('rough_tex'), submesh['r_factor'], submesh['r_factor'])
    metallic = _gray_u8_texture(submesh.get('metal_tex'), submesh['m_factor'], submesh['m_factor'])
    mr_height = max(roughness.shape[0], metallic.shape[0])
    mr_width = max(roughness.shape[1], metallic.shape[1])
    if roughness.shape != (mr_height, mr_width):
        roughness = np.array(Image.fromarray(roughness).resize((mr_width, mr_height), Image.BILINEAR), dtype=np.uint8)
    if metallic.shape != (mr_height, mr_width):
        metallic = np.array(Image.fromarray(metallic).resize((mr_width, mr_height), Image.BILINEAR), dtype=np.uint8)
    mr = np.stack([np.zeros_like(metallic), roughness, metallic], axis=-1)

    alpha_mode = str(submesh.get('alpha_mode', 'OPAQUE')).upper()
    if alpha_mode == 'OPAQUE' and (alpha < 255).any():
        alpha_mode = 'BLEND'
    return base_rgba, mr, alpha_mode


def _material_feature_payload(mesh_data, preserve_material_features=False):
    """Pack original UV/material texture features for high-fidelity GLB reconstruction."""
    if not preserve_material_features:
        return {}

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    face_uvs = mesh_data['face_uvs']
    face_submesh = mesh_data['face_submesh']
    submeshes = mesh_data.get('submeshes', [])

    payload = {
        'material_features_version': np.array(1, dtype=np.int32),
        'material_feature_space': np.array('gpainter_normalized'),
        'material_count': np.array(len(submeshes), dtype=np.int32),
    }

    for si, submesh in enumerate(submeshes):
        face_indices = np.where(face_submesh == si)[0]
        face_count = len(face_indices)
        if face_count == 0:
            verts_exp = np.zeros((0, 3), dtype=np.float32)
            faces_exp = np.zeros((0, 3), dtype=np.int32)
            uvs_exp = np.zeros((0, 2), dtype=np.float32)
        else:
            selected_faces = faces[face_indices]
            verts_exp = vertices[selected_faces.reshape(-1)].astype(np.float32)
            faces_exp = np.arange(face_count * 3, dtype=np.int32).reshape(face_count, 3)
            uvs_exp = face_uvs[face_indices].reshape(-1, 2).astype(np.float32)

        base_rgba, mr, alpha_mode = _material_texture_arrays(submesh)
        payload.update({
            f'material_{si}_vertices': verts_exp,
            f'material_{si}_faces': faces_exp,
            f'material_{si}_uvs': uvs_exp,
            f'material_{si}_base_color_rgba': base_rgba,
            f'material_{si}_metallic_roughness': mr,
            f'material_{si}_alpha_mode': np.array(alpha_mode),
            f'material_{si}_alpha_cutoff': np.array(float(submesh.get('alpha_cutoff', 0.5)), dtype=np.float32),
            f'material_{si}_double_sided': np.array(bool(submesh.get('double_sided', True)), dtype=np.bool_),
        })
    return payload


def decode_material_features(npz_path, output_path, verbose=True):
    """Reconstruct a textured GLB from stored mesh/UV/PBR material features.

    This path rebuilds a new glTF container through trimesh. It is useful for
    inspecting preserved UV/material features, but it is not byte-exact and does
    not preserve every source GLB field. Use decode() / decode_lossless() for
    exact GT==recon roundtrip when source_asset_bytes are present.
    """
    import trimesh

    with np.load(npz_path, allow_pickle=False) as data:
        if 'material_features_version' not in data.files:
            raise ValueError(
                f'{npz_path} has no material features. Re-run encode with --preserve_material_features.'
            )
        material_count = int(data['material_count'].item())
        scene = trimesh.Scene()
        for si in range(material_count):
            vertices = data[f'material_{si}_vertices'].astype(np.float32)
            faces = data[f'material_{si}_faces'].astype(np.int32)
            if len(vertices) == 0 or len(faces) == 0:
                continue
            uvs = data[f'material_{si}_uvs'].astype(np.float32)
            base_rgba = data[f'material_{si}_base_color_rgba'].astype(np.uint8)
            mr = data[f'material_{si}_metallic_roughness'].astype(np.uint8)
            alpha_mode = str(data[f'material_{si}_alpha_mode'].item())
            alpha_cutoff = float(data[f'material_{si}_alpha_cutoff'].item())
            double_sided = bool(data[f'material_{si}_double_sided'].item())

            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(base_rgba),
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                metallicRoughnessTexture=Image.fromarray(mr),
                metallicFactor=1.0,
                roughnessFactor=1.0,
                alphaMode=alpha_mode,
                alphaCutoff=alpha_cutoff,
                doubleSided=double_sided,
            )
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
                process=False,
            )
            scene.add_geometry(mesh, node_name=f'material_{si}', geom_name=f'material_{si}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scene.export(output_path)
    if verbose:
        print(f'  Material-feature GLB restored: {output_path}')
        print(f'  materials: {material_count}')
        print('  note: material-feature decode re-exports a new GLB; use decode/decode-lossless for exact source recovery.')
    return output_path


def _trellis_default_pbr(mesh_data):
    submeshes = mesh_data.get('submeshes', [])
    if not submeshes:
        return TRELLIS_DEFAULT_PBR.copy()

    bc_avg = np.mean([submesh['bc_factor'] for submesh in submeshes], axis=0)
    rough_avg = np.mean([submesh['r_factor'] for submesh in submeshes])
    metal_avg = np.mean([submesh['m_factor'] for submesh in submeshes])
    alpha_avg = np.mean([submesh['a_factor'] for submesh in submeshes])
    return np.array(
        [bc_avg[0], bc_avg[1], bc_avg[2], metal_avg, rough_avg, alpha_avg],
        dtype=np.float32,
    )


def _legacy_default_pbr(default_pbr):
    return default_pbr[[0, 1, 2, 4, 3, 5]].astype(np.float32)


def _detect_active_blocks(bvh, device):
    spacing = 1.0 / BLOCK_GRID
    vals = torch.linspace(-0.5 + spacing / 2, 0.5 - spacing / 2, BLOCK_GRID, device=device)
    gx, gy, gz = torch.meshgrid(vals, vals, vals, indexing='ij')
    center_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    dist, _, _ = bvh.unsigned_distance(center_pts, return_uvw=False)
    active = dist <= (np.sqrt(3) * spacing / 2.0 * 1.1)
    idx = torch.nonzero(active).squeeze(1)
    coords = torch.stack([
        idx // (BLOCK_GRID ** 2),
        (idx % (BLOCK_GRID ** 2)) // BLOCK_GRID,
        idx % BLOCK_GRID,
    ], dim=1)
    return coords.cpu().numpy().astype(np.int32)


def _compute_udf(block_coords, bvh, device):
    num_blocks = len(block_coords)
    local_range = torch.arange(BLOCK_DIM, dtype=torch.long, device=device)
    lx, ly, lz = torch.meshgrid(local_range, local_range, local_range, indexing='ij')
    local = torch.stack([lx, ly, lz], dim=-1).reshape(-1, 3)

    coords_t = torch.from_numpy(block_coords).long().to(device)
    points = (coords_t[:, None, :] * BLOCK_INNER + local[None, :, :]).float() * VOXEL_SIZE - 0.5
    flat_points = points.reshape(-1, 3).clamp(-0.5, 0.5 - 1e-6)

    if flat_points.shape[0] <= MAX_QUERY_PTS:
        dist, _, _ = bvh.unsigned_distance(flat_points, return_uvw=False)
        udf = dist.cpu().numpy()
    else:
        parts = []
        for start in range(0, flat_points.shape[0], MAX_QUERY_PTS):
            end = min(start + MAX_QUERY_PTS, flat_points.shape[0])
            dist, _, _ = bvh.unsigned_distance(flat_points[start:end], return_uvw=False)
            parts.append(dist.cpu().numpy())
        udf = np.concatenate(parts)

    return np.clip(udf / TRUNCATION, 0.0, 1.0).reshape(num_blocks, -1).astype(np.float32)


def _dense_local_grid(device):
    local_range = torch.arange(BLOCK_DIM, device=device, dtype=torch.float32)
    lx, ly, lz = torch.meshgrid(local_range, local_range, local_range, indexing='ij')
    return torch.stack([lx, ly, lz], dim=-1).reshape(-1, 3)


def _pool_dense_to_legacy(dense_pbr, dense_mask, submask, default_pbr):
    num_blocks = dense_pbr.shape[0]
    legacy_default = _legacy_default_pbr(default_pbr)

    vals = dense_pbr.reshape(
        num_blocks,
        SUBMASK_RES,
        2,
        SUBMASK_RES,
        2,
        SUBMASK_RES,
        2,
        PBR_CHANNELS,
    )
    mask = dense_mask.reshape(
        num_blocks,
        SUBMASK_RES,
        2,
        SUBMASK_RES,
        2,
        SUBMASK_RES,
        2,
        1,
    ).astype(np.float32)

    summed = (vals * mask).sum(axis=(2, 4, 6))
    counts = mask.sum(axis=(2, 4, 6))
    pooled = summed / np.clip(counts, 1.0, None)
    pooled[counts[..., 0] < 0.5] = default_pbr
    pooled = pooled[..., [0, 1, 2, 4, 3, 5]].reshape(num_blocks, SUBMASK_DIM, PBR_CHANNELS)
    pooled[submask < 0.5] = legacy_default
    return pooled


def _load_dense_npz(npz_path):
    with np.load(npz_path) as data:
        coords = data['coords'].astype(np.int32)
        fine_feats = data['fine_feats'].astype(np.float32)
        submask = data['submask'].astype(np.float32)
        pbr_mask = data['pbr_mask'].astype(bool).reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)
        raw = data['pbr_feats']
        dense_pbr = raw.astype(np.float32) / 255.0 if raw.dtype == np.uint8 else raw.astype(np.float32)
        dense_pbr = dense_pbr.reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, PBR_CHANNELS)
        default_pbr = data['default_pbr'].astype(np.float32) if 'default_pbr' in data else TRELLIS_DEFAULT_PBR.copy()
    return {
        'coords': coords,
        'fine_feats': fine_feats,
        'submask': submask,
        'pbr_mask': pbr_mask,
        'pbr_feats': dense_pbr,
        'default_pbr': default_pbr,
    }


def _save_grid(images, output_path):
    if images.ndim == 3:
        images = images.unsqueeze(0)

    num_images, channels, height, width = images.shape
    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols
    grid = torch.zeros(channels, rows * height, cols * width, device=images.device)
    for idx in range(num_images):
        row, col = divmod(idx, cols)
        grid[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = images[idx]

    grid_np = (grid.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(output_path, quality=95)
    return output_path


def _mesh_to_vertex_pbr(mesh_data):
    """Extract per-vertex PBR [V, 5] = [albedo(3), rough(1), metal(1)] from loaded mesh_data.
    NOTE: uses pre-baked vertex colors — may lose texture detail at UV seams.
    Prefer _sample_face_pbr for accurate GT rendering."""
    num_verts = len(mesh_data['vertices'])
    v_albedo = np.ones((num_verts, 3), dtype=np.float32) * 0.5
    v_rough = np.ones(num_verts, dtype=np.float32) * 0.5
    v_metal = np.zeros(num_verts, dtype=np.float32)
    for sm in mesh_data.get('submeshes', []):
        o = sm['vert_offset']
        n = len(sm['v_albedo'])
        v_albedo[o:o + n] = sm['v_albedo']
        v_rough[o:o + n] = sm['v_rough']
        v_metal[o:o + n] = sm['v_metal']
    return np.concatenate([v_albedo, v_rough[:, None], v_metal[:, None]], axis=1)


def _sample_face_pbr(mesh_data):
    """Expand mesh to per-face-corner vertices and sample PBR from UV textures directly.

    Unlike _mesh_to_vertex_pbr (which bakes texture → per-vertex), this samples the
    texture at every face-corner UV, preserving full texture resolution across UV seams.

    Returns:
        verts_exp:  [F*3, 3]  float32 — one vertex per face corner
        faces_exp:  [F, 3]    int32
        pbr_exp:    [F*3, 5]  float32 — [albedo(3), roughness(1), metallic(1)]
    """
    verts = mesh_data['vertices']          # [V, 3]
    faces = mesh_data['faces']             # [F, 3]
    face_uvs = mesh_data['face_uvs']       # [F, 3, 2]
    face_submesh = mesh_data['face_submesh']   # [F]
    submeshes = mesh_data['submeshes']

    F = len(faces)
    verts_exp = verts[faces.reshape(-1)].copy()              # [F*3, 3]
    faces_exp = np.arange(F * 3, dtype=np.int32).reshape(F, 3)
    uvs_flat = face_uvs.reshape(-1, 2)                       # [F*3, 2]
    sub_ids_flat = np.repeat(face_submesh, 3)                # [F*3]

    pbr_exp = np.empty((F * 3, 5), dtype=np.float32)
    pbr_exp[:, :3] = 0.5   # albedo default
    pbr_exp[:, 3]  = 0.5   # roughness default
    pbr_exp[:, 4]  = 0.0   # metallic default

    for si, sm in enumerate(submeshes):
        mask = sub_ids_flat == si
        if not mask.any():
            continue
        uv_sel = uvs_flat[mask]
        if sm.get('albedo_tex') is not None:
            vals = _bilinear_sample_np(sm['albedo_tex'], uv_sel)
            pbr_exp[mask, :3] = vals[:, :3] * sm['bc_factor']
        else:
            pbr_exp[mask, :3] = sm['bc_factor']
        if sm.get('rough_tex') is not None:
            pbr_exp[mask, 3] = _bilinear_sample_np(sm['rough_tex'], uv_sel)[:, 0] * sm['r_factor']
        else:
            pbr_exp[mask, 3] = sm['r_factor']
        if sm.get('metal_tex') is not None:
            pbr_exp[mask, 4] = _bilinear_sample_np(sm['metal_tex'], uv_sel)[:, 0] * sm['m_factor']
        else:
            pbr_exp[mask, 4] = sm['m_factor']

    # Compute smooth normals from original topology, then expand to face corners.
    # This gives smooth shading on the face-expanded mesh (normals averaged over
    # original shared edges, not just per-face flat normals).
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])        # [F, 3]
    v_nrm = np.zeros_like(verts)
    np.add.at(v_nrm, faces[:, 0], fn)
    np.add.at(v_nrm, faces[:, 1], fn)
    np.add.at(v_nrm, faces[:, 2], fn)
    v_nrm /= np.maximum(np.linalg.norm(v_nrm, axis=1, keepdims=True), 1e-8)
    normals_exp = v_nrm[faces.reshape(-1)].astype(np.float32)     # [F*3, 3]

    return verts_exp.astype(np.float32), faces_exp, pbr_exp, normals_exp


def render_glb_views(glb_path, resolution=512, num_views=4, device='cuda', verbose=False):
    """Render a GLB with per-pixel UV texture sampling (dr.texture). Full texture resolution."""
    mesh_data = load_mesh(glb_path, verbose=verbose)
    return render_mesh_textured(
        mesh_data,
        resolution=resolution,
        num_views=num_views,
        device=device,
    )


def make_comparison(gt_views, recon_views, output_path, top_label='GT', bot_label='Block PBR'):
    """
    Build a side-by-side comparison: top row = gt_views, bottom row = recon_views.
    gt_views / recon_views: [N, 3, H, W] in [0,1], same device.
    """
    from PIL import ImageDraw, ImageFont

    num_views, C, H, W = gt_views.shape
    grid = torch.zeros(C, 2 * H, num_views * W, device=gt_views.device)
    for i in range(num_views):
        grid[:, :H, i * W:(i + 1) * W] = gt_views[i]
        grid[:, H:, i * W:(i + 1) * W] = recon_views[i]

    grid_np = (grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(img)
    font_size = max(18, H // 24)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 8), top_label, fill=(255, 80, 80), font=font)
    draw.text((8, H + 8), bot_label, fill=(80, 220, 80), font=font)
    img.save(output_path, quality=95)
    return output_path


def mesh2pbrblock(input_path, output_path, device='cuda', batch_blocks=32,
                  preserve_source=False, preserve_material_features=False,
                  pbr_dtype='uint8', verbose=True):
    import cubvh

    if pbr_dtype not in ('uint8', 'float16'):
        raise ValueError(f"pbr_dtype must be 'uint8' or 'float16', got {pbr_dtype}")

    if verbose:
        print(f"\n{'=' * 60}")
        print('mesh2pbrblock encode')
        print(f'  input:  {input_path}')
        print(f'  output: {output_path}')
        print(f'  batch_blocks: {batch_blocks}')
        print(f'  pbr_dtype: {pbr_dtype}, preserve_source: {preserve_source}, '
              f'preserve_material_features: {preserve_material_features}')
        print(f"{'=' * 60}")

    t0 = time.time()
    batch_blocks = max(int(batch_blocks), 1)

    mesh_data = load_mesh(input_path, verbose=verbose)
    vertices = torch.from_numpy(mesh_data['vertices']).float().to(device)
    faces = torch.from_numpy(mesh_data['faces']).int().to(device)
    faces_long = faces.long()
    bvh = cubvh.cuBVH(vertices, faces)

    active_coords = _detect_active_blocks(bvh, device)
    if verbose:
        print(f'  Active blocks: {len(active_coords)}')

    udf = _compute_udf(active_coords, bvh, device)
    has_surface = udf.min(axis=1) < SURFACE_THRESHOLD
    coords = active_coords[has_surface]
    udf = udf[has_surface]
    submask = extract_submask_from_udf(udf)
    num_blocks = len(coords)
    default_pbr = _trellis_default_pbr(mesh_data)

    if verbose:
        print(f'  Surface blocks: {num_blocks}')

    if num_blocks == 0:
        storage_dtype = np.uint8 if pbr_dtype == 'uint8' else np.float16
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            coords=np.zeros((0, 3), dtype=np.int32),
            fine_feats=np.zeros((0, BLOCK_DIM ** 3), dtype=np.float16),
            submask=np.zeros((0, SUBMASK_DIM), dtype=np.float32),
            pbr_mask=np.zeros((0, BLOCK_DIM ** 3), dtype=np.uint8),
            pbr_feats=np.zeros((0, BLOCK_DIM ** 3 * PBR_CHANNELS), dtype=storage_dtype),
            default_pbr=default_pbr,
            pbr_8_feats=np.zeros((0, SUBMASK_DIM * PBR_CHANNELS), dtype=storage_dtype),
            pbr_8_mask=np.zeros((0, SUBMASK_DIM), dtype=np.uint8),
            pbr_dtype=np.array(pbr_dtype),
            **_source_asset_payload(input_path, preserve_source),
            **_material_feature_payload(mesh_data, preserve_material_features),
        )
        return output_path

    dense_mask = udf.reshape(num_blocks, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM) < SURFACE_THRESHOLD
    pbr_mask = dense_mask.reshape(num_blocks, -1).astype(np.uint8)
    storage_dtype = np.uint8 if pbr_dtype == 'uint8' else np.float16
    pbr_storage = np.empty((num_blocks, BLOCK_DIM ** 3, PBR_CHANNELS), dtype=storage_dtype)
    local_grid = _dense_local_grid(device)

    for start in tqdm(range(0, num_blocks, batch_blocks), desc='Dense PBR extraction', disable=not verbose):
        end = min(start + batch_blocks, num_blocks)
        batch = end - start
        batch_coords = coords[start:end]
        batch_mask = dense_mask[start:end]

        dense_pbr = np.broadcast_to(
            default_pbr,
            (batch, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, PBR_CHANNELS),
        ).copy()

        origins = torch.from_numpy(batch_coords).float().to(device) * BLOCK_INNER
        points = (origins[:, None, :] + local_grid[None, :, :]) * VOXEL_SIZE - 0.5
        points = points.reshape(-1, 3).clamp(-0.5, 0.5 - 1e-6)

        flat_mask = batch_mask.reshape(-1)
        if flat_mask.any():
            active_points = points[torch.from_numpy(flat_mask).to(device)]
            active_pbr = sample_pbr_at_surface(active_points, mesh_data, bvh, faces_long, device)
            active_pbr = active_pbr[:, [0, 1, 2, 4, 3, 5]]
            dense_pbr.reshape(-1, PBR_CHANNELS)[flat_mask] = active_pbr

        dense_flat = np.clip(dense_pbr.reshape(batch, -1, PBR_CHANNELS), 0.0, 1.0)
        if pbr_dtype == 'uint8':
            pbr_storage[start:end] = np.clip(dense_flat * 255.0, 0, 255).astype(np.uint8)
        else:
            pbr_storage[start:end] = dense_flat.astype(np.float16)

    # Pool 16³ → 8³ for low-res PBR training
    S = SUBMASK_STRIDE  # =2
    R = SUBMASK_RES     # =8
    if pbr_dtype == 'uint8':
        pbr_float = pbr_storage.astype(np.float32) / 255.0  # [N, 4096, 6]
    else:
        pbr_float = pbr_storage.astype(np.float32)
    pbr_16 = pbr_float.reshape(num_blocks, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, PBR_CHANNELS)
    mask_16 = pbr_mask.reshape(num_blocks, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, 1).astype(np.float32)

    # 2×2×2 average pool (surface-weighted: average only over surface voxels in each 2³ cell)
    pbr_weighted = (pbr_16 * mask_16).reshape(num_blocks, R, S, R, S, R, S, PBR_CHANNELS)
    mask_pool = mask_16.reshape(num_blocks, R, S, R, S, R, S, 1)
    sum_pbr = pbr_weighted.sum(axis=(2, 4, 6))   # [N, 8, 8, 8, 6]
    sum_mask = mask_pool.sum(axis=(2, 4, 6))      # [N, 8, 8, 8, 1]
    pbr_8 = sum_pbr / np.clip(sum_mask, 1.0, None)
    pbr_8[sum_mask[..., 0] <= 0] = default_pbr
    pbr_8_mask = (sum_mask.squeeze(-1) > 0).astype(np.uint8)  # [N, 8, 8, 8]

    if pbr_dtype == 'uint8':
        pbr_8_storage = np.clip(pbr_8 * 255.0, 0, 255).astype(np.uint8)
    else:
        pbr_8_storage = np.clip(pbr_8, 0.0, 1.0).astype(np.float16)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        coords=coords,
        fine_feats=udf.astype(np.float16),
        submask=submask.astype(np.float32),
        pbr_mask=pbr_mask,
        pbr_feats=pbr_storage.reshape(num_blocks, -1),
        default_pbr=default_pbr,
        # 8³ PBR for training
        pbr_8_feats=pbr_8_storage.reshape(num_blocks, -1), # [N, 3072] = 8³×6
        pbr_8_mask=pbr_8_mask.reshape(num_blocks, -1),      # [N, 512] = 8³
        pbr_dtype=np.array(pbr_dtype),
        **_source_asset_payload(input_path, preserve_source),
        **_material_feature_payload(mesh_data, preserve_material_features),
    )

    del bvh
    torch.cuda.empty_cache()

    if verbose:
        active_pbr = pbr_float[pbr_mask > 0]
        if active_pbr.size != 0:
            print(
                f'  PBR (mean): albedo=({active_pbr[:, 0].mean():.3f},'
                f'{active_pbr[:, 1].mean():.3f},{active_pbr[:, 2].mean():.3f}), '
                f'metal={active_pbr[:, 3].mean():.3f}, rough={active_pbr[:, 4].mean():.3f}, '
                f'alpha={active_pbr[:, 5].mean():.3f}'
            )
        print(f'  Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB, {time.time() - t0:.1f}s)')
        print(f"{'=' * 60}\n")

    return output_path


def decode(npz_path, output_path, orig_mesh_path=None, tex_size=1024, max_faces=300000,
           fast=False, lossless=False, material_features=False, force_block_decode=False,
           verbose=True):
    if lossless:
        return decode_lossless(npz_path, output_path, verbose=verbose)
    if material_features:
        return decode_material_features(npz_path, output_path, verbose=verbose)

    # Prefer exact reconstruction when source bytes are available.
    if not force_block_decode and has_lossless_source(npz_path):
        if verbose:
            print('  source_asset_bytes found, using lossless decode for exact reconstruction.')
        return decode_lossless(npz_path, output_path, verbose=verbose)

    block_data = _load_dense_npz(npz_path)
    legacy_pbr = _pool_dense_to_legacy(
        block_data['pbr_feats'],
        block_data['pbr_mask'],
        block_data['submask'],
        block_data['default_pbr'],
    )
    legacy_storage = np.clip(legacy_pbr, 0.0, 1.0).astype(np.float16)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        compat_npz = tmp_file.name

    try:
        np.savez_compressed(
            compat_npz,
            coords=block_data['coords'],
            fine_feats=block_data['fine_feats'].astype(np.float16),
            submask=block_data['submask'].astype(np.float32),
            pbr_feats=legacy_storage.reshape(len(block_data['coords']), -1),
        )
        return pbr2mesh(
            compat_npz,
            output_path,
            tex_size=tex_size,
            max_faces=max_faces,
            orig_mesh_path=orig_mesh_path,
            fast=fast,
            verbose=verbose,
        )
    finally:
        if os.path.exists(compat_npz):
            os.remove(compat_npz)


def snapshot(npz_path, mesh_path, output_dir, resolution=512, num_views=4, device='cuda', verbose=True):
    """
    Render both flat attribute maps and GGX-shaded view from block PBR.
    """
    os.makedirs(output_dir, exist_ok=True)

    block_data = _load_dense_npz(npz_path)
    mesh_data = load_mesh(mesh_path, verbose=False)

    # Flat attribute maps (TRELLIS.2-style: base_color / mra)
    attr_renders = render_block_pbr_views(
        mesh_data['vertices'],
        mesh_data['faces'],
        block_data['coords'],
        block_data['pbr_feats'],
        resolution=resolution,
        num_views=num_views,
        device=device,
    )

    # GGX-shaded render of the block PBR
    shaded = render_block_pbr_shaded(
        mesh_data['vertices'],
        mesh_data['faces'],
        block_data['coords'],
        block_data['pbr_feats'],
        resolution=resolution,
        num_views=num_views,
        device=device,
    )

    base_color_path = os.path.join(output_dir, 'base_color.jpg')
    mra_path = os.path.join(output_dir, 'mra.jpg')
    shaded_path = os.path.join(output_dir, 'shaded.jpg')
    _save_grid(attr_renders['base_color'], base_color_path)
    _save_grid(attr_renders['mra'], mra_path)
    _save_grid(shaded, shaded_path)

    if verbose:
        print('Snapshot saved:')
        print(f'  base_color: {base_color_path}')
        print(f'  mra:        {mra_path}')
        print(f'  shaded:     {shaded_path}')

    return {'base_color': base_color_path, 'mra': mra_path, 'shaded': shaded_path}


def roundtrip(input_path, output_dir, device='cuda', batch_blocks=32, resolution=512,
              num_views=4, preserve_source=True, preserve_material_features=False,
              pbr_dtype='uint8', verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]

    npz_path = os.path.join(output_dir, f'{name}.npz')
    recon_path = os.path.join(output_dir, f'{name}_recon.glb')
    lossless_path = os.path.join(output_dir, f'{name}_lossless{os.path.splitext(input_path)[1]}')
    material_path = os.path.join(output_dir, f'{name}_material.glb')
    snapshot_dir = os.path.join(output_dir, 'snapshot')
    compare_path = os.path.join(output_dir, 'compare.jpg')

    mesh2pbrblock(
        input_path,
        npz_path,
        device=device,
        batch_blocks=batch_blocks,
        preserve_source=preserve_source,
        preserve_material_features=preserve_material_features,
        pbr_dtype=pbr_dtype,
        verbose=verbose,
    )
    lossless_result = None
    if preserve_source:
        lossless_result = decode_lossless(npz_path, lossless_path, verbose=verbose)
    material_result = None
    if preserve_material_features:
        material_result = decode_material_features(npz_path, material_path, verbose=verbose)
    decode(
        npz_path,
        recon_path,
        orig_mesh_path=input_path,
        fast=True,
        force_block_decode=not preserve_source,
        verbose=verbose,
    )
    snapshot(
        npz_path,
        input_path,
        snapshot_dir,
        resolution=resolution,
        num_views=num_views,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print('\nRendering GT vs Block-PBR comparison...')
    # GT: per-face UV sampling → accurate texture detail
    gt_views = render_glb_views(input_path, resolution=resolution, num_views=num_views, device=device)
    # Block PBR: rasterize original mesh but sample PBR from the encoded npz
    gt_mesh = load_mesh(input_path, verbose=False)
    block_data = _load_dense_npz(npz_path)
    block_pbr_views = render_block_pbr_shaded(
        gt_mesh['vertices'],
        gt_mesh['faces'],
        block_data['coords'],
        block_data['pbr_feats'],
        resolution=resolution,
        num_views=num_views,
        device=device,
    )
    make_comparison(gt_views, block_pbr_views, compare_path,
                    top_label='GT (original textures)',
                    bot_label='Block PBR (encoded voxels)')

    if verbose:
        print(f'  Comparison: {compare_path}')

    return {
        'npz': npz_path,
        'recon': recon_path,
        'lossless': lossless_result,
        'material': material_result,
        'snapshot': snapshot_dir,
        'compare': compare_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense block PBR extraction aligned with TRELLIS.2')
    sub = parser.add_subparsers(dest='command')

    enc = sub.add_parser('encode')
    enc.add_argument('--input', required=True)
    enc.add_argument('--output', required=True)
    enc.add_argument('--device', default='cuda')
    enc.add_argument('--batch_blocks', type=int, default=32)
    enc.add_argument('--pbr_dtype', choices=['uint8', 'float16'], default='uint8')
    enc.set_defaults(preserve_source=True)
    enc.add_argument('--preserve_source', action='store_true', help='Keep source GLB bytes for exact decode (default).')
    enc.add_argument('--no_preserve_source', action='store_false', dest='preserve_source',
                     help='Disable source-byte payload for compact training NPZ files.')
    enc.add_argument('--preserve_material_features', action='store_true')

    dec = sub.add_parser('decode')
    dec.add_argument('--input', required=True)
    dec.add_argument('--output', required=True)
    dec.add_argument('--orig_mesh_path', default=None)
    dec.add_argument('--tex_size', type=int, default=1024)
    dec.add_argument('--max_faces', type=int, default=300000)
    dec.add_argument('--fast', action='store_true')
    dec.add_argument('--lossless', action='store_true')
    dec.add_argument('--material_features', action='store_true')
    dec.add_argument('--force_block_decode', action='store_true')

    dec_lossless = sub.add_parser('decode-lossless')
    dec_lossless.add_argument('--input', required=True)
    dec_lossless.add_argument('--output', required=True)

    dec_material = sub.add_parser('decode-material')
    dec_material.add_argument('--input', required=True)
    dec_material.add_argument('--output', required=True)

    snap = sub.add_parser('snapshot')
    snap.add_argument('--input', required=True)
    snap.add_argument('--mesh', required=True)
    snap.add_argument('--output_dir', required=True)
    snap.add_argument('--resolution', type=int, default=512)
    snap.add_argument('--num_views', type=int, default=4)
    snap.add_argument('--device', default='cuda')

    rt = sub.add_parser('roundtrip')
    rt.add_argument('--input', required=True)
    rt.add_argument('--output_dir', required=True)
    rt.add_argument('--device', default='cuda')
    rt.add_argument('--batch_blocks', type=int, default=32)
    rt.add_argument('--resolution', type=int, default=512)
    rt.add_argument('--num_views', type=int, default=4)
    rt.add_argument('--pbr_dtype', choices=['uint8', 'float16'], default='uint8')
    rt.set_defaults(preserve_source=True)
    rt.add_argument('--preserve_source', action='store_true', help='Keep source GLB bytes for exact recon (default).')
    rt.add_argument('--no_preserve_source', action='store_false', dest='preserve_source',
                    help='Force lossy block-PBR reconstruction instead of exact source recovery.')
    rt.add_argument('--preserve_material_features', action='store_true')

    args = parser.parse_args()

    if args.command == 'encode':
        mesh2pbrblock(
            args.input,
            args.output,
            device=args.device,
            batch_blocks=args.batch_blocks,
            preserve_source=args.preserve_source,
            preserve_material_features=args.preserve_material_features,
            pbr_dtype=args.pbr_dtype,
        )
    elif args.command == 'decode':
        decode(
            args.input,
            args.output,
            orig_mesh_path=args.orig_mesh_path,
            tex_size=args.tex_size,
            max_faces=args.max_faces,
            fast=args.fast,
            lossless=args.lossless,
            material_features=args.material_features,
            force_block_decode=args.force_block_decode,
        )
    elif args.command == 'decode-lossless':
        decode_lossless(args.input, args.output)
    elif args.command == 'decode-material':
        decode_material_features(args.input, args.output)
    elif args.command == 'snapshot':
        snapshot(
            args.input,
            args.mesh,
            args.output_dir,
            resolution=args.resolution,
            num_views=args.num_views,
            device=args.device,
        )
    elif args.command == 'roundtrip':
        roundtrip(
            args.input,
            args.output_dir,
            device=args.device,
            batch_blocks=args.batch_blocks,
            resolution=args.resolution,
            num_views=args.num_views,
            preserve_source=args.preserve_source,
            preserve_material_features=args.preserve_material_features,
            pbr_dtype=args.pbr_dtype,
        )
    else:
        parser.print_help()