"""
Lightweight PBR renderer using nvdiffrast.

Renders per-vertex PBR attributes (albedo, roughness, metallic) with
Cook-Torrance GGX BRDF under 3-point lighting.

Camera convention follows Hunyuan3D / nvdiffrast:
  - Row-vector: pos_clip = pos_homo @ mvp.T
  - Y-up world, camera looks along -Z in view space

Usage:
    from gpainter.renderers.pbr_renderer import render_pbr_views
    images = render_pbr_views(vertices, faces, v_pbr, num_views=4)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from ..dataset_toolkits.mesh2block import (
    BLOCK_DIM,
    BLOCK_GRID,
    BLOCK_INNER,
    PBR_CHANNELS,
    SAMPLE_RES,
)


# ===================== Camera (Hunyuan-style) =====================

def _get_mv(elev_deg, azim_deg, radius, center=(0, 0, 0)):
    """World-to-camera matrix. Y-up world, camera looks at -Z in view space."""
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    # Spherical coords (Y-up)
    eye = np.array([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    # c2w: [right, up, -fwd] column-major
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    # w2c = inv(c2w)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3] = -c2w[:3, :3].T @ eye
    return w2c


def _get_proj(fov_deg, aspect=1.0, near=0.1, far=10.0):
    """Perspective projection (OpenGL, same as Hunyuan get_perspective_projection_matrix)."""
    fovy_rad = math.radians(fov_deg)
    return np.array([
        [1.0 / (math.tan(fovy_rad / 2.0) * aspect), 0, 0, 0],
        [0, 1.0 / math.tan(fovy_rad / 2.0), 0, 0],
        [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def _transform_pos(mtx, pos):
    """Transform positions: pos_homo @ mtx.T → [1, V, 4] clip coords.
    Same convention as Hunyuan's transform_pos."""
    posw = torch.cat([pos, torch.ones(pos.shape[0], 1, device=pos.device)], dim=1)
    return (posw @ mtx.T).unsqueeze(0)  # [1, V, 4]


# ===================== GGX BRDF =====================

def _ggx_brdf(normal, view_dir, light_dir, albedo, roughness, metallic, light_color):
    """Cook-Torrance GGX shading. All inputs linear-space, on GPU."""
    if light_dir.dim() == 1:
        light_dir = light_dir.unsqueeze(0).unsqueeze(0)

    H_vec = F.normalize(view_dir + light_dir, dim=-1)
    NdotL = torch.clamp((normal * light_dir).sum(-1, keepdim=True), 0, 1)
    NdotV = torch.clamp((normal * view_dir).sum(-1, keepdim=True), 0, 1)
    NdotH = torch.clamp((normal * H_vec).sum(-1, keepdim=True), 0, 1)
    VdotH = torch.clamp((view_dir * H_vec).sum(-1, keepdim=True), 0, 1)

    alpha = roughness * roughness
    alpha2 = alpha * alpha

    denom = NdotH * NdotH * (alpha2 - 1) + 1
    D = alpha2 / (math.pi * denom * denom + 1e-7)

    k = (roughness + 1) ** 2 / 8
    G1_V = NdotV / (NdotV * (1 - k) + k + 1e-7)
    G1_L = NdotL / (NdotL * (1 - k) + k + 1e-7)
    G = G1_V * G1_L

    F0 = 0.04 * (1 - metallic) + albedo * metallic
    F_term = F0 + (1 - F0) * (1 - VdotH).clamp(0, 1).pow(5)

    spec = D * G * F_term / (4 * NdotV * NdotL + 1e-7)
    kd = (1 - F_term) * (1 - metallic)
    diff = kd * albedo / math.pi

    lc = torch.tensor(light_color, device=albedo.device, dtype=albedo.dtype)
    return (diff + spec) * NdotL * lc


# ===================== Tonemapping =====================

def _aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


# ===================== Lighting =====================

def _studio_lights(device):
    """6-point studio setup approximating an overcast sky hemisphere."""
    return [
        (F.normalize(torch.tensor([1.0,  1.5,  1.0], device=device, dtype=torch.float32), dim=0), [1.80, 1.72, 1.65]),
        (F.normalize(torch.tensor([0.0,  1.0,  0.0], device=device, dtype=torch.float32), dim=0), [0.50, 0.52, 0.60]),
        (F.normalize(torch.tensor([-1.0, 0.3,  0.0], device=device, dtype=torch.float32), dim=0), [0.40, 0.40, 0.48]),
        (F.normalize(torch.tensor([1.0,  0.3,  0.0], device=device, dtype=torch.float32), dim=0), [0.35, 0.35, 0.42]),
        (F.normalize(torch.tensor([0.0,  0.2, -1.0], device=device, dtype=torch.float32), dim=0), [0.30, 0.30, 0.36]),
        (F.normalize(torch.tensor([0.0, -1.0,  0.3], device=device, dtype=torch.float32), dim=0), [0.10, 0.12, 0.14]),
    ]

_AMBIENT = 0.12


# ===================== Main Render =====================

@torch.no_grad()
def render_pbr_views(
    vertices, faces, v_pbr,
    resolution=1024, num_views=4,
    elevations=None, azimuths=None,
    fov=35.0, radius=2.0,
    bg_color=(1, 1, 1),
    device='cuda',
    v_normals=None,
):
    """
    Render multi-view PBR images from per-vertex attributes.
    Returns: [num_views, 3, H, W] torch tensor in [0,1].
    """
    import nvdiffrast.torch as dr

    if elevations is None:
        elevations = [20] * num_views
    if azimuths is None:
        azimuths = [i * 360 / num_views for i in range(num_views)]

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).float()
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).int()
    if isinstance(v_pbr, np.ndarray):
        v_pbr = torch.from_numpy(v_pbr).float()

    verts = vertices.to(device)
    tris = faces.to(device)
    pbr_attr = v_pbr.to(device)

    # Per-vertex normals: use provided (e.g. from _sample_face_pbr) or compute from mesh
    if v_normals is not None:
        if isinstance(v_normals, np.ndarray):
            v_normals = torch.from_numpy(v_normals).float()
        v_normals_t = F.normalize(v_normals.to(device), dim=1)
    else:
        tris_long = tris.long()
        v0, v1, v2 = verts[tris_long[:, 0]], verts[tris_long[:, 1]], verts[tris_long[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=1)
        v_normals_t = torch.zeros_like(verts)
        v_normals_t.scatter_add_(0, tris_long[:, 0:1].expand(-1, 3), fn)
        v_normals_t.scatter_add_(0, tris_long[:, 1:2].expand(-1, 3), fn)
        v_normals_t.scatter_add_(0, tris_long[:, 2:3].expand(-1, 3), fn)
        v_normals_t = F.normalize(v_normals_t, dim=1)

    # Pack: [normal(3), albedo(3), roughness(1), metallic(1)] = 8 channels
    v_attrs = torch.cat([v_normals_t, pbr_attr], dim=1)

    glctx = dr.RasterizeCudaContext(device=device)
    proj_np = _get_proj(fov)
    proj_t = torch.from_numpy(proj_np).to(device)
    images = []

    lights = _studio_lights(device)

    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)

    for elev, azim in zip(elevations, azimuths):
        mv_np = _get_mv(elev, azim, radius)
        mv_t = torch.from_numpy(mv_np).to(device)
        mvp = proj_t @ mv_t

        # Clip-space vertices (row-vector: pos @ mvp.T)
        v_clip = _transform_pos(mvp, verts)  # [1, V, 4]

        rast, _ = dr.rasterize(glctx, v_clip, tris, resolution=[resolution, resolution])

        # Interpolate attributes
        attr_img, _ = dr.interpolate(v_attrs.unsqueeze(0).contiguous(), rast, tris)
        attr_img = attr_img[0]  # [H, W, 8]

        mask = (rast[0, :, :, 3:4] > 0).float()

        normal = F.normalize(attr_img[:, :, :3], dim=-1)
        # sRGB → linear (TRELLIS.2: gb_basecolor ** 2.2 before shading)
        albedo = attr_img[:, :, 3:6].clamp(0, 1).pow(2.2)
        roughness = attr_img[:, :, 6:7].clamp(0.04, 1)
        metallic = attr_img[:, :, 7:8].clamp(0, 1)

        # World-space positions for view direction
        pos_img, _ = dr.interpolate(verts.unsqueeze(0).contiguous(), rast, tris)
        pos_img = pos_img[0]

        # Camera position in world = -R^T @ t
        cam_pos = -mv_t[:3, :3].T @ mv_t[:3, 3]
        view_dir = F.normalize(cam_pos.unsqueeze(0).unsqueeze(0) - pos_img, dim=-1)

        # Flip back-facing normals
        ndotv = (normal * view_dir).sum(-1, keepdim=True)
        normal = torch.where(ndotv < 0, -normal, normal)

        # Shade
        color = torch.zeros_like(albedo)
        for light_dir, light_col in lights:
            color = color + _ggx_brdf(
                normal, view_dir, light_dir,
                albedo, roughness, metallic, light_col)

        color = color + _AMBIENT * albedo

        # Tonemap + gamma
        color = _aces(color)
        color = color.clamp(0, 1) ** (1.0 / 2.2)

        # Composite; flip Y: nvdiffrast row-0 = OpenGL bottom → image row-0 = top
        img = color * mask + bg * (1 - mask)
        img = torch.flip(img, dims=[0])
        images.append(img.permute(2, 0, 1))

    return torch.stack(images, dim=0)


def render_pbr_grid(
    vertices, faces, v_pbr,
    output_path,
    resolution=1024,
    num_views=4,
    **kwargs,
):
    """Render multi-view PBR and save as a grid image."""
    imgs = render_pbr_views(
        vertices, faces, v_pbr,
        resolution=resolution, num_views=num_views, **kwargs,
    )
    # imgs: [N, 3, H, W]
    N = imgs.shape[0]
    cols = min(N, 4)
    rows = (N + cols - 1) // cols
    grid = torch.ones(3, rows * resolution, cols * resolution, device=imgs.device)
    for i in range(N):
        r, c = divmod(i, cols)
        grid[:, r*resolution:(r+1)*resolution, c*resolution:(c+1)*resolution] = imgs[i]

    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(grid_np).save(output_path, quality=95)
    return output_path


def _sample_dense_block_pbr(positions, block_lut, dense_pbr, default_pbr, chunk_size=200000):
    """Trilinear sample dense [B,16,16,16,6] block PBR at world positions."""
    outputs = []
    max_grid = SAMPLE_RES - 1e-6

    for start in range(0, positions.shape[0], chunk_size):
        end = min(start + chunk_size, positions.shape[0])
        pts = positions[start:end].clamp(-0.5, 0.5 - 1e-6)
        grid = ((pts + 0.5) * SAMPLE_RES).clamp(0.0, max_grid)

        block = torch.floor(grid / BLOCK_INNER).long().clamp(0, BLOCK_GRID - 1)
        local = grid - block.float() * BLOCK_INNER
        lower = torch.floor(local).long().clamp(0, BLOCK_DIM - 1)
        upper = (lower + 1).clamp(max=BLOCK_DIM - 1)
        weight = local - lower.float()

        block_key = (
            block[:, 0] * (BLOCK_GRID ** 2)
            + block[:, 1] * BLOCK_GRID
            + block[:, 2]
        )
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
                c000 * wx0 * wy0 * wz0
                + c001 * wx0 * wy0 * wz1
                + c010 * wx0 * wy1 * wz0
                + c011 * wx0 * wy1 * wz1
                + c100 * wx1 * wy0 * wz0
                + c101 * wx1 * wy0 * wz1
                + c110 * wx1 * wy1 * wz0
                + c111 * wx1 * wy1 * wz1
            )
            sampled[valid] = sampled_valid

        outputs.append(sampled)

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def render_block_pbr_views(
    vertices,
    faces,
    block_coords,
    block_pbr,
    resolution=512,
    num_views=4,
    elevations=None,
    azimuths=None,
    fov=30.0,
    radius=2.0,
    bg_color=(0, 0, 0),
    device='cuda',
):
    """
    Render TRELLIS-style attribute snapshots from dense block PBR.

    Args:
        vertices: [V, 3]
        faces: [F, 3]
        block_coords: [B, 3]
        block_pbr: [B, 16, 16, 16, 6] or [B, 16^3 * 6]

    Returns:
        dict with base_color / metallic / roughness / alpha / mra tensors.
    """
    import nvdiffrast.torch as dr

    if elevations is None:
        elevations = [20.0] * num_views
    if azimuths is None:
        azimuths = [i * 360.0 / num_views - 16.0 for i in range(num_views)]

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).float()
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).int()
    if isinstance(block_coords, np.ndarray):
        block_coords = torch.from_numpy(block_coords).long()
    if isinstance(block_pbr, np.ndarray):
        block_pbr = torch.from_numpy(block_pbr).float()

    verts = vertices.to(device)
    tris = faces.to(device)
    coords = block_coords.to(device).long()
    dense_pbr = block_pbr.to(device).float()
    if dense_pbr.ndim == 2:
        dense_pbr = dense_pbr.reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, PBR_CHANNELS)
    elif dense_pbr.ndim == 3:
        dense_pbr = dense_pbr.reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, dense_pbr.shape[-1])

    block_lut = torch.full((BLOCK_GRID ** 3,), -1, device=device, dtype=torch.long)
    block_keys = coords[:, 0] * (BLOCK_GRID ** 2) + coords[:, 1] * BLOCK_GRID + coords[:, 2]
    block_lut[block_keys] = torch.arange(coords.shape[0], device=device)

    default_pbr = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32)
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)

    glctx = dr.RasterizeCudaContext(device=device)
    proj = torch.from_numpy(_get_proj(fov)).to(device)

    outputs = {k: [] for k in ['base_color', 'metallic', 'roughness', 'alpha', 'mra']}

    for elev, azim in zip(elevations, azimuths):
        mv = torch.from_numpy(_get_mv(elev, azim, radius)).to(device)
        mvp = proj @ mv
        v_clip = _transform_pos(mvp, verts)

        rast, _ = dr.rasterize(glctx, v_clip, tris, resolution=[resolution, resolution])
        pos_img, _ = dr.interpolate(verts.unsqueeze(0).contiguous(), rast, tris)
        pos_img = pos_img[0]

        mask = rast[0, :, :, 3] > 0
        attrs = torch.zeros((resolution, resolution, PBR_CHANNELS), device=device, dtype=torch.float32)
        if mask.any():
            attrs[mask] = _sample_dense_block_pbr(
                pos_img[mask],
                block_lut,
                dense_pbr,
                default_pbr,
            )

        # flip Y: nvdiffrast row-0 = OpenGL bottom → image row-0 = top
        attrs = torch.flip(attrs, dims=[0])
        mask_f = torch.flip(mask.unsqueeze(-1).float(), dims=[0])
        base_color = attrs[..., :3] * mask_f + bg * (1.0 - mask_f)
        metallic = attrs[..., 3:4] * mask_f
        roughness = attrs[..., 4:5] * mask_f
        alpha = attrs[..., 5:6] * mask_f
        mra = torch.cat([metallic, roughness, alpha], dim=-1)

        outputs['base_color'].append(base_color.permute(2, 0, 1))
        outputs['metallic'].append(metallic.permute(2, 0, 1))
        outputs['roughness'].append(roughness.permute(2, 0, 1))
        outputs['alpha'].append(alpha.permute(2, 0, 1))
        outputs['mra'].append(mra.permute(2, 0, 1))

    return {k: torch.stack(v, dim=0) for k, v in outputs.items()}


@torch.no_grad()
def render_block_pbr_shaded(
    vertices,
    faces,
    block_coords,
    block_pbr,
    resolution=512,
    num_views=4,
    elevations=None,
    azimuths=None,
    fov=30.0,
    radius=2.0,
    bg_color=(1, 1, 1),
    device='cuda',
):
    """
    GGX-shaded render using dense block PBR (TRELLIS.2 pipeline).

    Rasterizes the mesh, trilinearly samples PBR from the block voxel grid at
    each visible pixel, gamma-decodes base_color, applies Cook-Torrance GGX
    with 4-point analytical lighting, ACES tonemaps, and gamma-corrects output.

    Channel order in block_pbr: [base_color(3), metallic(1), roughness(1), alpha(1)]

    Returns: [num_views, 3, H, W] tensor in [0, 1].
    """
    import nvdiffrast.torch as dr

    if elevations is None:
        elevations = [20.0] * num_views
    if azimuths is None:
        azimuths = [i * 360.0 / num_views for i in range(num_views)]

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).float()
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).int()
    if isinstance(block_coords, np.ndarray):
        block_coords = torch.from_numpy(block_coords).long()
    if isinstance(block_pbr, np.ndarray):
        block_pbr = torch.from_numpy(block_pbr).float()

    verts = vertices.to(device)
    tris = faces.to(device)
    coords = block_coords.to(device).long()
    dense_pbr = block_pbr.to(device).float()
    if dense_pbr.ndim == 2:
        dense_pbr = dense_pbr.reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, PBR_CHANNELS)
    elif dense_pbr.ndim == 3:
        dense_pbr = dense_pbr.reshape(-1, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, dense_pbr.shape[-1])

    # Build block LUT
    block_lut = torch.full((BLOCK_GRID ** 3,), -1, device=device, dtype=torch.long)
    block_keys = coords[:, 0] * (BLOCK_GRID ** 2) + coords[:, 1] * BLOCK_GRID + coords[:, 2]
    block_lut[block_keys] = torch.arange(coords.shape[0], device=device)
    default_pbr = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32)

    # Per-vertex normals (area-weighted)
    tris_long = tris.long()
    v0, v1, v2 = verts[tris_long[:, 0]], verts[tris_long[:, 1]], verts[tris_long[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    v_normals = torch.zeros_like(verts)
    v_normals.scatter_add_(0, tris_long[:, 0:1].expand(-1, 3), fn)
    v_normals.scatter_add_(0, tris_long[:, 1:2].expand(-1, 3), fn)
    v_normals.scatter_add_(0, tris_long[:, 2:3].expand(-1, 3), fn)
    v_normals = F.normalize(v_normals, dim=1)

    glctx = dr.RasterizeCudaContext(device=device)
    proj = torch.from_numpy(_get_proj(fov)).to(device)
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)

    lights = _studio_lights(device)

    images = []
    for elev, azim in zip(elevations, azimuths):
        mv = torch.from_numpy(_get_mv(elev, azim, radius)).to(device)
        mvp = proj @ mv
        v_clip = _transform_pos(mvp, verts)

        rast, _ = dr.rasterize(glctx, v_clip, tris, resolution=[resolution, resolution])

        pos_img, _ = dr.interpolate(verts.unsqueeze(0).contiguous(), rast, tris)
        pos_img = pos_img[0]  # [H, W, 3]
        normal_img, _ = dr.interpolate(v_normals.unsqueeze(0).contiguous(), rast, tris)
        normal_img = normal_img[0]  # [H, W, 3]

        mask = rast[0, :, :, 3] > 0
        mask_f = mask.unsqueeze(-1).float()

        # Sample block PBR at visible pixels
        pbr_img = default_pbr.expand(resolution, resolution, PBR_CHANNELS).clone()
        if mask.any():
            pbr_img = pbr_img.clone()
            pbr_img[mask] = _sample_dense_block_pbr(
                pos_img[mask], block_lut, dense_pbr, default_pbr,
            )

        # TRELLIS.2: gamma-decode base_color before GGX (sRGB → linear)
        albedo = pbr_img[..., :3].clamp(0, 1).pow(2.2)
        metallic = pbr_img[..., 3:4].clamp(0, 1)
        roughness = pbr_img[..., 4:5].clamp(0.04, 1)

        normal = F.normalize(normal_img, dim=-1)
        cam_pos = -mv[:3, :3].T @ mv[:3, 3]
        view_dir = F.normalize(cam_pos.unsqueeze(0).unsqueeze(0) - pos_img, dim=-1)

        # Flip back-facing normals
        ndotv = (normal * view_dir).sum(-1, keepdim=True)
        normal = torch.where(ndotv < 0, -normal, normal)

        # Cook-Torrance GGX shading + ambient
        color = torch.zeros_like(albedo)
        for light_dir, light_col in lights:
            color = color + _ggx_brdf(normal, view_dir, light_dir, albedo, roughness, metallic, light_col)
        color = color + _AMBIENT * albedo

        # ACES tonemap + gamma correct, then composite
        color = _aces(color).clamp(0, 1).pow(1.0 / 2.2)
        img = color * mask_f + bg * (1.0 - mask_f)

        # flip Y: nvdiffrast row-0 = OpenGL bottom → image row-0 = top
        img = torch.flip(img, dims=[0])
        images.append(img.permute(2, 0, 1))

    return torch.stack(images, dim=0)


@torch.no_grad()
def render_mesh_textured(
    mesh_data,
    resolution=512,
    num_views=4,
    elevations=None,
    azimuths=None,
    fov=35.0,
    radius=2.0,
    bg_color=(1, 1, 1),
    device='cuda',
):
    """
    Render a multi-material textured mesh with per-pixel UV texture sampling
    (nvdiffrast dr.texture).  This preserves full texture resolution for any
    triangle size — unlike the pre-baked vertex-color approach.

    mesh_data must come from load_mesh() and contain:
        vertices, faces, face_uvs [F,3,2], face_submesh [F], submeshes list.

    Returns: [num_views, 3, H, W] tensor in [0,1].
    """
    import nvdiffrast.torch as dr

    if elevations is None:
        elevations = [20] * num_views
    if azimuths is None:
        azimuths = [i * 360 / num_views for i in range(num_views)]

    verts     = torch.from_numpy(mesh_data['vertices']).float().to(device)
    faces_t   = torch.from_numpy(mesh_data['faces']).int().to(device)
    face_uvs  = mesh_data['face_uvs']        # [F, 3, 2] numpy float32
    face_sub  = mesh_data['face_submesh']    # [F]       numpy int32
    submeshes = mesh_data['submeshes']
    F_total   = len(mesh_data['faces'])

    # ---- smooth per-vertex normals (area-weighted) ----
    tris_l = faces_t.long()
    v0, v1, v2 = verts[tris_l[:, 0]], verts[tris_l[:, 1]], verts[tris_l[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    v_nrm = torch.zeros_like(verts)
    v_nrm.scatter_add_(0, tris_l[:, 0:1].expand(-1, 3), fn)
    v_nrm.scatter_add_(0, tris_l[:, 1:2].expand(-1, 3), fn)
    v_nrm.scatter_add_(0, tris_l[:, 2:3].expand(-1, 3), fn)
    v_nrm = F.normalize(v_nrm, dim=1)

    # ---- face-expanded UV vertex buffer for dr.interpolate ----
    # uv_verts: [F*3, 2], uv_tris: [F, 3] indexing into uv_verts
    uv_verts = torch.from_numpy(face_uvs.reshape(-1, 2)).float().to(device)
    uv_tris  = torch.arange(F_total * 3, device=device, dtype=torch.int32).reshape(F_total, 3)

    # ---- face → submesh LUT (on device) ----
    face_sub_t = torch.from_numpy(face_sub.astype(np.int64)).to(device)  # [F]

    # ---- preload per-submesh albedo textures as [1, H, W, 3] float ----
    # Flip V: PIL (0,0)=top-left, nvdiffrast UV (0,0)=bottom-left
    tex_cache = {}
    for si, sm in enumerate(submeshes):
        alb = sm.get('albedo_tex')
        if alb is not None:
            arr = np.array(alb.convert('RGB'), dtype=np.float32) / 255.0
            arr = arr[::-1, :, :].copy()          # flip V
            tex_cache[si] = torch.from_numpy(arr).float().to(device).unsqueeze(0)

    glctx  = dr.RasterizeCudaContext(device=device)
    proj   = torch.from_numpy(_get_proj(fov)).to(device)
    lights = _studio_lights(device)
    bg     = torch.tensor(bg_color, device=device, dtype=torch.float32)

    images = []
    for elev, azim in zip(elevations, azimuths):
        mv  = torch.from_numpy(_get_mv(elev, azim, radius)).to(device)
        mvp = proj @ mv
        v_clip = _transform_pos(mvp, verts)

        rast, _ = dr.rasterize(glctx, v_clip, faces_t, resolution=[resolution, resolution])

        # per-pixel: normal, world-pos
        nrm_img, _ = dr.interpolate(v_nrm.unsqueeze(0), rast, faces_t)
        nrm_img = F.normalize(nrm_img[0], dim=-1)          # [H, W, 3]
        pos_img, _ = dr.interpolate(verts.unsqueeze(0), rast, faces_t)
        pos_img = pos_img[0]                                 # [H, W, 3]

        # per-pixel UV (interpolated via face-expanded buffer)
        texc, _ = dr.interpolate(uv_verts.unsqueeze(0), rast, uv_tris)
        texc = texc[0]                                       # [H, W, 2]

        mask   = rast[0, :, :, 3] > 0                       # [H, W]
        mask_f = mask.unsqueeze(-1).float()

        # face ID → submesh ID (triid is 1-indexed; 0 = background)
        tri_id   = rast[0, :, :, 3].long()                  # [H, W]
        sub_id   = torch.where(mask, face_sub_t[tri_id.clamp(min=1) - 1],
                               torch.full_like(tri_id, -1))  # [H, W]

        # composite albedo, roughness, metallic per submesh
        albedo_img = torch.full((resolution, resolution, 3), 0.5, device=device)
        rough_img  = torch.full((resolution, resolution, 1), 0.5, device=device)
        metal_img  = torch.zeros((resolution, resolution, 1), device=device)

        for si, sm in enumerate(submeshes):
            si_mask = sub_id == si                           # [H, W]
            if not si_mask.any():
                continue
            si_m = si_mask.unsqueeze(-1)                     # [H, W, 1]

            if si in tex_cache:
                # per-pixel texture sample — full resolution
                sampled = dr.texture(
                    tex_cache[si],                           # [1, H_t, W_t, 3]
                    texc.unsqueeze(0),                       # [1, H, W, 2]
                    filter_mode='linear',
                    boundary_mode='wrap',
                )[0]                                         # [H, W, 3]
                bc = torch.tensor(sm['bc_factor'], device=device)
                sampled = sampled * bc
            else:
                bc = torch.tensor(sm['bc_factor'], device=device)
                sampled = bc.expand(resolution, resolution, 3)

            albedo_img = torch.where(si_m, sampled, albedo_img)
            rough_img  = torch.where(si_m, torch.full_like(rough_img, sm['r_factor']), rough_img)
            metal_img  = torch.where(si_m, torch.full_like(metal_img, sm['m_factor']), metal_img)

        # GGX shading
        albedo    = albedo_img.clamp(0, 1).pow(2.2) * mask_f
        roughness = rough_img.clamp(0.04, 1.0)
        metallic  = metal_img.clamp(0, 1)

        cam_pos  = -mv[:3, :3].T @ mv[:3, 3]
        view_dir = F.normalize(cam_pos - pos_img, dim=-1)
        ndotv    = (nrm_img * view_dir).sum(-1, keepdim=True)
        nrm_img  = torch.where(ndotv < 0, -nrm_img, nrm_img)

        color = torch.zeros_like(albedo)
        for light_dir, light_col in lights:
            color = color + _ggx_brdf(nrm_img, view_dir, light_dir, albedo, roughness, metallic, light_col)
        color = color + _AMBIENT * albedo

        color = _aces(color).clamp(0, 1).pow(1.0 / 2.2)
        img   = color * mask_f + bg * (1.0 - mask_f)
        img   = torch.flip(img, dims=[0])
        images.append(img.permute(2, 0, 1))

    return torch.stack(images, dim=0)
