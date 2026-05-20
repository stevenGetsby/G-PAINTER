"""
法线图渲染工具 — nvdiffrast (CUDA) 渲染 + CuMesh 减面.

公开接口:
    simplify_mesh_gpu(vertices, faces, target_faces) -> (v, f) CUDA tensors
    render_random_normals_grid(ply_path, output_path, ...)
"""

import os
import numpy as np
import trimesh
import torch
from PIL import Image

from ..representations.mesh import MeshExtractResult
from .mesh_renderer import MeshRenderer
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


# ------------------------------------------------------------------
# CuMesh GPU 减面
# ------------------------------------------------------------------

def simplify_mesh_gpu(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    target_faces: int = 5000_000,
    verbose: bool = True,
):
    """
    GPU 上清理 + 减面 (CuMesh).

    Args:
        vertices: [V, 3] float CUDA tensor
        faces:    [F, 3] int/long CUDA tensor
        target_faces: 目标面数
        verbose: 是否打印信息

    Returns:
        (vertices, faces) — CUDA tensors
    """
    import cumesh

    cm = cumesh.CuMesh()
    cm.init(vertices.float().cuda(), faces.int().cuda())
    cm.remove_duplicate_faces()
    cm.remove_degenerate_faces()

    if cm.num_faces > target_faces:
        if verbose:
            print(f"  Simplifying: {cm.num_faces:,} -> {target_faces:,} faces ...")
        cm.simplify(target_faces, verbose=False)

    v, f = cm.read()
    if verbose:
        print(f"  After simplify: {v.shape[0]:,} verts, {f.shape[0]:,} faces")
    return v.float().cuda(), f.long().cuda()


def load_and_simplify(ply_path: str, target_faces: int = 500_000, device: str = "cuda", verbose: bool = True):
    """
    从文件加载 mesh -> CuMesh GPU 减面 -> 返回 MeshExtractResult.
    """
    mesh = trimesh.load(ply_path, force="mesh", process=False)
    if verbose:
        print(f"  Loaded: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).int().to(device)

    if len(mesh.faces) > target_faces:
        vertices, faces = simplify_mesh_gpu(vertices, faces, target_faces, verbose=verbose)

    return MeshExtractResult(vertices, faces)


# ------------------------------------------------------------------
# 4 视角法线 2x2 Grid (GPU 渲染)
# ------------------------------------------------------------------

def render_random_normals_grid(
    ply_path: str,
    output_path: str,
    resolution: int = 512,
    fov: float = 40.0,
    radius: float = 2.0,
    target_faces: int = 500_000,
    verbose: bool = True,
):
    """
    加载 mesh -> CuMesh 减面 -> nvdiffrast 渲染 4 随机视角法线图 -> 2x2 grid 保存.
    """
    if not os.path.exists(ply_path):
        if verbose:
            print(f"  Error: file not found: {ply_path}")
        return

    device = "cuda"

    # 1. 加载 + 减面 -> MeshExtractResult (含 face_normal)
    mesh_data = load_and_simplify(ply_path, target_faces=target_faces, device=device, verbose=verbose)

    # 2. 渲染器
    renderer = MeshRenderer(device=device)
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 0.1
    renderer.rendering_options.far = 100.0
    renderer.rendering_options.ssaa = 4

    # 3. 生成 4 个随机视角
    num_views = 4
    elevations = np.random.uniform(10, 50, num_views)
    azimuths = np.random.uniform(0, 360, num_views)

    # yaw/pitch 转弧度, fov 保持角度 (render_utils 内部会 deg2rad)
    yaws_rad = [float(np.deg2rad(a)) for a in azimuths]
    pitchs_rad = [float(np.deg2rad(e)) for e in elevations]
    radii = [float(radius)] * num_views
    fovs = [float(fov)] * num_views

    extrinsics_list, intrinsics_list = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws_rad, pitchs_rad, radii, fovs,
    )
    if not isinstance(extrinsics_list, list):
        extrinsics_list = [extrinsics_list]
        intrinsics_list = [intrinsics_list]

    # 4. 渲染
    images = []
    if verbose:
        print(f"  Rendering {num_views} views (nvdiffrast) ...")
    for i, (extr, intr) in enumerate(zip(extrinsics_list, intrinsics_list)):
        try:
            res = renderer.render(mesh_data, extr, intr, return_types=["normal"])
            normal_np = res["normal"].permute(1, 2, 0).detach().cpu().numpy()
            img = Image.fromarray((np.clip(normal_np, 0, 1) * 255).astype(np.uint8))
            images.append(img)
        except Exception as e:
            if verbose:
                print(f"  View {i} render error: {e}")
            images.append(Image.new("RGB", (resolution, resolution), (0, 0, 0)))

    # 5. 拼图 2x2
    gw, gh = resolution * 2, resolution * 2
    grid = Image.new("RGB", (gw, gh))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (resolution, 0))
    grid.paste(images[2], (0, resolution))
    grid.paste(images[3], (resolution, resolution))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid.save(output_path)
    if verbose:
        print(f"  Normal map saved to {output_path}")
