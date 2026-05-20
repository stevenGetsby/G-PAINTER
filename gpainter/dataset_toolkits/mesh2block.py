import os
import time
import argparse
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d


BLOCK_GRID = 64                                 # 每轴 block 数
BLOCK_INNER = 15                                # 每 block 核心采样间隔数

BLOCK_DIM = BLOCK_INNER + 1                     # 每 block 采样维度 (顶点数)
SAMPLE_RES = BLOCK_GRID * BLOCK_INNER           # 全局采样分辨率
VOXEL_SIZE = 1.0 / SAMPLE_RES

# UDF 截断: 按体素数, 不按 block 宽度
UDF_TRUNC_VOXELS = 5                            # 截断体素数
TRUNCATION = UDF_TRUNC_VOXELS * VOXEL_SIZE      # 5/896 ≈ 0.00558
MC_THRESHOLD = 1.0 / UDF_TRUNC_VOXELS           # 1/5 = 0.2 (1 voxel in UDF space)
SURFACE_THRESHOLD = 2 * MC_THRESHOLD            # 2/5 = 0.4 (MC 插值带精确覆盖)

# GPU BVH 面数上限
MAX_FACES_FOR_BVH = 1000_000

# Per-block sub-mask: 从 UDF 直接提取, 标记 MC 表面穿越的子区域
SUBMASK_RES = 8                                 # sub-mask 每轴分辨率 (occ8, 需整除 BLOCK_DIM)
SUBMASK_STRIDE = BLOCK_DIM // SUBMASK_RES       # 每个 sub-cell 覆盖的 UDF 体素数
SUBMASK_DIM = SUBMASK_RES ** 3                  # per-block mask 展平维度

# ---- PBR ----
PBR_CHANNELS = 6                                # base_color(3) | metallic(1) | roughness(1) | alpha(1)

# ---- UDF query batch ----
MAX_QUERY_PTS = 50_000_000                      # max points per BVH query batch

assert BLOCK_DIM % SUBMASK_RES == 0, f"BLOCK_DIM({BLOCK_DIM}) must be divisible by SUBMASK_RES({SUBMASK_RES})"

# ---- 全局命名前缀 (文件夹 + metadata 列名) ----
COL_PREFIX = f'{BLOCK_GRID}_{BLOCK_INNER}_occ{SUBMASK_RES}'
BLOCK_FOLDER = f'blocks_{COL_PREFIX}'


# ===================== sub-mask: 从 UDF 直接提取 =====================

def extract_submask_from_udf(udf: np.ndarray) -> np.ndarray:
    """
    Args:
        udf: [N, BLOCK_DIM³] float, 归一化 UDF ∈ [0, 1]
    Returns:
        submask: [N, SUBMASK_DIM] float32, binary (0/1)
    """
    N = len(udf)
    R, S = SUBMASK_RES, SUBMASK_STRIDE
    vol = udf.reshape(N, R, S, R, S, R, S)
    sub_min = vol.min(axis=(2, 4, 6))                           # [N, R, R, R]
    submask = (sub_min < SURFACE_THRESHOLD).astype(np.float32)
    return submask.reshape(N, -1)


# ===================== mesh I/O =====================

def normalize_mesh_o3d(mesh: o3d.t.geometry.TriangleMesh):
    v = mesh.vertex.positions.numpy()
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    center = (vmin + vmax) / 2
    extent = (vmax - vmin).max()
    scale = 0.98 / max(extent, 1e-12)

    v = (v - center) * scale
    mesh.vertex.positions = o3d.core.Tensor(v, dtype=o3d.core.float32)
    return mesh, center, scale


def load_mesh(path: str, verbose: bool = True):
    mesh = o3d.t.io.read_triangle_mesh(path)
    if len(mesh.vertex.positions) == 0:
        legacy = o3d.io.read_triangle_mesh(path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)

    mesh, center, scale = normalize_mesh_o3d(mesh)

    if verbose:
        v = mesh.vertex.positions.numpy()
        f = mesh.triangle.indices.numpy()
        print(f"✅ Loaded: {len(v)} verts, {len(f)} faces")

    return mesh, center, scale


# ===================== mesh simplification =====================

def _simplify_for_bvh(mesh: o3d.t.geometry.TriangleMesh,
                       max_faces: int = MAX_FACES_FOR_BVH,
                       verbose: bool = True,
                       device: str = 'cpu') -> o3d.t.geometry.TriangleMesh:
    """面数超过 max_faces 时用 CuMesh GPU 简化, 避免 cubvh BVH 栈溢出。"""
    n_faces = len(mesh.triangle.indices.numpy())
    if n_faces <= max_faces:
        return mesh

    if verbose:
        print(f"⚠️  Mesh has {n_faces} faces (>{max_faces}), simplifying for BVH …")

    from cumesh import CuMesh
    v_np = mesh.vertex.positions.numpy().astype(np.float32)
    f_np = mesh.triangle.indices.numpy().astype(np.int32)
    dev = device if device != 'cpu' else 'cuda'
    v_cuda = torch.from_numpy(v_np).float().to(dev)
    f_cuda = torch.from_numpy(f_np).int().to(dev)
    cm = CuMesh()
    cm.init(v_cuda, f_cuda)
    cm.simplify(target_num_faces=max_faces, verbose=verbose)
    v_new = cm.vertices.cpu().numpy()
    f_new = cm.faces.cpu().numpy().astype(np.int32)

    simplified = o3d.t.geometry.TriangleMesh()
    simplified.vertex.positions = o3d.core.Tensor(v_new, dtype=o3d.core.float32)
    simplified.triangle.indices = o3d.core.Tensor(f_new, dtype=o3d.core.int32)

    if verbose:
        print(f"   ✅ Simplified: {n_faces} → {len(f_new)} faces (GPU)")
    return simplified


# ===================== block detection =====================

def detect_active_blocks(mesh, verbose: bool = True,
                         device: str = 'cpu', bvh=None) -> np.ndarray:
    """在 BLOCK_GRID³ 网格上检测含表面的 block"""
    spacing = 1.0 / BLOCK_GRID
    vals = torch.linspace(-0.5 + spacing / 2, 0.5 - spacing / 2, BLOCK_GRID)
    gx, gy, gz = torch.meshgrid(vals, vals, vals, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    if device != 'cpu' and bvh is not None:
        # GPU path: cubvh
        pts_cuda = pts.to(device)
        d, _, _ = bvh.unsigned_distance(pts_cuda, return_uvw=False)
        d = d.cpu()
    else:
        # CPU path: Open3D
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        pts_o3d = o3d.core.Tensor(pts.numpy(), dtype=o3d.core.float32,
                                   device=o3d.core.Device("CPU:0"))
        d = torch.from_numpy(scene.compute_distance(pts_o3d).numpy())

    threshold = (np.sqrt(3) * spacing) / 2.0 * 1.1
    active = d <= threshold

    idx = torch.nonzero(active).squeeze(1)
    ix = idx // (BLOCK_GRID ** 2)
    iy = (idx % (BLOCK_GRID ** 2)) // BLOCK_GRID
    iz = idx % BLOCK_GRID
    coords = torch.stack([ix, iy, iz], dim=1).cpu().numpy().astype(np.int32)

    if verbose:
        print(f"✅ Active blocks: {len(coords)}")
    return coords


# ===================== UDF computation =====================

def compute_block_udf(mesh, block_coords: np.ndarray,
                      verbose: bool = True,
                      device: str = 'cpu', bvh=None) -> np.ndarray:
    """
    计算每个 block 的 BLOCK_DIM³ UDF。

    坐标映射:
      block (bx, by, bz) in BLOCK_GRID³
      → 局部偏移 [0, BLOCK_DIM)
      → 全局采样坐标 = bx * BLOCK_INNER + offset
      → 物理坐标 = global_idx * VOXEL_SIZE - 0.5
    """
    use_gpu = (device != 'cpu' and bvh is not None)

    if not use_gpu:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

    n = len(block_coords)

    lr = torch.arange(BLOCK_DIM, dtype=torch.long)
    lx, ly, lz = torch.meshgrid(lr, lr, lr, indexing="ij")
    local = torch.stack([lx, ly, lz], dim=-1).reshape(-1, 3)  # [BLOCK_DIM³, 3]

    if use_gpu:
        # GPU 快速路径: 一次构造所有查询点, 尽量减少 batch 和拷贝次数
        b_all = torch.from_numpy(block_coords).long()          # [N, 3]
        gvidx = b_all[:, None, :] * BLOCK_INNER + local[None, :, :]
        q_all = (gvidx.float() * VOXEL_SIZE) - 0.5
        q_flat = q_all.reshape(-1, 3).clamp(-0.5, 0.5 - 1e-6)

        # cubvh 内部自动分 batch, 一次调用即可
        # 如果显存不够 (N > ~20K blocks ≈ 80M 点), 手动分 batch
        total_pts = q_flat.shape[0]
        MAX_PTS = 50_000_000  # 50M 点 ≈ ~600MB fp32

        if total_pts <= MAX_PTS:
            q_cuda = q_flat.to(device)
            d, _, _ = bvh.unsigned_distance(q_cuda, return_uvw=False)
            d_np = d.cpu().numpy()
        else:
            d_parts = []
            bs_pts = MAX_PTS
            it = range(0, total_pts, bs_pts)
            if verbose:
                it = tqdm(it, desc="Computing Block UDF (GPU)")
            for s in it:
                e = min(s + bs_pts, total_pts)
                q_cuda = q_flat[s:e].to(device)
                d_chunk, _, _ = bvh.unsigned_distance(q_cuda, return_uvw=False)
                d_parts.append(d_chunk.cpu().numpy())
            d_np = np.concatenate(d_parts)

        out = np.clip(d_np / TRUNCATION, 0.0, 1.0).reshape(n, -1).astype(np.float32)
        return out

    # CPU 路径
    out = np.zeros((n, BLOCK_DIM ** 3), dtype=np.float32)
    bs = 5000

    it = range(0, n, bs)
    if verbose:
        it = tqdm(it, desc="Computing Block UDF (CPU)")

    for s in it:
        e = min(s + bs, n)
        b = torch.from_numpy(block_coords[s:e]).long()

        gvidx = b[:, None, :] * BLOCK_INNER + local[None, :, :]
        q = (gvidx.float() * VOXEL_SIZE) - 0.5
        q = q.reshape(-1, 3).clamp(-0.5, 0.5 - 1e-6)

        dev = o3d.core.Device("CPU:0")
        q_o3d = o3d.core.Tensor(q.numpy(), dtype=o3d.core.float32, device=dev)
        d = scene.compute_distance(q_o3d).numpy()

        udf = np.clip(d / TRUNCATION, 0.0, 1.0)
        out[s:e] = udf.reshape(e - s, -1).astype(np.float32)

    return out


# ===================== reconstruction =====================

def extract_voxels(coords, fine_feats, keep_band=0.03, verbose=True):
    """从 BLOCK_DIM³ block 提取 MC 输入"""
    coords = np.asarray(coords, dtype=np.int32)
    n = len(coords)

    F_core = np.asarray(fine_feats, dtype=np.float32).reshape(
        n, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM,
    )

    vx, vy, vz = np.meshgrid(
        np.arange(BLOCK_INNER),
        np.arange(BLOCK_INNER),
        np.arange(BLOCK_INNER),
        indexing="ij",
    )
    local_vox = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)
    lx, ly, lz = local_vox[:, 0], local_vox[:, 1], local_vox[:, 2]

    all_coords, all_logits = [], []

    for bi in range(n):
        V = F_core[bi]

        v0 = V[lx,     ly,     lz    ]
        v1 = V[lx + 1, ly,     lz    ]
        v2 = V[lx + 1, ly + 1, lz    ]
        v3 = V[lx,     ly + 1, lz    ]
        v4 = V[lx,     ly,     lz + 1]
        v5 = V[lx + 1, ly,     lz + 1]
        v6 = V[lx + 1, ly + 1, lz + 1]
        v7 = V[lx,     ly + 1, lz + 1]
        logits = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=1).astype(np.float32)

        valid = np.isfinite(logits).all(axis=1) & (
            logits.min(axis=1) < (MC_THRESHOLD + keep_band)
        )
        if not np.any(valid):
            continue

        bc = coords[bi].astype(np.int64)
        g = bc[None, :] * BLOCK_INNER + local_vox
        all_coords.append(torch.from_numpy(g[valid]).long())
        all_logits.append(torch.from_numpy(logits[valid]).float())

    if not all_coords:
        return torch.empty(0, 3, dtype=torch.long), torch.empty(0, 8, dtype=torch.float32)

    out_c = torch.cat(all_coords, dim=0)
    out_l = torch.cat(all_logits, dim=0)

    if verbose:
        print(f"✅ Voxels for MC: {len(out_c):,}")
    return out_c, out_l


def reconstruct_mesh(npz_path: str, ply_path: str,
                     keep_band: float = 0.03, verbose: bool = True):
    """从 npz 重建 mesh"""
    import cubvh

    if verbose:
        print(f"\n{'=' * 60}")
        print("Reconstructing with Sparse Marching Cubes")
        print(f"{'=' * 60}")

    t0 = time.time()
    data = np.load(npz_path)

    raw_feats = data["fine_feats"]
    if raw_feats.dtype == np.float16:
        raw_feats = raw_feats.astype(np.float32)

    all_coords, all_logits = extract_voxels(
        coords=data["coords"],
        fine_feats=raw_feats,
        keep_band=keep_band,
        verbose=verbose,
    )

    if len(all_coords) == 0:
        print("⚠️ No voxels to reconstruct")
        return

    if verbose:
        print(f"✅ Total voxels: {len(all_coords):,}")
        print("🔨 Running Sparse Marching Cubes...")

    v, f = cubvh.sparse_marching_cubes(all_coords.cuda(), all_logits.cuda(), MC_THRESHOLD)
    v = v.cpu().numpy()
    f = f.cpu().numpy()

    v = (v / SAMPLE_RES) - 0.5

    # 后处理: CuMesh GPU 清理 + Open3D Laplacian 平滑
    from cumesh import CuMesh
    v_cuda = torch.from_numpy(v.astype(np.float32)).cuda()
    f_cuda = torch.from_numpy(f.astype(np.int32)).cuda()
    cm = CuMesh()
    cm.init(v_cuda, f_cuda)
    cm.remove_duplicate_faces()
    cm.remove_degenerate_faces()
    cm.remove_small_connected_components(min_area=1e-6)
    v_clean = cm.vertices.cpu().numpy().astype(np.float64)
    f_clean = cm.faces.cpu().numpy().astype(np.int32)

    # Laplacian 平滑: Open3D legacy
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(v_clean)
    legacy.triangles = o3d.utility.Vector3iVector(f_clean)
    legacy = legacy.filter_smooth_laplacian(number_of_iterations=3)
    o3d.io.write_triangle_mesh(ply_path, legacy)

    if verbose:
        print(f"✅ Saved to {ply_path}")
        print(f"⏱️  Total time: {time.time() - t0:.2f}s")
        print(f"{'=' * 60}\n")


# 兼容旧函数名
reconstruct_adaptive_mc = reconstruct_mesh


# ===================== 合并检测 + UDF 计算 =====================

def _detect_and_compute_udf(mesh, verbose=True, device='cpu', bvh=None):
    """
    合并 detect_active_blocks + compute_block_udf 为一次 BVH 查询.

    原来的流程:
      1. detect: 查询 64³=262K 个 block 中心点的距离 → 找到 active blocks
      2. compute: 对 active blocks 查询 N×BLOCK_DIM³ 个精细点的 UDF

    优化: 在 GPU 模式下, 直接把检测和 UDF 计算合并,
    用粗糙的 block 中心距离预筛选, 然后只对 active blocks 算精细 UDF.
    整个过程只构造一次查询张量, 减少 CPU↔GPU 拷贝次数.
    """
    use_gpu = (device != 'cpu' and bvh is not None)

    if not use_gpu:
        # CPU 路径: 保持原来的两步流程
        coords = detect_active_blocks(mesh, verbose=verbose, device=device, bvh=bvh)
        udf = compute_block_udf(mesh, coords, verbose=verbose, device=device, bvh=bvh)
        return coords, udf

    # ---- GPU 合并路径 ----
    spacing = 1.0 / BLOCK_GRID

    # Step 1: 粗筛 - block 中心点距离 (复用已建好的 bvh)
    vals = torch.linspace(-0.5 + spacing / 2, 0.5 - spacing / 2, BLOCK_GRID)
    gx, gy, gz = torch.meshgrid(vals, vals, vals, indexing="ij")
    center_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)

    d_center, _, _ = bvh.unsigned_distance(center_pts, return_uvw=False)
    threshold = (np.sqrt(3) * spacing) / 2.0 * 1.1
    active = d_center <= threshold

    idx = torch.nonzero(active).squeeze(1)
    ix = idx // (BLOCK_GRID ** 2)
    iy = (idx % (BLOCK_GRID ** 2)) // BLOCK_GRID
    iz = idx % BLOCK_GRID
    coords = torch.stack([ix, iy, iz], dim=1).cpu().numpy().astype(np.int32)

    if verbose:
        print(f"✅ Active blocks: {len(coords)}")

    # Step 2: 精细 UDF - 直接在 GPU 上构造查询点
    n = len(coords)
    lr = torch.arange(BLOCK_DIM, dtype=torch.long)
    lx, ly, lz = torch.meshgrid(lr, lr, lr, indexing="ij")
    local = torch.stack([lx, ly, lz], dim=-1).reshape(-1, 3)  # [BLOCK_DIM³, 3]

    b_all = torch.from_numpy(coords).long()
    gvidx = b_all[:, None, :] * BLOCK_INNER + local[None, :, :]
    q_all = (gvidx.float() * VOXEL_SIZE) - 0.5
    q_flat = q_all.reshape(-1, 3).clamp(-0.5, 0.5 - 1e-6)

    total_pts = q_flat.shape[0]
    MAX_PTS = 50_000_000

    if total_pts <= MAX_PTS:
        q_cuda = q_flat.to(device)
        d, _, _ = bvh.unsigned_distance(q_cuda, return_uvw=False)
        d_np = d.cpu().numpy()
    else:
        d_parts = []
        bs_pts = MAX_PTS
        it = range(0, total_pts, bs_pts)
        if verbose:
            it = tqdm(it, desc="Computing Block UDF (GPU)")
        for s in it:
            e = min(s + bs_pts, total_pts)
            q_cuda = q_flat[s:e].to(device)
            d_chunk, _, _ = bvh.unsigned_distance(q_cuda, return_uvw=False)
            d_parts.append(d_chunk.cpu().numpy())
        d_np = np.concatenate(d_parts)

    udf = np.clip(d_np / TRUNCATION, 0.0, 1.0).reshape(n, -1).astype(np.float32)

    if verbose:
        print(f"✅ Block UDF computed: {n} blocks × {BLOCK_DIM}³")

    return coords, udf


# ===================== pipeline =====================

def generate_adaptive_udf(input_path: str, output_npz_path: str,
                          verbose: bool = True, device: str = 'cpu'):
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Generating Block UDF Data (device={device})")
        print(f"{'=' * 60}")
        print(f"📏 Prefix: {COL_PREFIX} → folder: {BLOCK_FOLDER}")
        print(f"📏 Block grid: {BLOCK_GRID}³, Block dim: {BLOCK_DIM}³")
        print(f"📏 Sub-mask: {SUBMASK_RES}³={SUBMASK_DIM}d (stride={SUBMASK_STRIDE})")
        print(f"📏 Sample res: {SAMPLE_RES}³, Voxel: {VOXEL_SIZE:.6f}")
        print(f"📏 Truncation: {TRUNCATION:.6f}, "
              f"Surface: {SURFACE_THRESHOLD}, MC: {MC_THRESHOLD}")

    t0 = time.time()
    mesh, _, _ = load_mesh(input_path, verbose=verbose)

    # GPU 模式下, 如果面数太多则先简化, 防止 cubvh BVH 栈溢出
    if device != 'cpu':
        mesh = _simplify_for_bvh(mesh, verbose=verbose, device=device)

    # 构建 GPU BVH (如果使用 GPU)
    bvh = None
    if device != 'cpu':
        try:
            import cubvh
            verts = torch.from_numpy(mesh.vertex.positions.numpy()).float().to(device)
            faces = torch.from_numpy(mesh.triangle.indices.numpy()).int().to(device)
            bvh = cubvh.cuBVH(verts, faces)
            if verbose:
                print(f"🚀 Using GPU ({device}) via cubvh")
        except Exception as e:
            if verbose:
                print(f"⚠️ cubvh init failed ({e}), falling back to CPU")
            device = 'cpu'

    # 1. 检测活跃 block + 计算 UDF (合并查询, 减少 BVH 调用次数)
    coords, udf = _detect_and_compute_udf(
        mesh, verbose=verbose, device=device, bvh=bvh
    )

    # 2. 只保留表面 block (任意 UDF < 阈值)
    min_udf = udf.min(axis=1)
    has_surface = min_udf < SURFACE_THRESHOLD
    n_total = len(coords)
    n_surface = int(has_surface.sum())

    coords_s = coords[has_surface]
    udf_s = udf[has_surface]

    # 从 UDF 提取 sub-mask (带膨胀)
    submask_s = extract_submask_from_udf(udf_s)

    # 3. 保存
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        coords=coords_s,                        # (M, 3) int32
        fine_feats=udf_s.astype(np.float16),     # (M, BLOCK_DIM³) float16
        submask=submask_s,                       # (M, SUBMASK_DIM) float32
    )

    # 显式释放 GPU 显存
    del bvh
    if device != 'cpu':
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n✅ Saved to {output_npz_path}")
        print(f"   - total blocks: {n_total}, surface blocks: {n_surface} "
              f"({100 * n_surface / max(n_total, 1):.1f}%)")
        print(f"   - coords: {coords_s.shape}")
        print(f"   - fine: {udf_s.shape} ({BLOCK_DIM}³)")
        print(f"⏱️  {time.time() - t0:.2f}s")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./assets/mesh2.ply")
    parser.add_argument("--output_dir", type=str, default="./adaptive_udf")
    parser.add_argument("--keep_band", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda' / 'cuda:0' etc.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_path = os.path.join(args.output_dir, "adaptive_udf.npz")
    ply_path = os.path.join(args.output_dir, "reconstructed.ply")

    generate_adaptive_udf(args.input, npz_path, verbose=True, device=args.device)
    reconstruct_mesh(npz_path, ply_path, keep_band=args.keep_band, verbose=True)

    # 渲染法线图
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from gpainter.renderers.normal_render import render_random_normals_grid
    normal_path = os.path.join(args.output_dir, "normal.jpg")
    render_random_normals_grid(ply_path, normal_path, resolution=1024, radius=1.75)

    print("✅ All done!")