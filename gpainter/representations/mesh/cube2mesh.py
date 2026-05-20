import open3d as o3d
import torch
import numpy as np
import trimesh
import mcubes
from numba import njit
from tqdm import tqdm
import time
import pymeshlab
from ...modules.sparse import SparseTensor

class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None

    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals

# -----------------------------------------------------------------------------
# Numba Accelerated Accumulation (Copied from dataset_toolkits/mesh2sdf.py)
# -----------------------------------------------------------------------------
@njit(parallel=False)
def accumulate_numba(indices, tokens_4d, volume_sum, count_grid,
                     global_res, block_size, padding, padded_size):
    num_blocks = indices.shape[0]
    for i in range(num_blocks):
        ix, iy, iz = indices[i]

        base_x = int(ix * block_size)
        base_y = int(iy * block_size)
        base_z = int(iz * block_size)

        gs_x = int(base_x - padding)
        gs_y = int(base_y - padding)
        gs_z = int(base_z - padding)

        ge_x = int(base_x + block_size + padding)
        ge_y = int(base_y + block_size + padding)
        ge_z = int(base_z + block_size + padding)

        cs_x = max(0, gs_x)
        ce_x = min(global_res, ge_x)
        cs_y = max(0, gs_y)
        ce_y = min(global_res, ge_y)
        cs_z = max(0, gs_z)
        ce_z = min(global_res, ge_z)

        ls_x = cs_x - gs_x
        le_x = padded_size - (ge_x - ce_x)
        ls_y = cs_y - gs_y
        le_y = padded_size - (ge_y - ce_y)
        ls_z = cs_z - gs_z
        le_z = padded_size - (ge_z - ce_z)

        token_slice = tokens_4d[i, ls_x:le_x, ls_y:le_y, ls_z:le_z]

        volume_sum[cs_x:ce_x, cs_y:ce_y, cs_z:ce_z] += token_slice
        count_grid[cs_x:ce_x, cs_y:ce_y, cs_z:ce_z] += 1

class MeshExtractor:
    def __init__(
        self,
        global_res=1024,
        block_size=16,
        padding=2
    ):
        self.global_res = global_res
        self.block_size = block_size
        self.padding = padding
        self.padded_size = block_size + 2 * padding

    def sparse2mesh(self, x_0: SparseTensor):
        """
        Convert SparseTensor (UDF blocks) to a list of MeshExtractResult.

        Args:
            x_0: SparseTensor with coords (B, 4) and feats (B, PADDED_SIZE^3)

        Returns:
            List[MeshExtractResult]: One mesh result per batch item.
        """
        # Move data to CPU for processing
        coords = x_0.coords.cpu().numpy()
        feats = x_0.feats.cpu().numpy()

        batch_size = x_0.shape[0]
        results = []

        for b in range(batch_size):
            # Extract data for current batch index
            mask = coords[:, 0] == b
            if not np.any(mask):
                results.append(None)
                continue

            b_indices = coords[mask, 1:] # (M, 3)
            b_tokens = feats[mask]       # (M, 8000)

            # Allocate large volume on CPU (Warning: High Memory Usage ~8GB)
            volume_sum = np.zeros((self.global_res, self.global_res, self.global_res), dtype=np.float32)
            count_grid = np.zeros((self.global_res, self.global_res, self.global_res), dtype=np.int32)

            # Reshape tokens to 4D for numba function
            tokens_4d = b_tokens.astype(np.float32).reshape(-1, self.padded_size, self.padded_size, self.padded_size)

            # Accumulate blocks into global volume
            accumulate_numba(
                b_indices.astype(np.int32),
                tokens_4d,
                volume_sum,
                count_grid,
                self.global_res, self.block_size, self.padding, self.padded_size
            )

            # Average overlapping regions
            mask_vol = count_grid > 0
            volume_sum[~mask_vol] = 1.0 # Background distance
            volume_sum[mask_vol] /= count_grid[mask_vol]

            # Run Marching Cubes
            # Threshold 0.3 corresponds to the normalization used in data generation
            verts, faces = mcubes.marching_cubes(volume_sum, 0.3)

            # Free memory immediately
            del volume_sum, count_grid

            if verts.shape[0] > 0:
                # Normalize vertices to [-0.5, 0.5]
                verts = (verts / self.global_res) - 0.5

                # Create result object (vertices/faces on GPU for renderer)
                mesh_res = MeshExtractResult(
                    vertices=torch.from_numpy(verts).float().cuda(),
                    faces=torch.from_numpy(faces.astype(np.int64)).cuda(),
                    res=self.global_res
                )
                results.append(mesh_res)
            else:
                results.append(None)

        return results