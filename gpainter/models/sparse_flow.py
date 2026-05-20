from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.norm import LayerNorm32
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock, PooledSparseTransformerCrossBlock
from ..dataset_toolkits.mesh2block import BLOCK_DIM
from .dense_flow import TimestepEmbedder


class SparseFlowModel(nn.Module):
    """
    Sparse flow transformer for Stage 3 (UDF refinement).

        Config via resolution:
            Stage 2: resolution=SUBMASK_RES (4), bottleneck_dim=0, submask_resolution=0
            Stage 3: resolution=BLOCK_DIM (16), bottleneck_dim=128, submask_resolution=SUBMASK_RES (4)
    """

    # 6-面邻域偏移
    _FACE_OFFSETS = torch.tensor([
        [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
    ], dtype=torch.long)  # [6, 3]

    # 26-邻域偏移 (6面 + 12棱 + 8角)
    _NEIGHBOR_OFFSETS = torch.tensor([
        [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],           # 6 面
        [-1,-1,0],[-1,1,0],[1,-1,0],[1,1,0],                           # 12 棱
        [-1,0,-1],[-1,0,1],[1,0,-1],[1,0,1],
        [0,-1,-1],[0,-1,1],[0,1,-1],[0,1,1],
        [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],                       # 8 角
        [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1],
    ], dtype=torch.long)  # [26, 3]

    @staticmethod
    @torch.no_grad()
    def _compute_coarse_udf_prior(coords: torch.Tensor, layout, block_dim: int) -> torch.Tensor:
        """
        从 sparse coords 解析计算粗糙 UDF 模板, 作为 per-token 条件信号.

        原理: 如果一个 block 的某个面没有邻居, 表面大概率经过那个面.
        据此构造每个 block 内 BLOCK_DIM³ 个体素的近似距离场.

        Args:
            coords: [T, 4] (batch_idx, x, y, z)
            layout: list of slices
            block_dim: BLOCK_DIM (e.g. 16)

        Returns:
            prior: [T, block_dim³] float, 粗糙 UDF ∈ [0, 1]
        """
        T = coords.shape[0]
        D = block_dim
        device = coords.device
        face_offsets = SparseFlowModel._FACE_OFFSETS.to(device)  # [6, 3]

        # 预计算: 每个体素到 6 个面的归一化距离 [D, D, D, 6]
        # face 0(-x): dist = i/(D-1),  face 1(+x): dist = 1-i/(D-1)
        # face 2(-y): dist = j/(D-1),  face 3(+y): dist = 1-j/(D-1)
        # face 4(-z): dist = k/(D-1),  face 5(+z): dist = 1-k/(D-1)
        lin = torch.linspace(0, 1, D, device=device)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')  # [D,D,D]
        face_dists = torch.stack([
            gx, 1 - gx,       # -x, +x
            gy, 1 - gy,       # -y, +y
            gz, 1 - gz,       # -z, +z
        ], dim=-1)  # [D, D, D, 6]
        face_dists_flat = face_dists.reshape(-1, 6)  # [D³, 6]

        prior = torch.ones(T, D**3, device=device)  # 默认: UDF=1 (远离表面)

        for b, sl in enumerate(layout):
            bc = coords[sl]  # [N_b, 4]
            xyz = bc[:, 1:].long()  # [N_b, 3]
            N_b = xyz.shape[0]
            if N_b == 0:
                continue

            # Hash 查找 6 面邻居
            P1, P2 = 100003, 1009
            keys = xyz[:, 0] * P1 + xyz[:, 1] * P2 + xyz[:, 2]
            sorted_keys, _ = keys.sort()

            nb_xyz = xyz.unsqueeze(1) + face_offsets.unsqueeze(0)  # [N_b, 6, 3]
            nb_keys = nb_xyz[:, :, 0] * P1 + nb_xyz[:, :, 1] * P2 + nb_xyz[:, :, 2]

            flat_nb = nb_keys.reshape(-1)
            pos = torch.searchsorted(sorted_keys, flat_nb).clamp(max=N_b - 1)
            has_neighbor = (sorted_keys[pos] == flat_nb).reshape(N_b, 6)  # [N_b, 6]

            # 没有邻居的面 → 表面经过那里 → 距离 = 到该面的距离
            # 有邻居的面 → 远离表面 → 距离 = 1.0 (不贡献)
            boundary_dist = face_dists_flat.unsqueeze(0).expand(N_b, -1, -1).clone()  # [N_b, D³, 6]
            # 有邻居的面: 距离设为 1.0 (不影响 min)
            has_nb_expanded = has_neighbor.unsqueeze(1).expand(-1, D**3, -1)  # [N_b, D³, 6]
            boundary_dist[has_nb_expanded] = 1.0

            # 取 6 个面距离的最小值 → 粗糙 UDF
            block_prior = boundary_dist.min(dim=-1).values  # [N_b, D³]
            prior[sl] = block_prior

        return prior

    @staticmethod
    @torch.no_grad()
    def _compute_neighbor_occupancy(coords: torch.Tensor, layout) -> torch.Tensor:
        """
        从 sparse coords 计算每个 block 的 26 邻域 occupancy pattern.

        Args:
            coords: [T, 4] (batch_idx, x, y, z)
            layout: list of slices

        Returns:
            occ: [T, 26] float, 0/1 表示邻居是否存在
        """
        T = coords.shape[0]
        device = coords.device
        offsets = SparseFlowModel._NEIGHBOR_OFFSETS.to(device)  # [26, 3]

        occ = torch.zeros(T, 26, device=device)

        for b, sl in enumerate(layout):
            bc = coords[sl]  # [N_b, 4]
            xyz = bc[:, 1:].long()  # [N_b, 3]
            N_b = xyz.shape[0]
            if N_b == 0:
                continue

            # 用 hash 做快速查找
            P1, P2 = 100003, 1009
            keys = xyz[:, 0] * P1 + xyz[:, 1] * P2 + xyz[:, 2]  # [N_b]

            # 26 方向的邻居坐标 → hash
            nb_xyz = xyz.unsqueeze(1) + offsets.unsqueeze(0)  # [N_b, 26, 3]
            nb_keys = (nb_xyz[:, :, 0] * P1
                     + nb_xyz[:, :, 1] * P2
                     + nb_xyz[:, :, 2])  # [N_b, 26]

            # 排序 + searchsorted 查找
            sorted_keys, sort_idx = keys.sort()
            flat_nb = nb_keys.reshape(-1)
            pos = torch.searchsorted(sorted_keys, flat_nb).clamp(max=N_b - 1)
            matched = sorted_keys[pos] == flat_nb

            occ[sl] = matched.reshape(N_b, 26).float()

        return occ

    def __init__(
        self,
        resolution: int = 0,
        bottleneck_dim: int = 0,
        submask_resolution: int = 0,
        model_channels: int = 256,
        cond_channels: int = 1024,
        num_blocks: int = 8,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "rope",
        attn_mode: str = "windowed",
        window_size: int = 16,
        full_attn_threshold: int = 8192,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        num_moe_layers: int = 0,
        num_experts: int = 8,
        moe_top_k: int = 2,
        cross_attn_topk: int = 0,
        num_context_registers: int = 0,
        context_start_block: int = 0,
        pool_stride: int = 8,
        pool_stride_coarse: int = 0,
        num_pooled_layers: int = 0,
        upsample_to: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_context_registers = num_context_registers
        self.context_start_block = context_start_block

        # resolution → token_dim (default: BLOCK_DIM for Stage 3)
        resolution = resolution if resolution > 0 else BLOCK_DIM
        token_dim = kwargs.pop('token_dim_override', None) or resolution ** 3
        self.resolution = resolution
        self.token_dim = token_dim

        # submask conditioning (0 = disabled)
        submask_dim = kwargs.pop('submask_dim_override', None) or \
            (submask_resolution ** 3 if submask_resolution > 0 else 0)
        self.submask_resolution = submask_resolution

        if kwargs:
            print(f"[SparseFlowModel] ignored: {list(kwargs.keys())}")

        # Input embedding: bottleneck or direct
        if bottleneck_dim > 0:
            self.embed = nn.Sequential(
                nn.Linear(token_dim, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, model_channels),
            )
        else:
            self.embed = nn.Linear(token_dim, model_channels)

        # Optional sub-mask conditioning (Stage 3 only)
        if submask_dim > 0:
            self.mask_embed = nn.Sequential(
                nn.Linear(submask_dim, 128),
                nn.SiLU(),
                nn.Linear(128, model_channels),
            )

        # Output head
        if bottleneck_dim > 0:
            self.predict = nn.Sequential(
                nn.Linear(model_channels, model_channels),
                nn.GELU(),
                nn.Linear(model_channels, token_dim),
            )
        else:
            self.predict = nn.Sequential(
                nn.Linear(model_channels, model_channels),
                nn.GELU(),
                nn.Linear(model_channels, token_dim),
            )

        self.final_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        # Optional learned upsample head (e.g. 64D → 512D)
        self.upsample_to = upsample_to
        if upsample_to > 0 and upsample_to > resolution:
            out_dim = upsample_to ** 3
            self.upsample_head = nn.Sequential(
                nn.Linear(token_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
            # Zero-init last layer so initial output ≈ repeat
            nn.init.zeros_(self.upsample_head[-1].weight)
            nn.init.zeros_(self.upsample_head[-1].bias)
            self.output_dim = out_dim
        else:
            self.upsample_head = None
            self.output_dim = token_dim

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 9 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        if num_context_registers > 0:
            self.context_registers = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )
            self.context_register_pos = nn.Parameter(
                torch.randn(1, num_context_registers, cond_channels) * 0.02
            )

        blocks = []
        coarse_stride = pool_stride_coarse if pool_stride_coarse > 0 else pool_stride * 2
        for i in range(num_blocks):
            if i < num_pooled_layers:
                # Pooled Block: 全部计算在池化空间
                half_pooled = num_pooled_layers // 2
                stride_i = coarse_stride if i < half_pooled else pool_stride
                blocks.append(PooledSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                    cross_attn_topk=cross_attn_topk,
                    use_moe=(num_blocks - i <= num_moe_layers),
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    pool_stride=stride_i,
                ))
            else:
                # Standard Block: windowed self-attn + cross-attn + MLP
                blocks.append(ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_window=(
                        (0, 0, 0) if i % 2 == 0
                        else (window_size // 2, window_size // 2, window_size // 2)
                    ) if attn_mode == "windowed" else None,
                    full_attn_threshold=full_attn_threshold,
                    use_checkpoint=use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                    cross_attn_topk=cross_attn_topk,
                    use_moe=(num_blocks - i <= num_moe_layers),
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                ))
        self.blocks = nn.ModuleList(blocks)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[SparseFlowModel] res={resolution}, token_dim={token_dim}, "
              f"bottleneck={bottleneck_dim}, submask_res={submask_resolution}, "
              f"hidden={model_channels}x{num_blocks}, "
              f"attn={attn_mode}, w={window_size}, heads={self.num_heads}, "
              f"fp16={use_fp16}, params={n_params:.1f}M")
        if self.upsample_head is not None:
            n_up = sum(p.numel() for p in self.upsample_head.parameters())
            print(f"  upsample: {resolution}³→{upsample_to}³ (output_dim={self.output_dim}), head_params={n_up}")
        if num_pooled_layers > 0:
            print(f"  layers: {num_pooled_layers} pooled (stride coarse:{coarse_stride}/fine:{pool_stride}) "
                  f"+ {num_blocks - num_pooled_layers} standard")
        if num_context_registers > 0:
            print(f"  registers={num_context_registers}, start_block={context_start_block}")
        if num_moe_layers > 0:
            n_moe = sum(
                p.numel() for b in self.blocks[-num_moe_layers:] for p in b.mlp.parameters()
            ) / 1e6
            print(f"  MoE: last {num_moe_layers}, {num_experts} experts top-{moe_top_k}, {n_moe:.1f}M")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

        # mask_embed zero-init (if present)
        if hasattr(self, 'mask_embed'):
            nn.init.zeros_(self.mask_embed[-1].weight)
            nn.init.zeros_(self.mask_embed[-1].bias)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        submask: torch.Tensor = None,
        return_decoder_feats: bool = False,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, sp.SparseTensor]]:
        h_feats = self.embed(x.feats).type(self.dtype)

        # Optional sub-mask conditioning
        if submask is not None and hasattr(self, 'mask_embed'):
            mask_emb = self.mask_embed(submask).type(self.dtype)
            h_feats = h_feats + mask_emb

        h = sp.SparseTensor(
            feats=h_feats,
            coords=x.coords,
            shape=torch.Size([len(x.layout), h_feats.shape[-1]]),
            layout=x.layout,
        )

        t_emb = self.t_embedder(t * 1000).type(self.dtype)
        t_emb_tx = self.adaLN_modulation(t_emb) if self.share_mod else t_emb
        t_emb_tx = t_emb_tx.type(self.dtype)
        if cond is not None:
            cond = cond.type(self.dtype)

        if self.num_context_registers > 0:
            B = t.shape[0]
            registers = (self.context_registers + self.context_register_pos
                         ).expand(B, -1, -1).contiguous().type(self.dtype)
            cond_with_reg = torch.cat([registers, cond], dim=1)
        else:
            cond_with_reg = cond

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)

        for i, block in enumerate(self.blocks):
            ctx = cond_with_reg if (self.num_context_registers > 0 and i >= self.context_start_block) else cond
            h = block(h, t_emb_tx, ctx)

        h_out = self.final_norm(h.feats)
        udf = self.predict(h_out.type(x.dtype))

        # Learned upsample if configured
        if self.upsample_head is not None:
            udf = self.upsample_head(udf)  # [T, 64] → [T, 512]

        out = x.replace(udf)

        if return_decoder_feats:
            return out, h
        return out
