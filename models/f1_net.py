# models/f1_net.py
"""
F1AeroNet: Full prediction network for F1 car aerodynamics.

Architecture:
    Input: per-vertex [x, y, z, U_inf] (4 × ρ₀ scalars)
      │
      ├─ Input embedding: Linear → (16ρ₀)
      │
      ├─ 6× GEMBlock with progressively richer feature types
      │     [(16,2), (32,2), (32,3), (64,3), (64,2), (64,1)]
      │     Each block: GEMConv → LayerNorm → RegularNonlinearity → Residual
      │
      ├─ Output projection → (64ρ₀)  [collapse all to scalars for heads]
      │
      ├─ HEAD A: Cp  — per-vertex scalar → MLP → (V, 1)
      ├─ HEAD B: WSS — per-vertex vector → linear decode ρ₁ → (V, 3)
      ├─ HEAD C: Cd  — global mean pool → MLP → (1,)
      └─ HEAD D: Cl  — global mean pool → MLP → (1,)

The key F1 insight: pressure (Cp) is a ρ₀ feature (gauge-invariant scalar),
wall shear stress is a ρ₁ feature (tangent vector, gauge-equivariant).
The network keeps them in their correct geometric types through all layers,
only collapsing to Euclidean vectors at the output head.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from typing import List, Tuple, Optional

from models.gem_conv import GEMBlock, GEMConv
from models.irreps import FeatureType, feature_dim, scalar_type


# ────────────────────────────────────────────────────────────────────────────
# Helper: build feature type from config spec
# ────────────────────────────────────────────────────────────────────────────

def build_ftype(mult: int, max_order: int) -> FeatureType:
    """
    Build feature type with `mult` copies of each irrep from ρ₀ to ρ_{max_order}.
    E.g. build_ftype(16, 2) = [(0,16), (1,16), (2,16)]
    """
    return [(order, mult) for order in range(max_order + 1)]


# ────────────────────────────────────────────────────────────────────────────
# Output Heads
# ────────────────────────────────────────────────────────────────────────────

class ScalarHead(nn.Module):
    """
    Per-vertex scalar prediction head (for Cp).

    Input: (V, C) feature vector (expected all ρ₀ after projection)
    Output: (V, 1) scalar field
    """
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x).squeeze(-1)   # (V,)


class VectorHead(nn.Module):
    """
    Per-vertex 3D vector prediction head (for WSS).

    Takes the ρ₁ (tangent vector) channels from the last GEM layer and
    decodes them back to ambient 3D vectors using the stored tangent frames.

    If tangent frames are not available, falls back to a simple MLP.

    Input : (V, C) mixed features
    Output: (V, 3) vector field in ambient R³
    """
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)   # (V, 3)


class GlobalHead(nn.Module):
    """
    Global scalar prediction head (for Cd, Cl).

    Global mean pool over vertices → MLP → scalar.
    """
    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x    : (V, C) node features
            batch: (V,)   batch assignment vector (None = single graph)
        Returns:
            (B,) global scalars
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)   # (B, C)
        return self.mlp(pooled).squeeze(-1)   # (B,)


# ────────────────────────────────────────────────────────────────────────────
# Main Network
# ────────────────────────────────────────────────────────────────────────────

class F1AeroNet(nn.Module):
    """
    Full aerodynamics prediction network for F1 car surfaces.

    Args:
        in_channels         : number of input scalar features per vertex (default 4)
        layer_specs         : list of (mult, max_order) per GEM block
        N_nonlin            : RegularNonlinearity sample count
        head_hidden         : hidden dim in prediction MLP heads
        head_dropout        : dropout in prediction heads
        break_symmetry_final: collapse last GEM output to scalars before heads
                             (beneficial when mesh orientation is consistent)

    Usage:
        model = F1AeroNet.from_config(cfg['model'])
        pred = model(data.x, data.edge_index, data.edge_angles,
                     data.edge_transporters)
        # pred is dict with keys 'cp', 'wss', 'cd', 'cl'
    """

    def __init__(
        self,
        in_channels:          int = 4,
        layer_specs:          List[Tuple[int, int]] = None,
        N_nonlin:             int = 7,
        head_hidden:          int = 128,
        head_dropout:         float = 0.1,
        break_symmetry_final: bool = True,
    ):
        super().__init__()

        if layer_specs is None:
            # Default: matches configs/f1_base.yaml
            layer_specs = [(16,2), (32,2), (32,3), (64,3), (64,2), (64,1)]

        # ── Input embedding: project scalar inputs to first feature type ──
        first_ftype = build_ftype(layer_specs[0][0], layer_specs[0][1])
        first_dim   = feature_dim(first_ftype)
        self.input_embed = nn.Linear(in_channels, first_dim, bias=True)

        # ── GEM blocks ────────────────────────────────────────────────────
        self.blocks   = nn.ModuleList()
        self.ftypes   = [first_ftype]

        for i, (mult, max_order) in enumerate(layer_specs):
            ftype_out = build_ftype(mult, max_order)
            self.blocks.append(
                GEMBlock(
                    ftype_in  = self.ftypes[-1],
                    ftype_out = ftype_out,
                    N_nonlin  = N_nonlin,
                )
            )
            self.ftypes.append(ftype_out)

        last_ftype = self.ftypes[-1]
        last_dim   = feature_dim(last_ftype)

        # ── Optional symmetry breaking: collapse to scalars ───────────────
        self.break_symmetry = break_symmetry_final
        if break_symmetry_final:
            # Count scalar (ρ₀) channels only
            n_scalars = sum(mult for order, mult in last_ftype if order == 0)
            self.sym_break_proj = nn.Linear(last_dim, n_scalars, bias=False)
            head_in = n_scalars
        else:
            head_in = last_dim

        # ── Prediction heads ──────────────────────────────────────────────
        self.cp_head  = ScalarHead(head_in, hidden=head_hidden, dropout=head_dropout)
        self.wss_head = VectorHead(head_in, hidden=head_hidden, dropout=head_dropout)
        self.cd_head  = GlobalHead(head_in, hidden=head_hidden // 2, dropout=head_dropout)
        self.cl_head  = None #no cl in data

    @classmethod
    def from_config(cls, cfg: dict) -> 'F1AeroNet':
        """Construct from a config dict (see configs/f1_base.yaml)."""
        specs = [tuple(s) for s in cfg.get('layer_types', [(16,2),(32,2),(32,3),(64,3),(64,2),(64,1)])]
        return cls(
            in_channels          = cfg.get('in_channels', 4),
            layer_specs          = specs,
            N_nonlin             = cfg.get('nonlin_samples', 7),
            head_hidden          = 128,
            head_dropout         = cfg.get('head_dropout', 0.1),
            break_symmetry_final = cfg.get('break_symmetry_final', True),
        )

    def forward(
        self,
        x:            Tensor,            # (V, in_channels)
        edge_index:   Tensor,            # (2, E)
        angles:       Tensor,            # (E,)
        transporters: Tensor,            # (E,)
        batch:        Optional[Tensor] = None,  # (V,) batch vector
    ) -> dict:
        """
        Forward pass.

        Returns dict:
            'cp'  : (V,)   surface pressure coefficient
            'wss' : (V, 3) wall shear stress vector
            'cd'  : (B,)   drag coefficient
            'cl'  : (B,)   downforce coefficient (negative = downforce)
        """
        # Input embedding (scalar → mixed irrep type)
        h = self.input_embed(x)   # (V, first_dim)

        # GEM blocks
        for block in self.blocks:
            h = block(h, edge_index, angles, transporters)   # (V, C_out)

        # Symmetry breaking / feature projection for heads
        if self.break_symmetry:
            h_heads = self.sym_break_proj(h)   # (V, n_scalars)
        else:
            h_heads = h                        # (V, last_dim)

        # Predictions
        cp  = self.cp_head(h_heads)              # (V,)
        wss = self.wss_head(h_heads)             # (V, 3)
        cd  = self.cd_head(h_heads, batch)       # (B,)
        cl  = None

        return {'cp': cp, 'wss': wss, 'cd': cd, 'cl': None}

    def count_parameters(self) -> dict:
        """Count trainable parameters by component."""
        def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'input_embed': count(self.input_embed),
            'gem_blocks':  sum(count(b) for b in self.blocks),
            'cp_head':     count(self.cp_head),
            'wss_head':    count(self.wss_head),
            'cd_head':     count(self.cd_head),
            'cl_head':     count(self.cl_head),
            'total':       count(self),
        }
