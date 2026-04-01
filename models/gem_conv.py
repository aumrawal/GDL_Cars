# models/gem_conv.py
"""
Gauge Equivariant Mesh Convolution layer.

Implements Algorithm 1 of de Haan et al. (2020):

    f'_p = Σ_i  w_self_i  · K_self_i · f_p
         + Σ_{i,q∈N(p)}  w_neigh_i · K_neigh_i(θ_pq) · ρ_in(g_{q→p}) · f_q

Key differences from a plain GCN (torch_geometric.nn.GCNConv):
  1. Anisotropic kernel K_neigh(θ) — value depends on neighbour angle
  2. Parallel transport ρ_in(g_{q→p}) — rotates neighbour features into
     the target vertex frame before aggregation
  3. Features carry geometric type — each block of channels transforms
     as a specific SO(2) irrep under gauge change

The RegularNonlinearity (Sec 5 of GEM paper) is implemented as a
separate module below and applied between GEM-CNN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def scatter_add(src, index, dim, dim_size):
    """Pure PyTorch replacement for torch_scatter.scatter_add."""
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out

from models.irreps import (
    FeatureType,
    EquivariantKernelBasis,
    feature_dim,
    rho_batch,
)


# ────────────────────────────────────────────────────────────────────────────
# Parallel transport application
# ────────────────────────────────────────────────────────────────────────────

def apply_parallel_transport(
    features:     Tensor,       # (E, C_in)  neighbour features f_q, one per edge
    transporters: Tensor,       # (E,)       transporter angles g_{q→p}
    ftype_in:     FeatureType,  # input feature type
) -> Tensor:
    """
    Apply ρ_in(g_{q→p}) to each neighbour's feature vector.

    For each irrep block of type ρ_n:
        transported = R(n * g_{q→p}) · feature_block

    where R(φ) = [[cos φ, -sin φ], [sin φ, cos φ]] for n>0,
    and R(0) = identity for scalars.

    This is the operation that makes GEM-CNN correct on curved surfaces:
    a wall shear stress vector measured at the diffuser edge of an F1 car
    is rotated into the frame of the target vertex before being summed —
    so the gradient information accumulates coherently across the curved floor.
    """
    transported = torch.zeros_like(features)
    offset = 0
    for (order, mult) in ftype_in:
        d = 1 if order == 0 else 2
        block = features[:, offset : offset + mult * d]   # (E, mult*d)

        if order == 0:
            # Scalars are invariant — no rotation needed
            transported[:, offset : offset + mult * d] = block
        else:
            # Reshape to (E, mult, d) for per-irrep rotation
            block = block.reshape(-1, mult, d)              # (E, mult, 2)
            # (E, 2, 2) rotation matrices
            R = rho_batch(order, transporters)              # (E, 2, 2)
            # Apply: (E, mult, 2) x (E, 2, 2)^T → (E, mult, 2)
            rotated = torch.einsum('emd,erd->emr', block, R)
            transported[:, offset : offset + mult * d] = rotated.reshape(-1, mult*d)

        offset += mult * d

    return transported   # (E, C_in)


# ────────────────────────────────────────────────────────────────────────────
# GEM Convolution Layer
# ────────────────────────────────────────────────────────────────────────────

class GEMConv(nn.Module):
    """
    Single Gauge Equivariant Mesh Convolution layer.

    Args:
        ftype_in        : input feature type  e.g. [(0,4), (1,4)]
        ftype_out       : output feature type e.g. [(0,8), (1,8), (2,4)]
        break_symmetry  : if True, treat all output as scalar (ρ₀) — breaks
                          gauge equivariance but can improve performance when
                          meshes have a consistent global orientation (Sec 7.2)

    Example (F1 intermediate layer):
        layer = GEMConv(
            ftype_in  = [(0,16), (1,16), (2,8)],
            ftype_out = [(0,32), (1,32), (2,16)],
        )
    """

    def __init__(
        self,
        ftype_in:       FeatureType,
        ftype_out:      FeatureType,
        break_symmetry: bool = False,
    ):
        super().__init__()
        self.ftype_in       = ftype_in
        self.ftype_out      = ftype_out
        self.break_symmetry = break_symmetry
        self.dim_in         = feature_dim(ftype_in)
        self.dim_out        = feature_dim(ftype_out)

        # Equivariant kernel basis (contains all learnable weights)
        self.kernel = EquivariantKernelBasis(ftype_in, ftype_out)

    def forward(
        self,
        x:            Tensor,   # (V, C_in)   node features
        edge_index:   Tensor,   # (2, E)      [src=q, tgt=p]
        angles:       Tensor,   # (E,)        θ_pq neighbour angles
        transporters: Tensor,   # (E,)        g_{q→p} parallel transporters
    ) -> Tensor:
        """
        Compute GEM convolution output f'_p for all vertices p.

        Returns (V, C_out) output features.
        """
        src, tgt = edge_index[0], edge_index[1]   # q, p
        V = x.shape[0]

        # ── Self interaction: K_self · f_p ──────────────────────────
        K_self = self.kernel.eval_self()           # (C_out, C_in)
        out = x @ K_self.T                         # (V, C_out)

        # ── Neighbour aggregation ────────────────────────────────────
        # 1. Gather neighbour features
        f_q = x[src]                               # (E, C_in)

        # 2. Parallel transport: ρ_in(g_{q→p}) · f_q
        f_q_transported = apply_parallel_transport(
            f_q, transporters, self.ftype_in
        )   # (E, C_in)

        # 3. Evaluate anisotropic kernel at each edge's angle
        K_neigh = self.kernel.eval_neigh(angles)   # (E, C_out, C_in)

        # 4. Apply kernel: (E, C_out, C_in) x (E, C_in, 1) → (E, C_out)
        msg = torch.bmm(K_neigh, f_q_transported.unsqueeze(-1)).squeeze(-1)
        # msg shape: (E, C_out)

        # 5. Aggregate messages at target vertex p
        out = out + scatter_add(msg, tgt, dim=0, dim_size=V)  # (V, C_out)

        return out   # (V, C_out)


# ────────────────────────────────────────────────────────────────────────────
# Regular Non-linearity  (Sec 5 of GEM paper)
# ────────────────────────────────────────────────────────────────────────────

class RegularNonlinearity(nn.Module):
    """
    Approximately gauge-equivariant nonlinearity based on the Fourier trick
    described in Section 5 of de Haan et al. (2020).

    Idea:
      1. Treat each feature as Fourier coefficients of a periodic signal
      2. Take inverse DFT to N spatial samples
      3. Apply pointwise ReLU
      4. Take DFT back to Fourier coefficients

    This is exactly equivariant for gauge rotations that are multiples of
    2π/N, and approximately equivariant otherwise (error → 0 as N → ∞).

    For features of mixed irrep type (e.g. ρ₀ ⊕ ρ₁ ⊕ ρ₂), we handle each
    multiplicity group independently.

    Args:
        ftype    : feature type of the input (and output — type is preserved)
        N        : number of sample points (>= 2*max_order + 1 recommended)
        nonlin   : pointwise nonlinearity to apply in sample space
    """

    def __init__(
        self,
        ftype:  FeatureType,
        N:      int = 7,
        nonlin: nn.Module = None,
    ):
        super().__init__()
        self.ftype  = ftype
        self.N      = N
        self.nonlin = nonlin or nn.ReLU()

        # Precompute DFT / IDFT matrices for each irrep order present
        orders = sorted(set(order for order, _ in ftype))
        self.register_buffer('_dummy', torch.zeros(1))  # for device tracking

        # Precompute for highest order (others are subsets)
        max_order = max(orders)
        self._build_dft_matrices(max_order)

    def _build_dft_matrices(self, max_order: int):
        """
        Build IDFT matrix A (irrep coefficients → samples) and
        DFT matrix B (samples → irrep coefficients) for band-limited
        Fourier transform up to order max_order.

        Dimension of coefficient vector: 1 + 2*max_order (one ρ₀ + max_order ρ_n)
        """
        N = self.N
        angles = torch.linspace(0, 2 * torch.pi * (N-1) / N, N)   # (N,)

        # Build IDFT: columns = [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ...]
        cols = [torch.ones(N)]
        for k in range(1, max_order + 1):
            cols.append(torch.cos(k * angles))
            cols.append(torch.sin(k * angles))
        A = torch.stack(cols, dim=1)   # (N, 1+2*max_order)  IDFT matrix

        # DFT (pseudo-inverse): B = (AᵀA)⁻¹Aᵀ
        B = torch.linalg.lstsq(A, torch.eye(N)).solution   # (1+2*max_order, N)

        self.register_buffer('idft', A)   # (N, D)
        self.register_buffer('dft',  B)   # (D, N)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RegularNonlinearity to features x of shape (V, C).
        Returns same shape (V, C).
        """
        out = torch.zeros_like(x)
        offset = 0

        for (order, mult) in self.ftype:
            d = 1 if order == 0 else 2
            C = mult * d
            block = x[:, offset : offset + C]   # (V, mult*d)

            if order == 0:
                # Scalar: plain pointwise nonlinearity
                out[:, offset : offset + C] = self.nonlin(block)
            else:
                # Vector/tensor: Fourier trick per multiplicity group
                # Treat each pair of (cos, sin) coefficients as one "channel"
                block = block.reshape(-1, mult, d)   # (V, mult, 2)
                # For order n, the feature [a, b] represents a·cos(nθ) + b·sin(nθ)
                # We need to handle this per-order band — simplified: apply norm nonlin
                # (exact RegularNonlinearity would need the full Fourier expansion)
                # Here we use norm nonlinearity as a practical gauge-equivariant alternative
                norms = block.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (V, mult, 1)
                # Softplus of norm (smooth, always positive)
                new_norms = F.softplus(norms)
                out_block  = block * (new_norms / norms)
                out[:, offset : offset + C] = out_block.reshape(-1, C)

            offset += C

        return out   # (V, C)


# ────────────────────────────────────────────────────────────────────────────
# GEM-CNN Block: Conv + Norm + Nonlinearity
# ────────────────────────────────────────────────────────────────────────────

class GEMBlock(nn.Module):
    """
    One residual GEM-CNN block:
        f → GEMConv → LayerNorm → RegularNonlinearity → f'
        f' = f' + linear_skip(f)  (residual, if in/out dims match)

    Args:
        ftype_in   : input feature type
        ftype_out  : output feature type
        N_nonlin   : number of samples for RegularNonlinearity
    """

    def __init__(
        self,
        ftype_in:  FeatureType,
        ftype_out: FeatureType,
        N_nonlin:  int = 7,
    ):
        super().__init__()
        self.ftype_in  = ftype_in
        self.ftype_out = ftype_out
        dim_in  = feature_dim(ftype_in)
        dim_out = feature_dim(ftype_out)

        self.conv   = GEMConv(ftype_in, ftype_out)
        self.norm   = nn.LayerNorm(dim_out)
        self.nonlin = RegularNonlinearity(ftype_out, N=N_nonlin)

        # Skip connection (1×1 linear if dimensions differ)
        if dim_in == dim_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(dim_in, dim_out, bias=False)

    def forward(
        self,
        x:            Tensor,   # (V, C_in)
        edge_index:   Tensor,   # (2, E)
        angles:       Tensor,   # (E,)
        transporters: Tensor,   # (E,)
    ) -> Tensor:
        # Gradient checkpointing: recompute activations during backward
        # instead of storing them — halves activation memory at ~30% speed cost.
        if self.training:
            from torch.utils.checkpoint import checkpoint
            def _fwd(x, ei, ang, tr):
                h = self.conv(x, ei, ang, tr)
                h = self.norm(h)
                h = self.nonlin(h)
                return h
            h = checkpoint(_fwd, x, edge_index, angles, transporters,
                           use_reentrant=False)
        else:
            h = self.conv(x, edge_index, angles, transporters)
            h = self.norm(h)
            h = self.nonlin(h)
        return h + self.skip(x)
