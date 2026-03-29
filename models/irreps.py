# models/irreps.py
"""
SO(2) irreducible representations and gauge-equivariant kernel basis.

This module implements the kernel constraint solution from Table 1 of:
    de Haan et al., "Gauge Equivariant Mesh CNNs", 2020.

The key insight: any gauge-equivariant kernel K_neigh(θ) mapping from
irrep ρ_n to irrep ρ_m must satisfy:

    K_neigh(θ - g) = ρ_m(-g) · K_neigh(θ) · ρ_n(g)   ∀ g,θ ∈ [0, 2π)

The solution space is spanned by a small set of "basis kernels" — fixed
angular functions of θ multiplied by learned scalar weights.

For F1 application:
    - Scalar (ρ₀) features: pressure coefficient Cp, speed magnitude
    - Vector (ρ₁) features: wall shear stress direction, surface velocity
    - Tensor (ρ₂) features: strain-rate at wall (for turbulence sensing)
    Higher-order irreps capture increasingly fine angular detail.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Type definitions
# ────────────────────────────────────────────────────────────────────────────

# A "feature type" is a list of (irrep_order, multiplicity) pairs.
# E.g. [(0, 16), (1, 16), (2, 8)] = 16ρ₀ ⊕ 16ρ₁ ⊕ 8ρ₂
FeatureType = List[Tuple[int, int]]


def feature_dim(ftype: FeatureType) -> int:
    """Total dimension of a feature type."""
    return sum(
        mult * (1 if order == 0 else 2)
        for order, mult in ftype
    )


def scalar_type(n_channels: int) -> FeatureType:
    """All-scalar feature type: n_channels × ρ₀."""
    return [(0, n_channels)]


# ────────────────────────────────────────────────────────────────────────────
# SO(2) representation matrices
# ────────────────────────────────────────────────────────────────────────────

def rho(order: int, angle: Tensor) -> Tensor:
    """
    SO(2) irrep ρ_n evaluated at angle g.

    ρ₀(g) = [[1]]                              (scalar, invariant)
    ρ_n(g) = [[cos(ng), -sin(ng)],             (n > 0)
               [sin(ng),  cos(ng)]]

    Args:
        order: irrep order n ≥ 0
        angle: scalar tensor [radians]
    Returns:
        (d, d) rotation matrix where d = 1 if order=0, else 2
    """
    if order == 0:
        return torch.ones(1, 1, dtype=angle.dtype, device=angle.device)
    g = order * angle
    c, s = torch.cos(g), torch.sin(g)
    return torch.stack([
        torch.stack([ c, -s]),
        torch.stack([ s,  c]),
    ])   # (2, 2)


def rho_batch(order: int, angles: Tensor) -> Tensor:
    """
    Batched SO(2) irrep evaluation.

    Args:
        order  : irrep order n ≥ 0
        angles : (E,) edge angles
    Returns:
        (E, d, d) where d = 1 if order=0, else 2
    """
    E = angles.shape[0]
    if order == 0:
        return torch.ones(E, 1, 1, dtype=angles.dtype, device=angles.device)
    g = order * angles         # (E,)
    c = torch.cos(g)           # (E,)
    s = torch.sin(g)           # (E,)
    # Stack into (E, 2, 2)
    row0 = torch.stack([ c, -s], dim=-1)   # (E, 2)
    row1 = torch.stack([ s,  c], dim=-1)   # (E, 2)
    return torch.stack([row0, row1], dim=1)  # (E, 2, 2)


# ────────────────────────────────────────────────────────────────────────────
# Basis kernel evaluation  (Table 1 of GEM paper)
# ────────────────────────────────────────────────────────────────────────────

def basis_kernels_neigh(
    n_in:   int,     # input irrep order
    n_out:  int,     # output irrep order
    angles: Tensor,  # (E,) edge angles θ_pq
) -> Tensor:
    """
    Evaluate all basis kernels for K_neigh mapping ρ_{n_in} → ρ_{n_out}
    at the given angles.

    Returns tensor of shape (E, d_out, d_in, num_basis) where:
        d_in  = 1 if n_in=0 else 2
        d_out = 1 if n_out=0 else 2
        num_basis = number of linearly independent equivariant kernels

    From Table 1:
        ρ₀ → ρ₀  :  1 basis   K(θ) = [[1]]
        ρ_n → ρ₀ :  2 bases   K(θ) = [cos(nθ), sin(nθ)]
        ρ₀ → ρ_m :  2 bases   K(θ) = [[cos(mθ)], [sin(mθ)]] and [[sin(mθ)], [-cos(mθ)]]
        ρ_n → ρ_m:  4 bases   using cos/sin of (m±n)θ
    """
    E = angles.shape[0]
    device = angles.device

    if n_in == 0 and n_out == 0:
        # Single basis: K(θ) = 1  (isotropic — the GCN special case)
        return torch.ones(E, 1, 1, 1, device=device)

    elif n_in > 0 and n_out == 0:
        # 2 bases: [cos(nθ), sin(nθ)]
        c = torch.cos(n_in * angles)   # (E,)
        s = torch.sin(n_in * angles)
        b0 = torch.stack([ c,  s], dim=-1).reshape(E, 1, 2, 1)[:,:,:,0:1]
        b1 = torch.stack([ s, -c], dim=-1).reshape(E, 1, 2, 1)[:,:,:,0:1]
        # Shape: (E, 1, 2, 2) — 2 bases
        k0 = torch.stack([ c,  s], dim=-1).reshape(E, 1, 2)   # (E, 1, 2)
        k1 = torch.stack([ s, -c], dim=-1).reshape(E, 1, 2)
        return torch.stack([k0, k1], dim=-1)   # (E, 1, 2, 2)

    elif n_in == 0 and n_out > 0:
        # 2 bases
        c = torch.cos(n_out * angles)   # (E,)
        s = torch.sin(n_out * angles)
        k0 = torch.stack([ c,  s], dim=-1).reshape(E, 2, 1)   # (E, 2, 1)
        k1 = torch.stack([ s, -c], dim=-1).reshape(E, 2, 1)
        return torch.stack([k0, k1], dim=-1)   # (E, 2, 1, 2)

    else:  # n_in > 0 and n_out > 0 — 4 bases
        # c± = cos((m ± n)θ),  s± = sin((m ± n)θ)
        p = n_out + n_in
        q = abs(n_out - n_in)
        cp = torch.cos(p * angles); sp = torch.sin(p * angles)
        cq = torch.cos(q * angles); sq = torch.sin(q * angles)

        # Sign factor for |m-n|
        sign = 1.0 if n_out >= n_in else -1.0

        # 4 basis kernels (each (E, 2, 2))
        def mat(a, b, c, d):
            row0 = torch.stack([a, b], dim=-1)   # (E, 2)
            row1 = torch.stack([c, d], dim=-1)
            return torch.stack([row0, row1], dim=1)  # (E, 2, 2)

        k0 = mat( cq, -sq*sign,  sq*sign,  cq)
        k1 = mat( sq,  cq*sign, -cq*sign,  sq)
        k2 = mat( cp,  sp,       sp,      -cp)
        k3 = mat(-sp,  cp,       cp,       sp)

        return torch.stack([k0, k1, k2, k3], dim=-1)   # (E, 2, 2, 4)


def basis_kernels_self(
    n_in:  int,
    n_out: int,
    dim:   int,
) -> Tensor:
    """
    Basis kernels for K_self (angle-independent, Eq. 4).

    K_self must satisfy:  K_self = ρ_out(-g) K_self ρ_in(g)  ∀ g
    This is only non-zero when n_in == n_out (Table 1, bottom).

    Returns (d_out, d_in, num_basis) constant tensor.
        For n_in == n_out == 0: identity [[1]], 1 basis
        For n_in == n_out > 0 : [[1,0],[0,1]] and [[0,1],[-1,0]], 2 bases
    """
    if n_in != n_out:
        return None   # no equivariant self-interaction between different irreps

    if n_in == 0:
        return torch.ones(1, 1, 1)   # (1, 1, 1)

    # n_in == n_out > 0: 2 bases
    I = torch.eye(2).unsqueeze(-1)          # (2, 2, 1)
    J = torch.tensor([[0., 1.], [-1., 0.]]).unsqueeze(-1)  # (2, 2, 1)
    return torch.cat([I, J], dim=-1)         # (2, 2, 2)


# ────────────────────────────────────────────────────────────────────────────
# Full equivariant kernel from feature type to feature type
# ────────────────────────────────────────────────────────────────────────────

def count_parameters_neigh(ftype_in: FeatureType, ftype_out: FeatureType) -> int:
    """Count total learnable parameters for K_neigh."""
    total = 0
    for (n_out, m_out) in ftype_out:
        for (n_in, m_in) in ftype_in:
            n_basis = _n_basis_neigh(n_in, n_out)
            total += n_basis * m_in * m_out
    return total


def count_parameters_self(ftype_in: FeatureType, ftype_out: FeatureType) -> int:
    """Count total learnable parameters for K_self."""
    total = 0
    for (n_out, m_out) in ftype_out:
        for (n_in, m_in) in ftype_in:
            if n_in == n_out:
                n_basis = 1 if n_in == 0 else 2
                total += n_basis * m_in * m_out
    return total


def _n_basis_neigh(n_in: int, n_out: int) -> int:
    if n_in == 0 and n_out == 0: return 1
    if n_in == 0 or n_out == 0:  return 2
    return 4


class EquivariantKernelBasis(nn.Module):
    """
    Parameterised gauge-equivariant kernel K_neigh(θ) + K_self.

    Given learned weights w_neigh and w_self, evaluates:
        K_neigh(θ) = Σ_i  w_neigh_i * BasisKernel_i(θ)
        K_self     = Σ_i  w_self_i  * BasisKernel_i

    for all (n_in, n_out) irrep block pairs.

    Args:
        ftype_in   : input feature type
        ftype_out  : output feature type
    """

    def __init__(self, ftype_in: FeatureType, ftype_out: FeatureType):
        super().__init__()
        self.ftype_in  = ftype_in
        self.ftype_out = ftype_out
        self.dim_in    = feature_dim(ftype_in)
        self.dim_out   = feature_dim(ftype_out)

        # Count parameters
        n_w_neigh = count_parameters_neigh(ftype_in, ftype_out)
        n_w_self  = count_parameters_self(ftype_in, ftype_out)

        # Learned weights (initialised with Kaiming-like scale)
        self.w_neigh = nn.Parameter(
            torch.randn(n_w_neigh) * (2.0 / (self.dim_in + self.dim_out)) ** 0.5
        )
        self.w_self = nn.Parameter(
            torch.randn(n_w_self) * (2.0 / (self.dim_in + self.dim_out)) ** 0.5
        )

        # Precompute layout: which weight indices correspond to which block
        self._neigh_layout = self._build_neigh_layout()
        self._self_layout  = self._build_self_layout()

    def _build_neigh_layout(self):
        """Returns list of (n_in, mult_in, n_out, mult_out, n_basis, w_start)."""
        layout = []
        w = 0
        for (n_out, m_out) in self.ftype_out:
            for (n_in, m_in) in self.ftype_in:
                nb = _n_basis_neigh(n_in, n_out)
                layout.append((n_in, m_in, n_out, m_out, nb, w))
                w += nb * m_in * m_out
        return layout

    def _build_self_layout(self):
        """Returns list of (n_in, mult_in, n_out, mult_out, n_basis, w_start)."""
        layout = []
        w = 0
        for (n_out, m_out) in self.ftype_out:
            for (n_in, m_in) in self.ftype_in:
                if n_in == n_out:
                    nb = 1 if n_in == 0 else 2
                    layout.append((n_in, m_in, n_out, m_out, nb, w))
                    w += nb * m_in * m_out
        return layout

    def eval_neigh(self, angles: Tensor) -> Tensor:
        """
        Evaluate K_neigh at given angles.

        Args:
            angles: (E,) edge angles θ_pq
        Returns:
            K: (E, dim_out, dim_in) kernel matrices for each edge
        """
        E = angles.shape[0]
        K = torch.zeros(E, self.dim_out, self.dim_in,
                        device=angles.device, dtype=angles.dtype)

        # Track offset in output/input dimensions per irrep block
        out_offset = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            in_offset = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                nb   = _n_basis_neigh(n_in, n_out)

                # Find weight slice for this block
                entry = next(e for e in self._neigh_layout
                             if e[0]==n_in and e[2]==n_out)
                w_start, n_b = entry[5], entry[4]
                w_block = self.w_neigh[w_start : w_start + nb * m_in * m_out]
                w_block = w_block.reshape(m_out, m_in, nb)  # (m_out, m_in, nb)

                # Basis kernels: (E, d_out, d_in, nb)
                basis = basis_kernels_neigh(n_in, n_out, angles)

                # Contract: K_block = Σ_b w[i,j,b] * basis[e, :, :, b]
                # → (E, m_out*d_out, m_in*d_in)
                # Use einsum for clarity
                k_block = torch.einsum('ijb,exyb->eixjy',
                                       w_block, basis)   # (E, m_out, d_out, m_in, d_in)
                k_block = k_block.reshape(E, m_out*d_out, m_in*d_in)

                K[:, out_offset:out_offset+m_out*d_out,
                     in_offset:in_offset+m_in*d_in] = k_block
                in_offset += m_in * d_in
            out_offset += m_out * d_out

        return K   # (E, dim_out, dim_in)

    def eval_self(self) -> Tensor:
        """
        Evaluate K_self (angle-independent).
        Returns (dim_out, dim_in) matrix.
        """
        K = torch.zeros(self.dim_out, self.dim_in,
                        device=self.w_self.device, dtype=self.w_self.dtype)

        out_offset = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            in_offset = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                if n_in != n_out:
                    in_offset += m_in * d_in
                    continue

                nb = 1 if n_in == 0 else 2
                entry = next(e for e in self._self_layout
                             if e[0]==n_in and e[2]==n_out)
                w_start = entry[5]
                w_block = self.w_self[w_start : w_start + nb * m_in * m_out]
                w_block = w_block.reshape(m_out, m_in, nb)

                basis = basis_kernels_self(n_in, n_out, d_in)  # (d_out, d_in, nb)
                if basis is None:
                    in_offset += m_in * d_in
                    continue

                basis = basis.to(self.w_self.device)
                k_block = torch.einsum('ijb,xyb->ixjy',
                                       w_block, basis)
                k_block = k_block.reshape(m_out*d_out, m_in*d_in)

                K[out_offset:out_offset+m_out*d_out,
                  in_offset:in_offset+m_in*d_in] = k_block
                in_offset += m_in * d_in
            out_offset += m_out * d_out

        return K   # (dim_out, dim_in)
