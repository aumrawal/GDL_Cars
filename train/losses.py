# train/losses.py
"""
Loss functions for F1 aerodynamics prediction.

Four prediction targets, each with its own loss:

    Cp  (scalar field)     — MSE or L1 per vertex
    WSS (vector field)     — MSE or L1 on vector components per vertex
    Cd  (scalar global)    — L1 on drag coefficient
    Cl  (scalar global)    — L1 on lift/downforce coefficient

The total loss is a weighted sum:
    L = w_cp * L_cp + w_wss * L_wss + w_cd * L_cd + w_cl * L_cl

Cd and Cl get higher weights (default 10×) because they are single scalars
whose absolute error would otherwise be swamped by per-vertex field losses.

Gradient-weighted surface loss (optional):
    Aerodynamically critical regions (leading edges, wing tips) have large
    Cp gradients. We optionally weight the per-vertex Cp loss by the local
    Cp gradient magnitude so the model focuses on these zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Per-head loss functions
# ────────────────────────────────────────────────────────────────────────────

def scalar_field_loss(pred: Tensor, target: Tensor, mode: str = 'mse') -> Tensor:
    """
    Loss for per-vertex scalar field (Cp).

    Args:
        pred   : (V,) predicted values
        target : (V,) ground-truth values
        mode   : 'mse', 'l1', or 'huber'
    Returns:
        scalar loss
    """
    if mode == 'mse':
        return F.mse_loss(pred, target)
    elif mode == 'l1':
        return F.l1_loss(pred, target)
    elif mode == 'huber':
        return F.huber_loss(pred, target, delta=0.1)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def vector_field_loss(pred: Tensor, target: Tensor, mode: str = 'mse') -> Tensor:
    """
    Loss for per-vertex vector field (WSS).

    Computed on all 3 components jointly — equivalent to mean over component MSEs.

    Args:
        pred   : (V, 3) predicted vectors
        target : (V, 3) ground-truth vectors
        mode   : 'mse' or 'l1'
    Returns:
        scalar loss
    """
    if mode == 'mse':
        return F.mse_loss(pred, target)
    elif mode == 'l1':
        return F.l1_loss(pred, target)
    elif mode == 'huber':
        return F.huber_loss(pred, target, delta=0.01)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def global_coeff_loss(pred: Tensor, target: Tensor, mode: str = 'l1') -> Tensor:
    """
    Loss for global aerodynamic coefficient (Cd, Cl).

    L1 is preferred over MSE here: a drag error of 0.01 (counts!) should
    be penalised linearly, not quadratically.

    Args:
        pred   : (B,) or (B, 1)
        target : (B,) or (B, 1)
    Returns:
        scalar loss
    """
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    if mode == 'l1':
        return F.l1_loss(pred, target)
    elif mode == 'mse':
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


# ────────────────────────────────────────────────────────────────────────────
# Combined loss
# ────────────────────────────────────────────────────────────────────────────

class F1AeroLoss(nn.Module):
    """
    Combined weighted loss for all four F1 aerodynamics prediction targets.

    Args:
        weights   : dict with keys 'cp', 'wss', 'cd', 'cl'
        loss_types: dict with keys 'cp', 'wss', 'cd', 'cl' → 'mse'/'l1'/'huber'
    """

    def __init__(
        self,
        weights:    Optional[dict] = None,
        loss_types: Optional[dict] = None,
    ):
        super().__init__()
        self.weights = weights or {
            'cp': 1.0, 'wss': 1.0, 'cd': 10.0, 'cl': 10.0
        }
        self.loss_types = loss_types or {
            'cp': 'mse', 'wss': 'mse', 'cd': 'l1', 'cl': 'l1'
        }

    def forward(
        self,
        pred:   dict,    # model output dict
        batch:  object,  # PyG Data batch
    ) -> Tuple[Tensor, dict]:
        """
        Compute total loss and per-head loss breakdown.

        Returns:
            total : scalar total loss
            parts : dict of individual losses (for logging)
        """
        losses = {}

        # ── Cp loss ────────────────────────────────────────────────
        if pred['cp'] is not None and hasattr(batch, 'y_cp') and batch.y_cp is not None:
            losses['cp'] = scalar_field_loss(
                pred['cp'], batch.y_cp, mode=self.loss_types['cp']
            )

        # ── WSS loss ───────────────────────────────────────────────
        if pred['wss'] is not None and hasattr(batch, 'y_wss') and batch.y_wss is not None:
            losses['wss'] = vector_field_loss(
                pred['wss'], batch.y_wss, mode=self.loss_types['wss']
            )

        # ── Cd loss ────────────────────────────────────────────────
        if pred['cd'] is not None and hasattr(batch, 'y_cd') and batch.y_cd is not None:
            target_cd = batch.y_cd.reshape(-1)
            if target_cd.abs().sum() > 0:  # skip if all zeros (no metadata)
                losses['cd'] = global_coeff_loss(
                    pred['cd'], target_cd, mode=self.loss_types['cd']
                )

        # ── Cl loss ────────────────────────────────────────────────
        if pred['cl'] is not None and hasattr(batch, 'y_cl') and batch.y_cl is not None:
            target_cl = batch.y_cl.reshape(-1)
            if target_cl.abs().sum() > 0:
                losses['cl'] = global_coeff_loss(
                    pred['cl'], target_cl, mode=self.loss_types['cl']
                )

        # ── Weighted total ─────────────────────────────────────────
        total = sum(
            self.weights.get(k, 1.0) * v
            for k, v in losses.items()
        )

        return total, losses