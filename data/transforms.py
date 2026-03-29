# data/transforms.py
"""
Normalisation and data augmentation for F1 aero meshes.

Normalisation is critical: GEM-CNN kernels learn angular patterns,
but if vertex coordinates span vastly different ranges between designs,
the model will struggle to generalise. We normalise each mesh to a
unit bounding box centred at the origin.
"""

import torch
from torch import Tensor


def normalise_mesh(x: Tensor) -> Tensor:
    """
    Normalise the XYZ coordinates of a mesh to fit within [-1, 1]³.
    The U_inf channel (last column) is left unchanged.

    Args:
        x: (V, 4)  [x, y, z, U_inf]
    Returns:
        x_norm: (V, 4)  with xyz in [-1, 1]³
    """
    xyz = x[:, :3]
    rest = x[:, 3:]

    # Centre then scale
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    centre  = (xyz_min + xyz_max) / 2.0
    scale   = (xyz_max - xyz_min).max().clamp(min=1e-8) / 2.0

    xyz_norm = (xyz - centre) / scale
    return torch.cat([xyz_norm, rest], dim=-1)


def normalise_fields(cp: Tensor, wss: Tensor):
    """
    Standardise Cp and WSS fields to zero mean, unit variance.
    Returns (cp_norm, wss_norm, stats) where stats can be stored
    to invert the normalisation at evaluation time.

    For WSS (vector field), we standardise by the scalar magnitude
    std to preserve direction information.
    """
    cp_mean = cp.mean()
    cp_std  = cp.std().clamp(min=1e-8)
    cp_norm = (cp - cp_mean) / cp_std

    wss_mean = wss.mean()
    wss_std  = wss.std().clamp(min=1e-8)
    wss_norm = (wss - wss_mean) / wss_std

    stats = {
        'cp_mean': cp_mean, 'cp_std': cp_std,
        'wss_mean': wss_mean, 'wss_std': wss_std,
    }
    return cp_norm, wss_norm, stats


def denormalise_cp(cp_norm: Tensor, stats: dict) -> Tensor:
    return cp_norm * stats['cp_std'] + stats['cp_mean']


def denormalise_wss(wss_norm: Tensor, stats: dict) -> Tensor:
    return wss_norm * stats['wss_std'] + stats['wss_mean']


def random_rotation_augment(x: Tensor, angles: Tensor, transporters: Tensor,
                             sigma_deg: float = 5.0):
    """
    Apply a small random rotation around the vertical (Z) axis to augment
    training data. This simulates slight yaw angle variations in F1 testing
    (crosswind, track camber).

    Only rotates the XYZ positions — angles and transporters are recomputed
    from the rotated mesh in the full pipeline, so this is a lightweight
    approximation suitable for on-the-fly augmentation.

    Args:
        x          : (V, 4) node features
        angles     : (E,)   edge angles (returned unchanged — caller should
                            recompute geometry for exact augmentation)
        transporters: (E,)  parallel transporters (same caveat)
        sigma_deg  : std of random yaw angle in degrees
    Returns:
        x_aug: (V, 4) with rotated XYZ
    """
    angle_rad = torch.randn(1).item() * (sigma_deg * 3.14159 / 180.0)
    c, s = torch.cos(torch.tensor(angle_rad)), torch.sin(torch.tensor(angle_rad))
    Rz = torch.tensor([
        [c, -s, 0.],
        [s,  c, 0.],
        [0., 0., 1.],
    ])
    xyz_rot = x[:, :3] @ Rz.T
    return torch.cat([xyz_rot, x[:, 3:]], dim=-1)
