# eval/evaluator.py
"""
Evaluation for F1AeroNet predictions.

Metrics:
  - Per-vertex Cp:  RMSE, MAE, R²
  - Per-vertex WSS: RMSE, MAE, angular error (direction accuracy)
  - Cd:             absolute error, relative error %
  - Cl:             absolute error, relative error %

Force integration:
  Cd and Cl can also be computed by integrating the predicted Cp and WSS
  fields over the mesh surface — this is the physically grounded approach
  and a useful consistency check against the directly predicted scalars.

Usage:
    python -m eval.evaluator --config configs/f1_base.yaml --checkpoint runs/best.pt
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset, make_synthetic_dataset


# ────────────────────────────────────────────────────────────────────────────
# Metric functions
# ────────────────────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return (1.0 - ss_res / (ss_tot + 1e-10)).item()


def vector_angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean angular error between predicted and target WSS vectors (degrees).
    Useful for checking if the model captures shear direction correctly.
    """
    pred_n   = torch.nn.functional.normalize(pred,   dim=-1)
    target_n = torch.nn.functional.normalize(target, dim=-1)
    cos_sim  = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    angle_deg = torch.acos(cos_sim) * 180.0 / torch.pi
    return angle_deg.mean().item()


# ────────────────────────────────────────────────────────────────────────────
# Force integration from surface fields
# ────────────────────────────────────────────────────────────────────────────

def integrate_forces(
    cp:       torch.Tensor,   # (V,)    surface pressure coefficient
    wss:      torch.Tensor,   # (V, 3)  wall shear stress vectors
    normals:  torch.Tensor,   # (V, 3)  outward surface normals
    areas:    torch.Tensor,   # (V,)    vertex area weights (Voronoi areas)
    rho:      float = 1.225,
    U_inf:    float = 83.33,
    A_ref:    float = 1.0,    # reference area [m²]
) -> dict:
    """
    Integrate Cp and WSS fields to obtain total aerodynamic forces.

    Pressure force:   F_p = -∫ Cp * q_inf * n̂ dA
    Viscous force:    F_v =  ∫ τ dA
    Total force:      F   = F_p + F_v

    Cd = F_x / (q_inf * A_ref)       (drag — streamwise, x-direction)
    Cl = -F_z / (q_inf * A_ref)      (downforce — vertical, z-direction)
                                      (negative because downforce is -z)
    """
    q_inf = 0.5 * rho * U_inf ** 2

    # Vertex-area-weighted pressure force
    # dF_p at vertex i = -Cp_i * q_inf * n_i * A_i
    F_pressure = -(cp * areas).unsqueeze(-1) * normals    # (V, 3)  [non-dim]

    # Viscous force (WSS is already non-dimensionalised)
    F_viscous  = (wss * areas.unsqueeze(-1))              # (V, 3)

    F_total = (F_pressure + F_viscous).sum(dim=0)         # (3,)  [total force vector, non-dim]

    cd = F_total[0].item() / A_ref      # streamwise (x)
    cl = -F_total[2].item() / A_ref     # vertical   (z), negative = downforce

    return {
        'cd_integrated': cd,
        'cl_integrated': cl,
        'F_total': F_total.tolist(),
    }


def compute_vertex_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute Voronoi vertex area weights from mesh faces.
    Each vertex gets 1/3 of the area of each adjacent face.
    Returns (V,) tensor.
    """
    V = vertices.shape[0]
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_areas = torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1) / 2.0  # (F,)

    vert_areas = torch.zeros(V, device=vertices.device)
    for i in range(3):
        vert_areas.scatter_add_(0, faces[:, i], face_areas / 3.0)
    return vert_areas


# ────────────────────────────────────────────────────────────────────────────
# Full evaluation pass
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:   F1AeroNet,
    loader:  DataLoader,
    device:  torch.device,
    verbose: bool = True,
) -> dict:
    """
    Run evaluation over all samples in loader.

    Returns dict of aggregated metrics.
    """
    model.eval()

    all_cp_pred, all_cp_true   = [], []
    all_wss_pred, all_wss_true = [], []
    all_cd_pred, all_cd_true   = [], []
    all_cl_pred, all_cl_true   = [], []

    for batch in loader:
        batch = batch.to(device)

        pred = model(
            x            = batch.x,
            edge_index   = batch.edge_index,
            angles       = batch.edge_angles,
            transporters = batch.edge_transporters,
            batch        = batch.batch if hasattr(batch, 'batch') else None,
        )

        if pred['cp'] is not None and batch.y_cp is not None:
            all_cp_pred.append(pred['cp'].cpu())
            all_cp_true.append(batch.y_cp.cpu())

        if pred['wss'] is not None and batch.y_wss is not None:
            all_wss_pred.append(pred['wss'].cpu())
            all_wss_true.append(batch.y_wss.cpu())

        if pred['cd'] is not None and batch.y_cd is not None:
            all_cd_pred.append(pred['cd'].cpu())
            all_cd_true.append(batch.y_cd.reshape(-1).cpu())

        if pred['cl'] is not None and batch.y_cl is not None:
            all_cl_pred.append(pred['cl'].cpu())
            all_cl_true.append(batch.y_cl.reshape(-1).cpu())

    metrics = {}

    if all_cp_pred:
        cp_p = torch.cat(all_cp_pred)
        cp_t = torch.cat(all_cp_true)
        metrics.update({
            'cp_rmse': rmse(cp_p, cp_t),
            'cp_mae':  mae(cp_p, cp_t),
            'cp_r2':   r2_score(cp_p, cp_t),
        })

    if all_wss_pred:
        wss_p = torch.cat(all_wss_pred)
        wss_t = torch.cat(all_wss_true)
        metrics.update({
            'wss_rmse':       rmse(wss_p, wss_t),
            'wss_mae':        mae(wss_p, wss_t),
            'wss_angle_err':  vector_angular_error_deg(wss_p, wss_t),
        })

    if all_cd_pred:
        cd_p = torch.cat(all_cd_pred)
        cd_t = torch.cat(all_cd_true)
        metrics.update({
            'cd_mae': mae(cd_p, cd_t),
            'cd_rel': (torch.abs(cd_p - cd_t) / (cd_t.abs() + 1e-6)).mean().item() * 100,
        })

    if all_cl_pred:
        cl_p = torch.cat(all_cl_pred)
        cl_t = torch.cat(all_cl_true)
        metrics.update({
            'cl_mae': mae(cl_p, cl_t),
            'cl_rel': (torch.abs(cl_p - cl_t) / (cl_t.abs() + 1e-6)).mean().item() * 100,
        })

    if verbose:
        print("\n── Evaluation Results ───────────────────────────────")
        if 'cp_rmse' in metrics:
            print(f"  Cp   RMSE={metrics['cp_rmse']:.5f}  "
                  f"MAE={metrics['cp_mae']:.5f}  "
                  f"R²={metrics['cp_r2']:.4f}")
        if 'wss_rmse' in metrics:
            print(f"  WSS  RMSE={metrics['wss_rmse']:.5f}  "
                  f"MAE={metrics['wss_mae']:.5f}  "
                  f"AngErr={metrics['wss_angle_err']:.2f}°")
        if 'cd_mae' in metrics:
            print(f"  Cd   MAE={metrics['cd_mae']:.5f}  "
                  f"RelErr={metrics['cd_rel']:.2f}%")
        if 'cl_mae' in metrics:
            print(f"  Cl   MAE={metrics['cl_mae']:.5f}  "
                  f"RelErr={metrics['cl_rel']:.2f}%")
        print("─────────────────────────────────────────────────────\n")

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate F1AeroNet')
    parser.add_argument('--config',     default='configs/f1_base.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    model = F1AeroNet.from_config(cfg['model']).to(device)
    ckpt_path = args.checkpoint or cfg['eval']['checkpoint']
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path} — using random weights")

    # Load data
    data_cfg = cfg['data']
    mesh_dir = os.path.join(data_cfg['data_root'], 'meshes')
    if os.path.exists(mesh_dir):
        test_data = DrivAerNetDataset(
            data_root    = data_cfg['data_root'],
            split        = 'test',
            max_vertices = data_cfg.get('max_vertices'),
        )
    else:
        print("Using synthetic test data")
        test_data = make_synthetic_dataset(n_meshes=8, n_vertices=300)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    evaluate(model, test_loader, device, verbose=True)
