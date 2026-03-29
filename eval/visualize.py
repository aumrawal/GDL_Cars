# eval/visualize.py
"""
Visualisation utilities for F1AeroNet predictions.

Outputs:
  1. VTK PolyData (.vtp) files for ParaView — predicted and ground-truth
     Cp, WSS fields rendered on the car surface mesh.
  2. Quick matplotlib plots — Cp distribution comparison, Cd/Cl scatter.

Usage:
    python -m eval.visualize --config configs/f1_base.yaml \
                              --checkpoint runs/best.pt \
                              --design design_0001 \
                              --out outputs/design_0001_pred.vtp
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset, make_synthetic_dataset

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ────────────────────────────────────────────────────────────────────────────
# VTK export
# ────────────────────────────────────────────────────────────────────────────

def export_vtp(
    vertices: np.ndarray,    # (V, 3)
    faces:    np.ndarray,    # (F, 3)
    fields:   dict,          # name → np.ndarray (V,) or (V,3)
    out_path: str,
):
    """
    Write predicted fields to a VTK PolyData file.
    Open in ParaView to visualise Cp/WSS coloured on the car surface.
    """
    if not HAS_PYVISTA:
        print("[WARNING] pyvista not available — skipping VTK export.")
        return

    # Build pyvista mesh
    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = faces.shape[0]
    pvfaces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        faces,
    ]).flatten()

    mesh = pv.PolyData(vertices, pvfaces)

    # Add fields
    for name, arr in fields.items():
        if arr is not None:
            mesh.point_data[name] = arr

    mesh.save(out_path)
    print(f"Saved VTK output: {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Matplotlib plots
# ────────────────────────────────────────────────────────────────────────────

def plot_cp_comparison(
    vertices:   np.ndarray,    # (V, 3)
    cp_pred:    np.ndarray,    # (V,)
    cp_true:    np.ndarray,    # (V,)
    out_path:   str = None,
):
    """
    Side-by-side top-view Cp contour plot (predicted vs ground truth).
    Colour map: blue (low pressure / suction) → red (high pressure / stagnation)
    """
    if not HAS_MPL:
        print("[WARNING] matplotlib not available — skipping Cp plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Surface Pressure Coefficient (Cp)', fontsize=14, fontweight='bold')

    x, z = vertices[:, 0], vertices[:, 2]
    triang = tri.Triangulation(x, z)
    vmin = min(cp_pred.min(), cp_true.min())
    vmax = max(cp_pred.max(), cp_true.max())

    for ax, data, title in zip(axes, [cp_true, cp_pred], ['Ground Truth (CFD)', 'Predicted (GEM-CNN)']):
        tcf = ax.tricontourf(triang, data, levels=50, cmap='RdBu_r',
                             vmin=vmin, vmax=vmax)
        fig.colorbar(tcf, ax=ax, label='Cp')
        ax.set_title(title)
        ax.set_xlabel('x (streamwise) [m]')
        ax.set_ylabel('z (vertical) [m]')
        ax.set_aspect('equal')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved Cp comparison plot: {out_path}")
    else:
        plt.show()
    plt.close()


def plot_cd_cl_scatter(
    cd_pred: np.ndarray,  # (N,)
    cd_true: np.ndarray,
    cl_pred: np.ndarray,  # (N,)
    cl_true: np.ndarray,
    out_path: str = None,
):
    """Scatter plot of predicted vs true Cd and Cl."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Global Aerodynamic Coefficients', fontsize=13, fontweight='bold')

    for ax, pred, true, label in zip(
        axes,
        [cd_pred, cl_pred],
        [cd_true, cl_true],
        ['Drag coefficient (Cd)', 'Downforce coefficient (Cl)'],
    ):
        lo = min(true.min(), pred.min()) * 0.95
        hi = max(true.max(), pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='Perfect prediction')
        ax.scatter(true, pred, alpha=0.7, s=40, c='steelblue', edgecolors='white', lw=0.5)
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(label)
        ax.legend(fontsize=8)

        mae_val = np.mean(np.abs(pred - true))
        ax.text(0.05, 0.92, f'MAE = {mae_val:.4f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved Cd/Cl scatter plot: {out_path}")
    else:
        plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Main visualisation routine
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualise_sample(
    model:     F1AeroNet,
    data,
    device:    torch.device,
    out_dir:   str,
    design_id: str = 'sample',
):
    """Run model on one sample and write all visualisation outputs."""
    os.makedirs(out_dir, exist_ok=True)

    data = data.to(device)
    pred = model(
        x            = data.x,
        edge_index   = data.edge_index,
        angles       = data.edge_angles,
        transporters = data.edge_transporters,
    )

    verts = data.pos.cpu().numpy()         # (V, 3)
    faces = data.face.T.cpu().numpy()      # (F, 3)

    cp_pred  = pred['cp'].cpu().numpy()
    wss_pred = pred['wss'].cpu().numpy()
    cp_true  = data.y_cp.cpu().numpy()  if data.y_cp  is not None else np.zeros_like(cp_pred)
    wss_true = data.y_wss.cpu().numpy() if data.y_wss is not None else np.zeros_like(wss_pred)

    # 1. VTP export (ParaView)
    export_vtp(
        vertices = verts,
        faces    = faces,
        fields   = {
            'Cp_pred':  cp_pred,
            'Cp_true':  cp_true,
            'Cp_error': np.abs(cp_pred - cp_true),
            'WSS_pred': wss_pred,
            'WSS_true': wss_true,
        },
        out_path = os.path.join(out_dir, f'{design_id}_predictions.vtp'),
    )

    # 2. Cp comparison plot
    plot_cp_comparison(
        vertices = verts,
        cp_pred  = cp_pred,
        cp_true  = cp_true,
        out_path = os.path.join(out_dir, f'{design_id}_cp_comparison.png'),
    )

    # 3. Print Cd/Cl
    print(f"\n  Cd predicted: {pred['cd'].item():.4f}"
          f"  true: {data.y_cd.item():.4f}")
    print(f"  Cl predicted: {pred['cl'].item():.4f}"
          f"  true: {data.y_cl.item():.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/f1_base.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out',        default='outputs/')
    parser.add_argument('--design',     default=None, help='Design ID to visualise')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model  = F1AeroNet.from_config(cfg['model']).to(device)

    ckpt_path = args.checkpoint or cfg['eval']['checkpoint']
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])

    # Get one sample
    data_cfg = cfg['data']
    mesh_dir = os.path.join(data_cfg['data_root'], 'meshes')
    if os.path.exists(mesh_dir):
        dataset = DrivAerNetDataset(data_root=data_cfg['data_root'], split='test',
                                    max_vertices=data_cfg.get('max_vertices'))
        sample = dataset[0]
        did = args.design or sample.design_id
    else:
        sample = make_synthetic_dataset(n_meshes=1, n_vertices=300)[0]
        did = 'synthetic_0000'

    model.eval()
    visualise_sample(model, sample, device, out_dir=args.out, design_id=did)
