# eval/visualize.py
"""
Visualisation utilities for F1AeroNet predictions.

Outputs per design:
  1. VTK PolyData (.vtp)  — predicted and ground-truth Cp, WSS-magnitude,
     and per-vertex absolute error fields.  Open in ParaView.
  2. Cp comparison plot   — side-by-side top-view contour (pred vs CFD).
  3. WSS-magnitude plot   — same layout for ||WSS||.
  4. Console summary      — Cd / Cl predicted vs true.

Usage (run from project root):
    python -m eval.visualize \
        --config      configs/f1_base.yaml \
        --checkpoint  runs/best.pt \
        --data_root   data/drivaernet_real \
        --split       test \
        --n_samples   2 \
        --out         outputs/

Edits over the original script
───────────────────────────────
  • Loop over n_samples (default 2) instead of a single mesh.
  • --data_root / --split CLI args so you can point at drivaernet_real
    without touching the YAML.
  • Passes `batch` tensor to model.forward() so Cd/Cl global pooling works
    on unbatched single-Data objects.
  • Handles WSS as (V,3) vector: exports magnitude to VTP and plots ||WSS||.
  • Guards Cl output in case cl_head is None / disabled (all-zero label).
  • Robust checkpoint loading: tries 'model' key first, then bare state-dict.
  • Falls back to first N meshes from the full split when test set is empty.
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
    print("[WARNING] pyvista not installed — VTP export disabled.  "
          "Run: pip install pyvista vtk")

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not installed — plots disabled.")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _wss_magnitude(wss: np.ndarray) -> np.ndarray:
    """Return per-vertex WSS magnitude. Works for (V,) or (V,3) arrays."""
    if wss.ndim == 2:
        return np.linalg.norm(wss, axis=1)
    return np.abs(wss)


def _safe_item(t) -> float:
    """Safely convert a tensor (possibly None) to a Python float."""
    if t is None:
        return float('nan')
    return float(t.detach().cpu().reshape(-1)[0])


def _make_batch_vector(data, device: torch.device) -> torch.Tensor:
    """
    PyG Data objects loaded outside a DataLoader have no .batch attribute.
    Synthesise a zero vector of length num_nodes so global_mean_pool works.
    """
    if hasattr(data, 'batch') and data.batch is not None:
        return data.batch.to(device)
    return torch.zeros(data.num_nodes, dtype=torch.long, device=device)


# ────────────────────────────────────────────────────────────────────────────
# VTK export
# ────────────────────────────────────────────────────────────────────────────

def export_vtp(
    vertices: np.ndarray,   # (V, 3)
    faces:    np.ndarray,   # (F, 3)  int indices
    fields:   dict,         # name → np.ndarray  (V,) scalar or (V,3) vector
    out_path: str,
):
    """Write predicted fields to a VTK PolyData file (.vtp) for ParaView."""
    if not HAS_PYVISTA:
        print("[SKIP] pyvista unavailable — no VTP written.")
        return

    n_faces  = faces.shape[0]
    pv_faces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        faces.astype(np.int64),
    ]).flatten()

    mesh = pv.PolyData(vertices.astype(np.float64), pv_faces)

    for name, arr in fields.items():
        if arr is not None:
            mesh.point_data[name] = arr

    mesh.save(out_path)
    print(f"  [VTP]  saved → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Matplotlib plots
# ────────────────────────────────────────────────────────────────────────────

def _plot_scalar_comparison(
    vertices:   np.ndarray,   # (V, 3)
    pred_field: np.ndarray,   # (V,)  scalar
    true_field: np.ndarray,   # (V,)  scalar
    field_name: str,
    cmap:       str,
    out_path:   str = None,
):
    """Generic side-by-side top-view (x–z plane) contour comparison."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Surface Field: {field_name}', fontsize=14, fontweight='bold')

    x, z    = vertices[:, 0], vertices[:, 2]
    triang  = tri.Triangulation(x, z)
    vmin    = min(pred_field.min(), true_field.min())
    vmax    = max(pred_field.max(), true_field.max())
    # Avoid degenerate colormap range
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    for ax, data, title in zip(
        axes,
        [true_field, pred_field],
        [f'Ground Truth (CFD)', f'Predicted (GEM-CNN)'],
    ):
        tcf = ax.tricontourf(triang, data, levels=50, cmap=cmap,
                             vmin=vmin, vmax=vmax)
        fig.colorbar(tcf, ax=ax, label=field_name, shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x  [m]  (streamwise)')
        ax.set_ylabel('z  [m]  (vertical)')
        ax.set_aspect('equal')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  [PNG]  saved → {out_path}")
    else:
        plt.show()
    plt.close()


def plot_cp_comparison(vertices, cp_pred, cp_true, out_path=None):
    _plot_scalar_comparison(
        vertices, cp_pred, cp_true,
        field_name='Cp  (pressure coefficient)',
        cmap='RdBu_r',
        out_path=out_path,
    )


def plot_wss_comparison(vertices, wss_pred, wss_true, out_path=None):
    """Plot ||WSS|| magnitude comparison (handles vector or scalar input)."""
    _plot_scalar_comparison(
        vertices,
        _wss_magnitude(wss_pred),
        _wss_magnitude(wss_true),
        field_name='||WSS||  (wall-shear-stress magnitude)  [Pa]',
        cmap='hot',
        out_path=out_path,
    )


def plot_cd_cl_scatter(cd_pred, cd_true, cl_pred, cl_true, out_path=None):
    """Scatter plot — predicted vs true Cd and Cl (multi-sample summary)."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Global Aerodynamic Coefficients', fontsize=13, fontweight='bold')

    for ax, pred, true, label in zip(
        axes,
        [cd_pred, cl_pred],
        [cd_true, cl_true],
        ['Drag coefficient  Cd', 'Downforce coefficient  Cl'],
    ):
        pred = np.asarray(pred)
        true = np.asarray(true)
        lo   = min(true.min(), pred.min()) * 0.95
        hi   = max(true.max(), pred.max()) * 1.05
        if np.isclose(lo, hi):
            hi = lo + 1e-4
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='Perfect prediction')
        ax.scatter(true, pred, alpha=0.8, s=60, c='steelblue',
                   edgecolors='white', lw=0.5)
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
        print(f"  [PNG]  saved → {out_path}")
    else:
        plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Main visualisation routine — one sample
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualise_sample(
    model:     F1AeroNet,
    data,
    device:    torch.device,
    out_dir:   str,
    design_id: str = 'sample',
):
    """Run model on one PyG Data object and write all visualisation outputs."""
    os.makedirs(out_dir, exist_ok=True)

    data   = data.to(device)
    batch  = _make_batch_vector(data, device)

    pred = model(
        x            = data.x,
        edge_index   = data.edge_index,
        angles       = data.edge_angles,
        transporters = data.edge_transporters,
        batch        = batch,
    )

    verts = data.pos.cpu().numpy()          # (V, 3)
    # face stored as (3, F) in PyG → transpose to (F, 3)
    faces = data.face.T.cpu().numpy()       # (F, 3)

    # ── Predictions ──────────────────────────────────────────────────────
    cp_pred  = pred['cp'].cpu().numpy().reshape(-1)          # (V,)
    wss_pred = pred['wss'].cpu().numpy()                     # (V,) or (V,3)

    # ── Ground truth (fall back to zeros if not present) ─────────────────
    cp_true  = (data.y_cp.cpu().numpy().reshape(-1)
                if data.y_cp is not None else np.zeros_like(cp_pred))
    wss_true = (data.y_wss.cpu().numpy()
                if data.y_wss is not None else np.zeros_like(wss_pred))

    # ── Cd / Cl ───────────────────────────────────────────────────────────
    cd_pred_val = _safe_item(pred.get('cd'))
    cl_pred_val = _safe_item(pred.get('cl'))
    cd_true_val = _safe_item(getattr(data, 'y_cd', None))
    cl_true_val = _safe_item(getattr(data, 'y_cl', None))

    prefix = os.path.join(out_dir, design_id)

    # 1. VTP export for ParaView
    export_vtp(
        vertices = verts,
        faces    = faces,
        fields   = {
            'Cp_pred':     cp_pred,
            'Cp_true':     cp_true,
            'Cp_error':    np.abs(cp_pred - cp_true),
            'WSS_mag_pred': _wss_magnitude(wss_pred),
            'WSS_mag_true': _wss_magnitude(wss_true),
            'WSS_mag_err':  np.abs(_wss_magnitude(wss_pred) - _wss_magnitude(wss_true)),
            # Also store raw WSS vector if it's (V,3)
            **(
                {'WSS_pred_vec': wss_pred, 'WSS_true_vec': wss_true}
                if wss_pred.ndim == 2 else {}
            ),
        },
        out_path = f'{prefix}_predictions.vtp',
    )

    # 2. Cp comparison plot
    plot_cp_comparison(
        vertices = verts,
        cp_pred  = cp_pred,
        cp_true  = cp_true,
        out_path = f'{prefix}_cp_comparison.png',
    )

    # 3. WSS magnitude comparison plot
    plot_wss_comparison(
        vertices  = verts,
        wss_pred  = wss_pred,
        wss_true  = wss_true,
        out_path  = f'{prefix}_wss_comparison.png',
    )

    # 4. Console summary
    print(f"\n  ── {design_id} ──")
    print(f"     Cd   pred={cd_pred_val:.4f}   true={cd_true_val:.4f}   "
          f"err={abs(cd_pred_val - cd_true_val):.4f}")
    print(f"     Cl   pred={cl_pred_val:.4f}   true={cl_true_val:.4f}   "
          f"err={abs(cl_pred_val - cl_true_val):.4f}")
    print(f"     Cp   mean_abs_err={np.mean(np.abs(cp_pred - cp_true)):.4f}   "
          f"max={np.max(np.abs(cp_pred - cp_true)):.4f}")
    wss_mag_err = np.abs(_wss_magnitude(wss_pred) - _wss_magnitude(wss_true))
    print(f"     WSS  mean_abs_err={np.mean(wss_mag_err):.4f}   "
          f"max={np.max(wss_mag_err):.4f}")

    return {
        'design_id':  design_id,
        'cd_pred':    cd_pred_val,
        'cd_true':    cd_true_val,
        'cl_pred':    cl_pred_val,
        'cl_true':    cl_true_val,
    }


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, model: F1AeroNet, device: torch.device):
    """
    Load weights from checkpoint. Handles two formats:
      • dict with key 'model'  (saved by train/trainer.py → save_checkpoint)
      • bare state-dict
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint not found: {ckpt_path} — using random weights.")
        return
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
        epoch = ckpt.get('epoch', '?')
        print(f"  Loaded checkpoint '{ckpt_path}'  (epoch={epoch})")
    else:
        state = ckpt
        print(f"  Loaded bare state-dict from '{ckpt_path}'")
    model.load_state_dict(state, strict=True)


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1AeroNet — Visualise predictions')
    parser.add_argument('--config',      default='configs/f1_base.yaml',
                        help='Path to YAML config')
    parser.add_argument('--checkpoint',  default=None,
                        help='Path to best.pt (overrides config eval.checkpoint)')
    parser.add_argument('--data_root',   default=None,
                        help='Path to dataset root (overrides config data.data_root). '
                             'For 200-car run use data/drivaernet_real')
    parser.add_argument('--split',       default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to sample from (default: test)')
    parser.add_argument('--n_samples',   type=int, default=2,
                        help='Number of car meshes to visualise (default: 2)')
    parser.add_argument('--out',         default='outputs/',
                        help='Output directory for plots and VTP files')
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Device ───────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Device: CPU")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n[Model]")
    model = F1AeroNet.from_config(cfg['model']).to(device)
    param_counts = model.param_counts()
    print(f"  Total parameters: {param_counts['total']:,}")

    ckpt_path = args.checkpoint or cfg.get('eval', {}).get('checkpoint', 'runs/best.pt')
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    # ── Dataset ──────────────────────────────────────────────────────────
    print("\n[Dataset]")
    data_cfg  = cfg['data']
    data_root = args.data_root or data_cfg['data_root']
    mesh_dir  = os.path.join(data_root, 'meshes')

    if os.path.exists(mesh_dir):
        dataset = DrivAerNetDataset(
            data_root    = data_root,
            split        = args.split,
            max_vertices = data_cfg.get('max_vertices'),
            rho          = data_cfg.get('rho', 1.225),
            U_inf        = data_cfg.get('U_inf', 83.33),
        )
        print(f"  data_root : {data_root}")
        print(f"  split     : {args.split}  ({len(dataset)} meshes)")

        # If the requested split is empty (e.g. tiny dataset has no dedicated
        # test split), fall back to loading directly from 'train'.
        if len(dataset) == 0:
            print(f"  [INFO] '{args.split}' split is empty — "
                  f"falling back to 'train' split.")
            dataset = DrivAerNetDataset(
                data_root    = data_root,
                split        = 'train',
                max_vertices = data_cfg.get('max_vertices'),
                rho          = data_cfg.get('rho', 1.225),
                U_inf        = data_cfg.get('U_inf', 83.33),
            )
            print(f"  Fallback  : train split  ({len(dataset)} meshes)")
    else:
        print(f"  [INFO] Mesh dir not found ({mesh_dir}). "
              f"Using synthetic data for pipeline smoke-test.")
        dataset = make_synthetic_dataset(
            n_meshes  = max(args.n_samples, 4),
            n_vertices = 300,
            U_inf      = data_cfg.get('U_inf', 83.33),
            rho        = data_cfg.get('rho', 1.225),
        )

    n_vis = min(args.n_samples, len(dataset))
    if n_vis == 0:
        print("[ERROR] Dataset is empty — nothing to visualise.")
        sys.exit(1)

    print(f"\n  Visualising {n_vis} mesh(es) → {args.out}/\n")

    # ── Run ──────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    records = []

    for i in range(n_vis):
        sample    = dataset[i]
        design_id = getattr(sample, 'design_id', f'sample_{i:04d}')
        print(f"[{i+1}/{n_vis}]  design_id = {design_id}  "
              f"({sample.num_nodes:,} vertices)")

        rec = visualise_sample(
            model     = model,
            data      = sample,
            device    = device,
            out_dir   = args.out,
            design_id = design_id,
        )
        records.append(rec)

    # ── Multi-sample Cd/Cl scatter (only meaningful when n_vis > 1) ──────
    if n_vis > 1 and HAS_MPL:
        cd_pred_all = np.array([r['cd_pred'] for r in records])
        cd_true_all = np.array([r['cd_true'] for r in records])
        cl_pred_all = np.array([r['cl_pred'] for r in records])
        cl_true_all = np.array([r['cl_true'] for r in records])

        scatter_path = os.path.join(args.out, 'cd_cl_scatter.png')
        plot_cd_cl_scatter(
            cd_pred_all, cd_true_all,
            cl_pred_all, cl_true_all,
            out_path = scatter_path,
        )

    print("\nDone.  Open the .vtp files in ParaView to explore 3-D fields.")