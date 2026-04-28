#!/usr/bin/env python3
"""
visualise_local.py — Run on Mac M4 from the project root.

Loads best.pt, runs inference on drivaernet_tiny meshes, and generates:
  1. Cp contour maps (ground truth vs prediction vs error)
  2. WSS magnitude maps
  3. Scatter plots (predicted vs true, all vertices)
  4. Error histograms
  5. VTP files for ParaView (optional, requires pyvista)

Usage:
    cd /Users/aum/Desktop/f1_aero_gem
    python visualise_local.py

    # Or with custom paths:
    python visualise_local.py \
        --checkpoint runs/best.pt \
        --data_root data/drivaernet_tiny \
        --out outputs/viz_2 \
        --split val \
        --max_samples 5
"""

import os
import sys
import argparse
import torch
import numpy as np
from data.transforms import inv_symlog
from data.drivaernet_dataset import load_merged_vtp
import matplotlib
matplotlib.use('TkAgg')  # interactive backend for Mac — change to 'Agg' for headless
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[INFO] pyvista not installed — VTP export disabled. pip install pyvista vtk")




# ═══════════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        print("✓ Using Metal Performance Shaders (MPS)")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("✓ Using CUDA")
        return torch.device('cuda')
    else:
        print("⚠ Using CPU")
        return torch.device('cpu')


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

# def load_model(checkpoint_path, device):
#     """Load model from checkpoint, handling both trainer formats."""
#     ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

#     # Extract config
#     cfg = ckpt.get('config', ckpt.get('cfg', {}))
#     model_cfg = cfg.get('model', {
#         'in_channels': 4,
#         'layer_types': [[8,2],[8,2],[16,2],[16,2],[8,1],[8,1]],
#         'nonlin_samples': 5,
#         'head_dropout': 0.1,
#         'break_symmetry_final': True,
#     })

#     model = F1AeroNet.from_config(model_cfg)

#     # Handle different state dict keys
#     state_dict = (
#         ckpt.get('model_state_dict')
#         or ckpt.get('model')
#         or ckpt  # raw state dict
#     )
#     model.load_state_dict(state_dict, strict=False)
#     model = model.to(device)
#     model.eval()

#     epoch = ckpt.get('epoch', '?')
#     best_val = ckpt.get('best_val', '?')
#     print(f"✓ Model loaded from {checkpoint_path}")
#     print(f"  Epoch: {epoch}, Best val: {best_val}")
#     params = model.count_parameters()
#     print(f"  Parameters: {params['total']:,}")

#     return model

def load_model(checkpoint_path, device):
    """Load model, inferring architecture strictly from checkpoint contents."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Step 1: try to get model config that was saved alongside the weights
    saved_cfg  = ckpt.get('cfg', ckpt.get('config', {}))
    model_cfg  = saved_cfg.get('model', None)
    state_dict = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt

    # Step 2: if no config saved, infer in_channels from input_embed weight shape
    #         and abort early with a clear message so the user can hardcode it
    if model_cfg is None:
        embed_w = state_dict.get('input_embed.weight')
        if embed_w is not None:
            in_ch = embed_w.shape[1]
            print(f"  [INFO] input_embed.weight shape: {tuple(embed_w.shape)}")
            print(f"         → in_channels={in_ch}, first_dim={embed_w.shape[0]}")
        print("\n  [ERROR] Checkpoint contains no 'cfg'/'config' key.")
        print("  Cannot auto-infer layer_types. Set KAGGLE_ARCH below to match")
        print("  the layer_types used in trainer_kaggle.py and re-run.\n")
        raise RuntimeError("No model config in checkpoint — see message above.")

    print(f"  layer_types from checkpoint: {model_cfg.get('layer_types')}")
    model = F1AeroNet.from_config(model_cfg)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded {checkpoint_path}  (epoch={ckpt.get('epoch','?')})")
    print(f"  Parameters: {model.count_parameters()['total']:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, data, device):
    """Run model on one sample."""
    data = data.to(device)
    pred = model(
        x=data.x,
        edge_index=data.edge_index,
        angles=data.edge_angles,
        transporters=data.edge_transporters,
    )
    return {
        'cp':  pred['cp'].cpu().numpy(),
        'wss': pred['wss'].cpu().numpy(),
        'cd':  pred['cd'].cpu().numpy() if pred['cd'] is not None else None,
        'cl':  pred['cl'].cpu().numpy() if pred['cl'] is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_cp_comparison(verts, cp_pred, cp_true, design_id, out_dir, show=False):
    """Side-by-side Cp contour: ground truth | prediction | error."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Surface Pressure (Cp) — {design_id}', fontsize=14, fontweight='bold')

    x, z = verts[:, 0], verts[:, 2]
    triang = tri.Triangulation(x, z)

    vmin = min(cp_pred.min(), cp_true.min())
    vmax = max(cp_pred.max(), cp_true.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    for ax, d, title, cmap in zip(
        axes,
        [cp_true, cp_pred, np.abs(cp_pred - cp_true)],
        ['Ground Truth (CFD)', 'Predicted (GEM-CNN)', 'Absolute Error'],
        ['RdBu_r', 'RdBu_r', 'hot'],
    ):
        if title == 'Absolute Error':
            tcf = ax.tricontourf(triang, d, levels=50, cmap=cmap)
        else:
            tcf = ax.tricontourf(triang, d, levels=50, cmap=cmap, norm=norm)
        fig.colorbar(tcf, ax=ax, label='Cp' if 'Error' not in title else '|Error|')
        ax.set_title(title)
        ax.set_xlabel('x (streamwise)')
        ax.set_ylabel('z (vertical)')
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{design_id}_cp.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_wss_comparison(verts, wss_pred, wss_true, design_id, out_dir, show=False):
    """Side-by-side WSS magnitude contour."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Wall Shear Stress |WSS| — {design_id}', fontsize=14, fontweight='bold')

    x, z = verts[:, 0], verts[:, 2]
    triang = tri.Triangulation(x, z)

    mag_true = np.linalg.norm(wss_true, axis=1)
    mag_pred = np.linalg.norm(wss_pred, axis=1)
    mag_err  = np.abs(mag_pred - mag_true)
    vmax = max(mag_true.max(), mag_pred.max())

    for ax, d, title, cmap in zip(
        axes,
        [mag_true, mag_pred, mag_err],
        ['Ground Truth', 'Predicted', 'Absolute Error'],
        ['viridis', 'viridis', 'hot'],
    ):
        if title == 'Absolute Error':
            tcf = ax.tricontourf(triang, d, levels=50, cmap=cmap)
        else:
            tcf = ax.tricontourf(triang, d, levels=50, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(tcf, ax=ax, label='|WSS|' if 'Error' not in title else '|Error|')
        ax.set_title(title)
        ax.set_xlabel('x (streamwise)')
        ax.set_ylabel('z (vertical)')
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{design_id}_wss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_3view_cp(verts, cp_data, title_prefix, design_id, out_dir, show=False):
    """3-view Cp plot: top (XY), side (XZ), front (YZ)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{title_prefix} — {design_id}', fontsize=14, fontweight='bold')

    views = [
        (verts[:, 0], verts[:, 1], 'x', 'y', 'Top View'),
        (verts[:, 0], verts[:, 2], 'x', 'z', 'Side View'),
        (verts[:, 1], verts[:, 2], 'y', 'z', 'Front View'),
    ]

    vmin, vmax = cp_data.min(), cp_data.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    for ax, (px, py, xl, yl, vtitle) in zip(axes, views):
        triang = tri.Triangulation(px, py)
        tcf = ax.tricontourf(triang, cp_data, levels=50, cmap='RdBu_r', norm=norm)
        fig.colorbar(tcf, ax=ax, label='Cp')
        ax.set_title(vtitle)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect('equal')

    plt.tight_layout()
    suffix = title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(out_dir, f'{design_id}_{suffix}_3view.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_scatter(all_results, out_dir, show=False):
    """Scatter: predicted vs true for Cp and |WSS| across all meshes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Predicted vs Ground Truth (all vertices)', fontsize=13)

    # Subsample for readability
    cp_true_all = np.concatenate([r['cp_true'] for r in all_results])
    cp_pred_all = np.concatenate([r['cp_pred'] for r in all_results])
    n = len(cp_true_all)
    idx = np.random.choice(n, min(10000, n), replace=False)

    ax = axes[0]
    ax.scatter(cp_true_all[idx], cp_pred_all[idx], alpha=0.15, s=3, c='steelblue')
    lims = [min(cp_true_all.min(), cp_pred_all.min()),
            max(cp_true_all.max(), cp_pred_all.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    corr = np.corrcoef(cp_true_all, cp_pred_all)[0, 1]
    ax.set_xlabel('True Cp (normalised)')
    ax.set_ylabel('Predicted Cp')
    ax.set_title(f'Cp  (corr={corr:.4f})')
    ax.legend()

    wss_true_all = np.concatenate([np.linalg.norm(r['wss_true'], axis=1) for r in all_results])
    wss_pred_all = np.concatenate([np.linalg.norm(r['wss_pred'], axis=1) for r in all_results])
    idx2 = np.random.choice(len(wss_true_all), min(10000, len(wss_true_all)), replace=False)

    ax = axes[1]
    ax.scatter(wss_true_all[idx2], wss_pred_all[idx2], alpha=0.15, s=3, c='coral')
    lims = [0, max(wss_true_all.max(), wss_pred_all.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True |WSS| (normalised)')
    ax.set_ylabel('Predicted |WSS|')
    ax.set_title('WSS Magnitude')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, 'scatter_all.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_error_histogram(all_results, out_dir, show=False):
    """Histogram of per-vertex errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Per-vertex Error Distribution', fontsize=13)

    cp_errors = np.concatenate([r['cp_pred'] - r['cp_true'] for r in all_results])
    axes[0].hist(cp_errors, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_xlabel('Cp Error (pred − true)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Cp Error  MAE={np.abs(cp_errors).mean():.4f}')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    wss_errors = np.concatenate([
        np.linalg.norm(r['wss_pred'] - r['wss_true'], axis=1) for r in all_results
    ])
    axes[1].hist(wss_errors, bins=100, color='coral', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel('WSS Vector Error (L2 norm)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'WSS Error  mean={wss_errors.mean():.4f}')

    plt.tight_layout()
    path = os.path.join(out_dir, 'error_histograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# VTP export for ParaView
# ═══════════════════════════════════════════════════════════════════════════

def export_vtp(verts, faces, fields, out_path):
    """Write prediction fields to VTP for ParaView visualisation."""
    if not HAS_PYVISTA:
        return

    n_faces = faces.shape[0]
    pvfaces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        faces,
    ]).flatten()

    mesh = pv.PolyData(verts, pvfaces)
    for name, arr in fields.items():
        if arr is not None:
            mesh.point_data[name] = arr

    mesh.save(out_path)
    print(f"    Saved VTP: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualise F1AeroNet predictions locally')
    parser.add_argument('--checkpoint', default='runs_200/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', default='data/drivaernet_tiny',
                        help='Path to dataset root')
    parser.add_argument('--out', default='outputs/viz_2',
                        help='Output directory for plots and VTPs')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                        help='Which data split to visualise')
    parser.add_argument('--max_samples', type=int, default=2,
                        help='Max number of meshes to visualise')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (in addition to saving)')
    parser.add_argument('--no_vtp', action='store_true',
                        help='Skip VTP export even if pyvista is available')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device()

    # ── Model ─────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {args.split} split from {args.data_root}")
    ds = DrivAerNetDataset(
        data_root=args.data_root,
        split=args.split,
        force_reload=False,
    )
    n_samples = min(args.max_samples, len(ds))
    print(f"  {len(ds)} meshes available, visualising {n_samples}\n")

    # ── Run inference and generate plots ──────────────────────────────────
    print("=" * 60)
    print("GENERATING VISUALISATIONS")
    print("=" * 60)

    all_results = []

    for i in range(n_samples):
        data = ds[i]
        did = data.design_id if hasattr(data, 'design_id') else f'{args.split}_{i}'

        print(f"\n[{i+1}/{n_samples}] {did}")
        print("-" * 40)

        pred = predict(model, data, device)
        verts    = data.pos.cpu().numpy()
        cp_true  = data.y_cp.cpu().numpy()
        wss_true = data.y_wss.cpu().numpy()
        cp_pred  = pred['cp']
        wss_pred = pred['wss']

        # Add this to visualise_local.py, after getting predictions



    # Load raw mesh to get original Cp for denorm stats
    vtp_path = os.path.join(args.data_root, 'meshes', f'{did}.vtp')
    raw = load_merged_vtp(vtp_path)

    # Recompute the normalisation stats (same as mesh_to_pyg_data)


    rho, U_inf = 1.225, 83.33
    q_inf = 0.5 * rho * U_inf ** 2

    # Cp stats
    cp_raw = (raw['pressure'] - 0.0) / q_inf
    cp_sl = np.sign(cp_raw) * np.log1p(np.abs(cp_raw))
    cp_mean = cp_sl.mean()
    cp_std = cp_sl.std().clip(1e-8)

    # WSS stats  
    mu, L_ref = 1.81e-5, 4.6
    tau_ref = mu * U_inf / L_ref
    wss_raw = raw['wss'] / tau_ref
    wss_sl = np.sign(wss_raw) * np.log1p(np.abs(wss_raw))
    wss_mean = wss_sl.mean()
    wss_std = wss_sl.std().clip(1e-8)

    # Denormalise predictions back to physical units
    cp_denorm = cp_pred * cp_std + cp_mean          # undo standardisation
    cp_physical = np.sign(cp_denorm) * (np.exp(np.abs(cp_denorm)) - 1.0)  # undo symlog

    wss_denorm = wss_pred * wss_std + wss_mean
    wss_physical = np.sign(wss_denorm) * (np.exp(np.abs(wss_denorm)) - 1.0)

    result = {
            'design_id': did,
            'cp_pred': cp_pred, 'cp_true': cp_true,
            'wss_pred': wss_pred, 'wss_true': wss_true,
        }
    all_results.append(result)

        # Stats
    cp_mae  = np.abs(cp_pred - cp_true).mean()
    cp_corr = np.corrcoef(cp_pred, cp_true)[0, 1]
    wss_mae = np.abs(wss_pred - wss_true).mean()
    print(f"  Cp  MAE={cp_mae:.4f}  corr={cp_corr:.4f}")
    print(f"  WSS MAE={wss_mae:.4f}")

    if pred['cd'] is not None and hasattr(data, 'y_cd') and data.y_cd is not None:
        print(f"  Cd  pred={pred['cd'].item():.4f}  true={data.y_cd.item():.4f}")

    # Per-mesh plots
    plot_cp_comparison(verts, cp_pred, cp_true, did, args.out, show=args.show)
    plot_wss_comparison(verts, wss_pred, wss_true, did, args.out, show=args.show)
    plot_3view_cp(verts, cp_pred, 'Cp Predicted', did, args.out, show=args.show)

    # VTP export
    if HAS_PYVISTA and not args.no_vtp and hasattr(data, 'face'):
        faces = data.face.T.cpu().numpy()
        export_vtp(verts, faces, {
            'Cp_pred':  cp_pred,
            'Cp_true':  cp_true,
            'Cp_error': np.abs(cp_pred - cp_true),
            'WSS_pred_mag': np.linalg.norm(wss_pred, axis=1),
            'WSS_true_mag': np.linalg.norm(wss_true, axis=1),
            'WSS_pred': wss_pred,
            'WSS_true': wss_true,
        }, os.path.join(args.out, f'{did}_predictions.vtp'))

    # ── Summary plots ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY PLOTS")
    print("=" * 60)

    plot_scatter(all_results, args.out, show=args.show)
    plot_error_histogram(all_results, args.out, show=args.show)

    # ── Print summary ─────────────────────────────────────────────────────
    cp_mae_avg  = np.mean([np.abs(r['cp_pred'] - r['cp_true']).mean() for r in all_results])
    cp_corr_avg = np.mean([np.corrcoef(r['cp_pred'], r['cp_true'])[0,1] for r in all_results])
    wss_mae_avg = np.mean([np.abs(r['wss_pred'] - r['wss_true']).mean() for r in all_results])

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Meshes evaluated:  {n_samples}")
    print(f"  Avg Cp  MAE:       {cp_mae_avg:.4f}")
    print(f"  Avg Cp  Corr:      {cp_corr_avg:.4f}")
    print(f"  Avg WSS MAE:       {wss_mae_avg:.4f}")
    print(f"\n  Plots saved to:    {os.path.abspath(args.out)}/")

    if HAS_PYVISTA and not args.no_vtp:
        print(f"\n  VTP files for ParaView also saved.")
        print(f"  Open in ParaView:  paraview {args.out}/*.vtp")

    print(f"\n{'=' * 60}")
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()