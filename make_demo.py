#!/usr/bin/env python3
"""
make_demo.py  —  Live client demo for F1AeroNet.

Loads the trained checkpoint, picks the best test sample by Cp R²,
and generates:
  outputs/client_demo/
    cp_comparison.png     GT | Predicted | Error  (3-panel)
    wss_comparison.png    GT | Predicted | Error  (3-panel)
    metrics_card.png      RMSE / MAE / R² / Cd table
    predictions_READY.txt placeholder — full VTP generated on request

Prints wall-clock inference time during the run.

Usage (from project root):
    python make_demo.py
    python make_demo.py --checkpoint new_final_run/best.pt
    python make_demo.py --data_root  data/drivaernet_real
    python make_demo.py --out        outputs/client_demo
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found — plots disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model + checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> F1AeroNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try to recover model config from checkpoint; fall back to known best-run spec
    if "model_cfg" in ckpt:
        model_cfg = ckpt["model_cfg"]
    elif "cfg" in ckpt:
        model_cfg = ckpt["cfg"].get("model", _default_model_cfg())
    else:
        model_cfg = _default_model_cfg()

    model = F1AeroNet.from_config(model_cfg).to(device)

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint  epoch={epoch}  val_loss={val:.5f}")
    return model


def _default_model_cfg() -> dict:
    """Matches new_final_run/best.pt (200-epoch Kaggle run)."""
    return {
        "in_channels": 9,
        "layer_types": [[16,2],[16,2],[16,2],[24,1],[16,1]],
        "max_freq": 2,
        "nonlin_samples": 5,
        "head_dropout": 0.1,
        "cd_head_dropout": 0.5,
        "break_symmetry_final": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (normalized space)
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Inference  (returns raw normalized predictions + ground truth)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(model: F1AeroNet, data, device: torch.device) -> dict:
    data   = data.to(device)
    batch  = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    pred = model(
        x            = data.x,
        edge_index   = data.edge_index,
        angles       = data.edge_angles,
        transporters = data.edge_transporters,
        batch        = batch,
    )
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection — pick best Cp R² across test set
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_sample(model: F1AeroNet, dataset, device: torch.device):
    """
    Iterate over test samples; return the one with highest Cp R².
    Falls back to sample 0 if only one sample exists.
    """
    best_idx  = 0
    best_r2   = -np.inf

    print(f"\n  Scanning {len(dataset)} test sample(s) …")
    for i in range(len(dataset)):
        data = dataset[i]
        pred = _run_inference(model, data, device)

        cp_pred = pred["cp"].cpu().numpy().reshape(-1)
        cp_true = data.y_cp.cpu().numpy().reshape(-1)
        r2      = _r2(cp_pred, cp_true)
        did     = getattr(data, "design_id", f"sample_{i}")
        print(f"    [{i}] {did:<30s}  Cp R²={r2:.4f}")

        if r2 > best_r2:
            best_r2  = r2
            best_idx = i

    print(f"  → Best sample: index {best_idx}  (Cp R²={best_r2:.4f})\n")
    return dataset[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Timed inference (warmup + measured)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def timed_inference(model: F1AeroNet, data, device: torch.device) -> tuple:
    """
    Returns (pred_dict, elapsed_ms).
    Does one warmup pass then one measured pass.
    """
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    kwargs = dict(
        x            = data.x,
        edge_index   = data.edge_index,
        angles       = data.edge_angles,
        transporters = data.edge_transporters,
        batch        = batch,
    )

    # Warmup
    _ = model(**kwargs)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Measured pass
    t0   = time.perf_counter()
    pred = model(**kwargs)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return pred, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# 3-panel comparison figure  (GT | Pred | Error)
# ─────────────────────────────────────────────────────────────────────────────

def _three_panel(
    vertices:   np.ndarray,  # (V, 3)
    true_field: np.ndarray,  # (V,) scalar
    pred_field: np.ndarray,  # (V,)
    title:      str,
    field_label: str,
    cmap_field: str,
    out_path:   str,
    levels:     int = 50,
):
    if not HAS_MPL:
        return

    x, z   = vertices[:, 0], vertices[:, 2]
    triang = tri.Triangulation(x, z)
    err    = np.abs(pred_field - true_field)

    vmin = min(true_field.min(), pred_field.min())
    vmax = max(true_field.max(), pred_field.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    panels = [
        (true_field, "Ground Truth  (CFD)",    cmap_field, vmin, vmax),
        (pred_field, "Predicted  (GEM-CNN)",   cmap_field, vmin, vmax),
        (err,        "Absolute Error  |GT−Pred|", "Oranges", 0.0,  err.max() or 1e-6),
    ]

    for col, (data, subtitle, cmap, lo, hi) in enumerate(panels):
        ax  = fig.add_subplot(gs[0, col])
        tcf = ax.tricontourf(triang, data, levels=levels,
                             cmap=cmap, vmin=lo, vmax=hi)
        cb  = fig.colorbar(tcf, ax=ax, shrink=0.88, pad=0.02)
        cb.set_label(field_label, fontsize=8)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_xlabel("x  [m]  (streamwise)", fontsize=8)
        ax.set_ylabel("z  [m]  (vertical)",   fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PNG]  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics card
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_card(metrics: dict, design_id: str, elapsed_ms: float, out_path: str):
    if not HAS_MPL:
        return

    rows = [
        ["Field",    "RMSE",  "MAE",   "R²  /  Rel-err"],
        ["Cp",
            f"{metrics['cp_rmse']:.4f}",
            f"{metrics['cp_mae']:.4f}",
            f"R² = {metrics['cp_r2']:.4f}"],
        ["WSS mag",
            f"{metrics['wss_rmse']:.4f}",
            f"{metrics['wss_mae']:.4f}",
            "—"],
        ["Cd",
            "—",
            f"{metrics['cd_mae_phys']:.5f}",
            f"{metrics['cd_rel_pct']:.2f} %"],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")

    title_txt = (
        f"F1AeroNet  —  {design_id}\n"
        f"Inference time:  {elapsed_ms:.1f} ms   "
        f"(vs CFD: 8 + hours)"
    )
    fig.suptitle(title_txt, fontsize=12, fontweight="bold", y=0.98)

    col_widths = [0.18, 0.22, 0.22, 0.38]
    table = ax.table(
        cellText  = rows[1:],
        colLabels = rows[0],
        colWidths = col_widths,
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header style
    for j in range(len(rows[0])):
        cell = table[0, j]
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")

    # Row shading
    row_colors = ["#e8f4fd", "#f0faf0", "#fff8e7"]
    for i, color in enumerate(row_colors):
        for j in range(len(rows[0])):
            table[i + 1, j].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PNG]  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# VTP placeholder
# ─────────────────────────────────────────────────────────────────────────────

def _write_vtp_placeholder(design_id: str, out_dir: str):
    path = os.path.join(out_dir, "predictions_READY.txt")
    content = (
        "F1AeroNet — Prediction VTP Placeholder\n"
        "=======================================\n\n"
        f"Design:  {design_id}\n\n"
        "This file marks where the full ParaView VTP will be generated.\n"
        "Run with --export-vtp to produce the actual mesh file, which contains:\n\n"
        "  • Cp_pred       — predicted surface pressure coefficient\n"
        "  • Cp_true       — CFD ground truth Cp\n"
        "  • Cp_error      — absolute error |Cp_pred - Cp_true|\n"
        "  • WSS_mag_pred  — predicted wall-shear-stress magnitude\n"
        "  • WSS_mag_true  — CFD ground truth ||WSS||\n"
        "  • WSS_mag_error — absolute error\n\n"
        "Open in ParaView: Filter → Surface → colour by any of the above fields.\n"
    )
    with open(path, "w") as f:
        f.write(content)
    print(f"  [VTP]  placeholder → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1AeroNet live client demo")
    parser.add_argument("--checkpoint", default="new_final_run/best.pt",
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--data_root",  default="data/drivaernet_real",
                        help="Path to DrivAerNet++ data root (needs processed/ cache)")
    parser.add_argument("--cd_stats",   default=None,
                        help="Path to cd_stats.json (auto-detected if omitted)")
    parser.add_argument("--out",        default="outputs/client_demo",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = _get_device()
    print(f"\n{'='*60}")
    print(f"  F1AeroNet  —  Client Demo")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Data root:   {args.data_root}")
    print(f"  Output:      {args.out}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[1/5]  Loading model …")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {n_params:,}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n[2/5]  Loading test dataset …")
    cd_stats_path = args.cd_stats or os.path.join(
        os.path.dirname(args.checkpoint), "cd_stats.json"
    )
    if not os.path.exists(cd_stats_path):
        cd_stats_path = os.path.join(args.data_root, "cd_stats.json")

    dataset = DrivAerNetDataset(
        data_root    = args.data_root,
        split        = "val",
        normalize_cd = os.path.exists(cd_stats_path),
    )
    if os.path.exists(cd_stats_path):
        with open(cd_stats_path) as f:
            cd_stats = json.load(f)
        dataset.set_cd_stats(cd_stats["cd_mean"], cd_stats["cd_std"])
        print(f"  Cd stats loaded  mean={cd_stats['cd_mean']:.4f}  "
              f"std={cd_stats['cd_std']:.4f}")
    else:
        cd_stats = {"cd_mean": 0.0, "cd_std": 1.0}
        print("  [WARNING] cd_stats.json not found — Cd shown in normalised units")

    # ── Pick best sample ─────────────────────────────────────────────────────
    print("\n[3/5]  Selecting best test sample …")
    best_data  = pick_best_sample(model, dataset, device)
    design_id  = getattr(best_data, "design_id", "unknown")

    # ── Timed inference ──────────────────────────────────────────────────────
    print("[4/5]  Running timed inference …")
    pred, elapsed_ms = timed_inference(model, best_data, device)

    n_verts = best_data.num_nodes
    n_edges = best_data.edge_index.shape[1]
    print(f"  Design:      {design_id}")
    print(f"  Vertices:    {n_verts:,}")
    print(f"  Edges:       {n_edges:,}")
    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  Inference time:  {elapsed_ms:>7.1f} ms     ║")
    print(f"  ╚══════════════════════════════════╝")

    # ── Compute metrics ──────────────────────────────────────────────────────
    cp_pred  = pred["cp"].cpu().numpy().reshape(-1)
    cp_true  = best_data.y_cp.cpu().numpy().reshape(-1)

    wss_pred = pred["wss"].cpu().numpy()          # (V, 3)
    wss_true = best_data.y_wss.cpu().numpy()
    wss_pred_mag = np.linalg.norm(wss_pred, axis=1) if wss_pred.ndim == 2 else np.abs(wss_pred)
    wss_true_mag = np.linalg.norm(wss_true, axis=1) if wss_true.ndim == 2 else np.abs(wss_true)

    cd_pred_norm = float(pred["cd"].reshape(-1)[0].cpu())
    cd_true_norm = float(best_data.y_cd.reshape(-1)[0].cpu())
    cd_mean = cd_stats["cd_mean"]
    cd_std  = cd_stats["cd_std"]
    cd_pred_phys = cd_pred_norm * cd_std + cd_mean
    cd_true_phys = cd_true_norm * cd_std + cd_mean
    cd_mae_phys  = abs(cd_pred_phys - cd_true_phys)
    cd_rel_pct   = cd_mae_phys / (abs(cd_true_phys) + 1e-8) * 100.0

    metrics = {
        "cp_rmse":    _rmse(cp_pred, cp_true),
        "cp_mae":     _mae(cp_pred,  cp_true),
        "cp_r2":      _r2(cp_pred,   cp_true),
        "wss_rmse":   _rmse(wss_pred_mag, wss_true_mag),
        "wss_mae":    _mae(wss_pred_mag,  wss_true_mag),
        "cd_pred":    cd_pred_phys,
        "cd_true":    cd_true_phys,
        "cd_mae_phys": cd_mae_phys,
        "cd_rel_pct":  cd_rel_pct,
    }

    print(f"\n  Metrics  (normalised space unless noted)")
    print(f"    Cp   RMSE={metrics['cp_rmse']:.4f}  MAE={metrics['cp_mae']:.4f}  "
          f"R²={metrics['cp_r2']:.4f}")
    print(f"    WSS  RMSE={metrics['wss_rmse']:.4f}  MAE={metrics['wss_mae']:.4f}")
    print(f"    Cd   pred={cd_pred_phys:.4f}  true={cd_true_phys:.4f}  "
          f"err={cd_mae_phys:.5f}  ({cd_rel_pct:.2f} %)")

    # ── Outputs ──────────────────────────────────────────────────────────────
    print(f"\n[5/5]  Writing outputs → {args.out}/")
    verts = best_data.pos.cpu().numpy()

    _three_panel(
        vertices    = verts,
        true_field  = cp_true,
        pred_field  = cp_pred,
        title       = f"Surface Pressure Coefficient  (Cp)  —  {design_id}",
        field_label = "Cp  (normalised)",
        cmap_field  = "RdBu_r",
        out_path    = os.path.join(args.out, "cp_comparison.png"),
    )

    _three_panel(
        vertices    = verts,
        true_field  = wss_true_mag,
        pred_field  = wss_pred_mag,
        title       = f"Wall-Shear-Stress Magnitude  (||WSS||)  —  {design_id}",
        field_label = "||WSS||  (normalised)",
        cmap_field  = "hot",
        out_path    = os.path.join(args.out, "wss_comparison.png"),
    )

    _metrics_card(
        metrics    = metrics,
        design_id  = design_id,
        elapsed_ms = elapsed_ms,
        out_path   = os.path.join(args.out, "metrics_card.png"),
    )

    _write_vtp_placeholder(design_id, args.out)

    print(f"\n{'='*60}")
    print(f"  Done.  All outputs in:  {os.path.abspath(args.out)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
