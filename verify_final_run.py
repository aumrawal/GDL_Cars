#!/usr/bin/env python3
"""
verify_final_run.py  —  Evaluate runs/final_run_(cp_spcfc)/best.pt and compare
against the README baseline metrics.

Usage (from project root, f1_aero env):
    python verify_final_run.py
    python verify_final_run.py --data_root data/drivaernet_real
    python verify_final_run.py --checkpoint runs/final_run_\(cp_spcfc\)/best.pt
"""

import os
import sys
import argparse

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset
from eval.evaluator import evaluate

# ── README baseline ──────────────────────────────────────────────────────────
README_METRICS = {
    "cp_mae":       0.12142,
    "cp_rmse":      0.63755,
    "cp_r2":        0.0846,
    "wss_mae":      0.14923,
    "wss_rmse":     0.79070,
    "wss_angle_err": 17.12,
}

DEFAULT_CKPT      = "runs/final_run_(cp_spcfc)/best.pt"
DEFAULT_DATA_ROOT = "data/drivaernet_real"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(ckpt_path: str, device: torch.device) -> tuple[F1AeroNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {
        "in_channels": 9,
        "layer_types": [[16, 2], [16, 2], [16, 2], [24, 1], [16, 1]],
        "nonlin_samples": 5,
        "head_dropout": 0.1,
        "cd_head_dropout": 0.5,
        "break_symmetry_final": True,
    })
    model = F1AeroNet.from_config(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _print_comparison(metrics: dict, n_samples: int) -> None:
    readme = README_METRICS
    tol    = 0.02  # 2 % tolerance for pass/fail

    print(f"\n{'─'*62}")
    print(f"  Metric           Computed      README       Δ        Pass?")
    print(f"{'─'*62}")

    rows = [
        ("Cp  MAE",       "cp_mae",       metrics.get("cp_mae")),
        ("Cp  RMSE",      "cp_rmse",      metrics.get("cp_rmse")),
        ("Cp  R²",        "cp_r2",        metrics.get("cp_r2")),
        ("WSS MAE",       "wss_mae",      metrics.get("wss_mae")),
        ("WSS RMSE",      "wss_rmse",     metrics.get("wss_rmse")),
        ("WSS AngErr",    "wss_angle_err",metrics.get("wss_angle_err")),
    ]

    for label, key, val in rows:
        ref  = readme.get(key)
        if val is None or ref is None:
            print(f"  {label:<16}  {'N/A':>10}    {ref:>10}    {'—':>7}   skipped")
            continue
        delta = val - ref
        rel   = abs(delta) / (abs(ref) + 1e-10)
        mark  = "PASS" if rel <= tol else "----"
        print(f"  {label:<16}  {val:>10.5f}    {ref:>10.5f}    {delta:>+7.4f}   {mark}")

    print(f"{'─'*62}")

    if n_samples < 45:
        print(f"\n  [!] Only {n_samples}/45 test samples available locally.")
        print(f"      Full verification requires the complete DrivAerNet++ test set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_root",  default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.data_root):
        sys.exit(f"[ERROR] data_root not found: {args.data_root}")

    device = _get_device()
    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Device     : {device}")

    model, ckpt = _load_model(args.checkpoint, device)
    print(f"Loaded     : epoch {ckpt.get('epoch')}  "
          f"best_val_loss={ckpt.get('best_val', float('nan')):.6f}")

    # ── dataset ──────────────────────────────────────────────────────────────
    test_ds = DrivAerNetDataset(
        data_root    = args.data_root,
        split        = "test",
        max_vertices = 50000,
    )

    # Inject cd_stats if not found in data_root (model was trained on Kaggle)
    if test_ds.cd_mean == 0.0 and test_ds.cd_std == 1.0:
        cd_stats_candidates = [
            "runs/output_293/cd_stats.json",
            "runs/new_final_run/cd_stats.json",
        ]
        import json
        for p in cd_stats_candidates:
            if os.path.exists(p):
                with open(p) as f:
                    s = json.load(f)
                test_ds.set_cd_stats(float(s["cd_mean"]), float(s["cd_std"]))
                print(f"Cd stats   : loaded from {p}  "
                      f"(mean={s['cd_mean']:.4f}, std={s['cd_std']:.4f})")
                break
        else:
            print("[WARNING] cd_stats.json not found — Cd metrics will be in normalised space")
    n = len(test_ds)
    print(f"Test set   : {n} samples\n")

    if n == 0:
        sys.exit("[ERROR] No test samples found in data_root.")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate(model, loader, device, verbose=True)

    # ── compare vs README ────────────────────────────────────────────────────
    print("\n  Comparison against README baseline:")
    _print_comparison(metrics, n_samples=n)
    print()


if __name__ == "__main__":
    main()
