#!/usr/bin/env python3
"""
check_best_model.py  —  Find the best F1AeroNet checkpoint by Cp R².

Evaluates each candidate directory's best.pt on a single input file
(processed .pt cache or raw .vtp mesh) and prints a ranked table.

Usage (from project root):
    python check_best_model.py --vtp data/drivaernet_real/processed/val/F_S_WWS_WM_504.pt
    python check_best_model.py --vtp data/drivaernet_real/meshes/F_S_WWS_WM_504.vtp
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import load_merged_vtp, mesh_to_pyg_data


# ─────────────────────────────────────────────────────────────────────────────
# Candidate checkpoint directories  (edit this list as needed)
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = [
    "runs/new_final_run",
    "runs/output_293",
    "runs/phase_2_train",
    "runs/final_run_(cp_spcfc)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def _default_model_cfg() -> dict:
    return {
        "in_channels": 9,
        "layer_types": [[16, 2], [16, 2], [16, 2], [24, 1], [16, 1]],
        "max_freq": 2,
        "nonlin_samples": 5,
        "head_dropout": 0.1,
        "cd_head_dropout": 0.5,
        "break_symmetry_final": True,
    }


def load_model(ckpt_path: str, device: torch.device) -> F1AeroNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return _model_from_ckpt(ckpt, device)


def _model_from_ckpt(ckpt: dict, device: torch.device) -> F1AeroNet:
    if "model_cfg" in ckpt:
        model_cfg = ckpt["model_cfg"]
    elif "cfg" in ckpt:
        model_cfg = ckpt["cfg"].get("model", _default_model_cfg())
    else:
        model_cfg = _default_model_cfg()

    model = F1AeroNet.from_config(model_cfg).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def load_sample(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        n_feat = data.x.shape[1] if hasattr(data, "x") and data.x is not None else 0
        if n_feat != 9:
            raise SystemExit(
                f"[ERROR] Stale cache: expected 9 features, got {n_feat}.\n"
                f"        Delete and regenerate:  rm {path}"
            )
        return data

    if ext == ".vtp":
        raw = load_merged_vtp(path)
        design_id = os.path.splitext(os.path.basename(path))[0]
        return mesh_to_pyg_data(raw, design_id=design_id)

    raise SystemExit(f"[ERROR] Expected .pt or .vtp file, got: {path}")


@torch.no_grad()
def run_inference(model: F1AeroNet, data, device: torch.device) -> dict:
    def _forward(dev):
        d     = data.to(dev)
        batch = torch.zeros(d.num_nodes, dtype=torch.long, device=dev)
        return model.to(dev)(
            x            = d.x,
            edge_index   = d.edge_index,
            angles       = d.edge_angles,
            transporters = d.edge_transporters,
            batch        = batch,
        )

    try:
        return _forward(device)
    except RuntimeError as e:
        if device.type == "mps" and ("out of memory" in str(e).lower() or "MPS" in str(e)):
            print(f"  [OOM on MPS — retrying on CPU]", end="", flush=True)
            torch.mps.empty_cache()
            return _forward(torch.device("cpu"))
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find best checkpoint by Cp R²")
    parser.add_argument("--vtp", required=True,
                        help="Input sample: processed .pt cache or raw .vtp mesh")
    args = parser.parse_args()

    device = _get_device()
    print(f"\nDevice:  {device}")
    print(f"Sample:  {args.vtp}\n")

    data = load_sample(args.vtp)
    cp_true = data.y_cp.cpu().numpy().reshape(-1)
    design_id = getattr(data, "design_id", os.path.basename(args.vtp))
    print(f"Loaded:  {design_id}  ({data.num_nodes:,} vertices)\n")

    results = []

    for candidate_dir in CANDIDATES:
        ckpt_path = os.path.join(candidate_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP]  {ckpt_path}")
            continue

        print(f"  Loading  {ckpt_path} …", end="", flush=True)
        try:
            ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch")
            model = _model_from_ckpt(ckpt, device)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        print(f"  epoch={epoch}  running inference …", end="", flush=True)

        pred    = run_inference(model, data, device)
        cp_pred = pred["cp"].cpu().numpy().reshape(-1)
        r2      = _r2(cp_pred, cp_true)

        print(f"  Cp R²={r2:+.4f}")
        results.append((ckpt_path, r2))

    if not results:
        print("\n[ERROR] No valid checkpoints found in any candidate directory.")
        sys.exit(1)

    results.sort(key=lambda x: x[1], reverse=True)
    best_path, best_r2 = results[0]

    print(f"\n{'='*55}")
    print(f"  Ranked results (highest Cp R² first):")
    for rank, (path, r2) in enumerate(results, 1):
        marker = "  <-- BEST" if rank == 1 else ""
        print(f"    {rank}. {r2:+.4f}  {path}{marker}")
    print(f"\n  Winner:  {best_path}")
    print(f"  Cp R²:   {best_r2:+.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
