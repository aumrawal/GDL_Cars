#!/usr/bin/env python3
"""
compute_cd_error.py — Compute and save Cd error using runs_200/best.pt
"""

import os
import sys
import torch
import numpy as np
import csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset

CHECKPOINT  = 'runs_200/best.pt'
DATA_ROOT   = 'data/drivaernet_tiny'
SPLIT       = 'val'
OUT_CSV     = 'outputs/cd_errors.csv'
MAX_SAMPLES = None   # set to an int to limit


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_cfg = ckpt.get('cfg', ckpt.get('config', {}))
    model_cfg = saved_cfg.get('model')
    state_dict = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt
    if model_cfg is None:
        raise RuntimeError("No model config in checkpoint.")
    model = F1AeroNet.from_config(model_cfg)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict_cd(model, data, device):
    data = data.to(device)
    pred = model(
        x=data.x,
        edge_index=data.edge_index,
        angles=data.edge_angles,
        transporters=data.edge_transporters,
    )
    cd_pred = pred['cd'].cpu().item() if pred['cd'] is not None else None
    cd_true = data.y_cd.cpu().item() if hasattr(data, 'y_cd') and data.y_cd is not None else None
    return cd_pred, cd_true


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    model = load_model(CHECKPOINT, device)
    print(f"Loaded: {CHECKPOINT}")

    ds = DrivAerNetDataset(data_root=DATA_ROOT, split=SPLIT, force_reload=True)
    n = len(ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(ds))
    print(f"Evaluating {n}/{len(ds)} samples from '{SPLIT}' split\n")

    rows = []
    cd_errors = []

    for i in range(n):
        data = ds[i]
        did = data.design_id if hasattr(data, 'design_id') else f'{SPLIT}_{i}'
        cd_pred, cd_true = predict_cd(model, data, device)

        if cd_pred is None or cd_true is None:
            print(f"  [{i+1}/{n}] {did}  — Cd not available, skipping")
            continue

        err = abs(cd_pred - cd_true)
        rel_err = err / abs(cd_true) if cd_true != 0 else float('nan')
        rows.append({'design_id': did, 'cd_true': cd_true, 'cd_pred': cd_pred,
                     'cd_abs_error': err, 'cd_rel_error': rel_err})
        cd_errors.append(err)
        print(f"  [{i+1}/{n}] {did}  true={cd_true:.4f}  pred={cd_pred:.4f}  "
              f"abs_err={err:.4f}  rel_err={rel_err*100:.2f}%")

    if not rows:
        print("No Cd data found — check that y_cd is present in your dataset.")
        return

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    cd_errors = np.array(cd_errors)
    print(f"\n{'='*50}")
    print(f"Cd Error Summary  ({len(rows)} samples)")
    print(f"  MAE:    {cd_errors.mean():.6f}")
    print(f"  Std:    {cd_errors.std():.6f}")
    print(f"  Max:    {cd_errors.max():.6f}")
    print(f"  Min:    {cd_errors.min():.6f}")
    print(f"\n  Saved: {os.path.abspath(OUT_CSV)}")
    print('='*50)


if __name__ == '__main__':
    main()
