#!/usr/bin/env python3
"""
gen_train_cd_scatter.py  —  Run inference on training split and emit
(cd_true, cd_pred) pairs as JSON for the demo scatter chart.
"""
import json, os, sys
import torch
from torch_geometric.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.f1_net import F1AeroNet
from data.drivaernet_dataset import DrivAerNetDataset

CKPT       = "runs/new_final_run/best.pt"
DATA_ROOT  = "data/drivaernet_real"
OUT_JSON   = "viz/train_cd_scatter.json"

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    model = F1AeroNet.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt['epoch']}")

    with open("runs/new_final_run/cd_stats.json") as f:
        cd_stats = json.load(f)
    cd_mean = cd_stats["cd_mean"]
    cd_std  = cd_stats["cd_std"]

    dataset = DrivAerNetDataset(
        data_root    = DATA_ROOT,
        split        = "train",
        normalize_cd = True,
    )
    dataset.set_cd_stats(cd_mean, cd_std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Training samples: {len(dataset)}")

    cd_true_list, cd_pred_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out   = model(
                x            = batch.x,
                edge_index   = batch.edge_index,
                angles       = batch.edge_angles,
                transporters = batch.edge_transporters,
                batch        = batch.batch,
            )
            cd_pred_norm = float(out["cd"].squeeze())
            cd_pred_phys = cd_pred_norm * cd_std + cd_mean

            cd_true_norm = float(batch.y_cd.squeeze())
            cd_true_phys = cd_true_norm * cd_std + cd_mean

            cd_true_list.append(round(cd_true_phys, 6))
            cd_pred_list.append(round(cd_pred_phys, 6))

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(dataset)} done")

    result = {"cd_true": cd_true_list, "cd_pred": cd_pred_list, "n": len(cd_true_list)}
    os.makedirs("viz", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f)
    print(f"\nSaved {len(cd_true_list)} pairs → {OUT_JSON}")

    import numpy as np
    t = np.array(cd_true_list)
    p = np.array(cd_pred_list)
    mae = float(np.mean(np.abs(t - p)))
    print(f"Train Cd MAE: {mae:.4f}  ({mae/t.mean()*100:.2f}%)")

if __name__ == "__main__":
    main()
