# train/trainer.py
"""
Training loop for F1AeroNet.

Features:
  - MPS (Apple Silicon M4) device support
  - Per-epoch validation with best-model checkpointing
  - ReduceLROnPlateau scheduling
  - Gradient clipping
  - Detailed console logging + optional CSV loss log
  - Resume from checkpoint

Usage:
    python -m train.trainer --config configs/f1_base.yaml
    python -m train.trainer --config configs/f1_base.yaml --resume runs/last.pt
"""

import os
import sys
import csv
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.f1_net import F1AeroNet
from train.losses import F1AeroLoss
from data.drivaernet_dataset import DrivAerNetDataset, make_synthetic_dataset


# ────────────────────────────────────────────────────────────────────────────
# Device selection (M4 MacBook — prefer MPS, fall back to CPU)
# ────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Select best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders) — M4 accelerated")
        return torch.device('mps')
    print("Using CPU")
    return torch.device('cpu')


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def load_datasets(cfg: dict):
    """
    Load train/val datasets from DrivAerNet++ or synthetic data.
    Returns (train_loader, val_loader).
    """
    data_cfg = cfg['data']
    data_root = data_cfg['data_root']

    # Check if real data exists — fall back to synthetic for development
    mesh_dir = os.path.join(data_root, 'meshes')
    use_synthetic = not os.path.exists(mesh_dir)

    if use_synthetic:
        print("=" * 60)
        print("  DrivAerNet++ data not found.")
        print(f"  Expected: {mesh_dir}")
        print("  Falling back to SYNTHETIC dataset for pipeline testing.")
        print("  Set data_root in configs/f1_base.yaml to use real data.")
        print("=" * 60)

        all_data = make_synthetic_dataset(
            n_meshes=32, n_vertices=300,
            U_inf=data_cfg.get('U_inf', 83.33),
            rho=data_cfg.get('rho', 1.225),
        )
        n_train = int(0.75 * len(all_data))
        train_data = all_data[:n_train]
        val_data   = all_data[n_train:]
    else:
        train_data = DrivAerNetDataset(
            data_root    = data_root,
            split        = 'train',
            max_vertices = data_cfg.get('max_vertices'),
            rho          = data_cfg.get('rho', 1.225),
            U_inf        = data_cfg.get('U_inf', 83.33),
        )
        val_data = DrivAerNetDataset(
            data_root    = data_root,
            split        = 'val',
            max_vertices = data_cfg.get('max_vertices'),
            rho          = data_cfg.get('rho', 1.225),
            U_inf        = data_cfg.get('U_inf', 83.33),
        )

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=0,   # 0 for MPS compatibility
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    print(f"Train: {len(train_data)} meshes | Val: {len(val_data)} meshes")
    return train_loader, val_loader


# ────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:      F1AeroNet,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  F1AeroLoss,
    device:     torch.device,
    grad_clip:  float = 1.0,
    log_every:  int = 10,
) -> dict:
    """Run one training epoch. Returns dict of mean losses."""
    model.train()
    total_loss = 0.0
    part_sums  = {'cp': 0.0, 'wss': 0.0, 'cd': 0.0, 'cl': 0.0}
    n_batches  = 0

    # Max edges allowed before OOM on 16 GB M4.
    # E ≈ 3V for triangular meshes; at 50k verts E ≈ 150k.
    # kernel tensor (E, C_out, C_in) at 64-ch layers ≈ 150k * 64 * 64 * 4B ≈ 2.4 GB — fine.
    # At 300k verts E ≈ 900k → 14 GB → OOM. Guard at 200k edges.
    MAX_EDGES = 180_000   # welded mesh at 50k verts gives ~150k edges

    for i, batch in enumerate(loader):
        # ── OOM guard: skip meshes that are too large ──────────────
        n_edges = batch.edge_index.shape[1]
        if n_edges > MAX_EDGES:
            n_verts = batch.x.shape[0]
            did = getattr(batch, 'design_id', '?')
            print(f"  [SKIP] {did}: {n_verts} verts / {n_edges} edges "
                  f"> {MAX_EDGES} limit — re-run prepare_data.py to fix")
            continue
        # ────────────────────────────────────────────────────────────

        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        pred = model(
            x            = batch.x,
            edge_index   = batch.edge_index,
            angles       = batch.edge_angles,
            transporters = batch.edge_transporters,
            batch        = batch.batch if hasattr(batch, 'batch') else None,
        )

        loss, parts = criterion(pred, batch)

        if torch.isnan(loss):
            print(f"  [WARNING] NaN loss at batch {i} — skipping")
            continue

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        for k, v in parts.items():
            part_sums[k] = part_sums.get(k, 0.0) + v.item()
        n_batches += 1

        if (i + 1) % log_every == 0:
            avg = total_loss / n_batches
            print(f"    batch {i+1:4d}/{len(loader)}  loss={avg:.5f}  "
                  + "  ".join(f"{k}={part_sums[k]/n_batches:.5f}"
                               for k in part_sums if part_sums[k] > 0))

    n = max(n_batches, 1)
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in part_sums.items()},
    }


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     F1AeroNet,
    loader:    DataLoader,
    criterion: F1AeroLoss,
    device:    torch.device,
) -> dict:
    """Run validation. Returns dict of mean losses."""
    model.eval()
    total_loss = 0.0
    part_sums  = {}
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(
            x            = batch.x,
            edge_index   = batch.edge_index,
            angles       = batch.edge_angles,
            transporters = batch.edge_transporters,
            batch        = batch.batch if hasattr(batch, 'batch') else None,
        )

        loss, parts = criterion(pred, batch)
        total_loss += loss.item()
        for k, v in parts.items():
            part_sums[k] = part_sums.get(k, 0.0) + v.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in part_sums.items()},
    }


# ────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    best_val: float, cfg: dict):
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val':  best_val,
        'cfg':       cfg,
    }, path)


def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['best_val']


# ────────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, resume: str = None):
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = load_datasets(cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    model = F1AeroNet.from_config(cfg['model']).to(device)
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k:20s}: {v:>10,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    train_cfg = cfg['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=train_cfg['lr_patience'],
        factor=train_cfg['lr_factor'],
        min_lr=train_cfg['min_lr'],
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = F1AeroLoss(
        weights    = train_cfg['loss_weights'],
        loss_types = train_cfg['loss_type'],
    ).to(device)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float('inf')
    if resume and os.path.exists(resume):
        start_epoch, best_val = load_checkpoint(resume, model, optimizer, scheduler)
        print(f"Resumed from {resume} (epoch {start_epoch}, best_val={best_val:.5f})")

    # ── Output dirs ───────────────────────────────────────────────────────
    ckpt_dir = train_cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV log
    log_path = os.path.join(ckpt_dir, 'training_log.csv')
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.DictWriter(log_file,
        fieldnames=['epoch', 'lr', 'train_total', 'train_cp', 'train_wss',
                    'train_cd', 'train_cl', 'val_total', 'val_cp', 'val_wss',
                    'val_cd', 'val_cl', 'time_s'])
    if start_epoch == 0:
        log_writer.writeheader()

    print(f"\nTraining for {train_cfg['epochs']} epochs on {device}\n")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, train_cfg['epochs'] + 1):
        t0 = time.time()

        print(f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=train_cfg['grad_clip'],
            log_every=train_cfg['log_every'],
        )
        val_losses = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        scheduler.step(val_losses['total'])

        # Console summary
        print(f"  TRAIN  total={train_losses['total']:.5f}  "
              + "  ".join(f"{k}={train_losses.get(k,0):.5f}"
                           for k in ['cp','wss','cd','cl']))
        print(f"  VAL    total={val_losses['total']:.5f}  "
              + "  ".join(f"{k}={val_losses.get(k,0):.5f}"
                           for k in ['cp','wss','cd','cl'])
              + f"  [{elapsed:.1f}s]")

        # CSV log
        log_writer.writerow({
            'epoch': epoch,
            'lr':    optimizer.param_groups[0]['lr'],
            'train_total': train_losses['total'],
            'train_cp':    train_losses.get('cp', 0),
            'train_wss':   train_losses.get('wss', 0),
            'train_cd':    train_losses.get('cd', 0),
            'train_cl':    train_losses.get('cl', 0),
            'val_total':   val_losses['total'],
            'val_cp':      val_losses.get('cp', 0),
            'val_wss':     val_losses.get('wss', 0),
            'val_cd':      val_losses.get('cd', 0),
            'val_cl':      val_losses.get('cl', 0),
            'time_s':      elapsed,
        })
        log_file.flush()

        # Checkpoints
        save_checkpoint(
            os.path.join(ckpt_dir, 'last.pt'),
            model, optimizer, scheduler, epoch, best_val, cfg,
        )
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            save_checkpoint(
                os.path.join(ckpt_dir, 'best.pt'),
                model, optimizer, scheduler, epoch, best_val, cfg,
            )
            print(f"  ✓ New best model saved (val={best_val:.5f})")

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {os.path.join(ckpt_dir, 'best.pt')}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train F1AeroNet')
    parser.add_argument('--config', default='configs/f1_base.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)
