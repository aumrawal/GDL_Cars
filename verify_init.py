#!/usr/bin/env python3
"""
verify_init.py — Run this BEFORE training to confirm the fixes work.

It constructs the model, feeds random input with std ≈ 1.0,
and prints prediction statistics. You want to see:
    Cp  std ≈ 0.5 - 2.0   (not 0.05!)
    WSS std ≈ 0.5 - 2.0   (not 0.19!)

Also tests symlog compression on example CFD-like data.

Usage:
    python verify_init.py
    # or on Kaggle:
    %run verify_init.py
"""

import torch
import numpy as np


def test_symlog():
    """Verify symlog compresses heavy tails."""
    from data.transforms import symlog, inv_symlog, normalise_fields

    print("=" * 60)
    print("TEST 1: Symlog tail compression")
    print("=" * 60)

    # Simulate heavy-tailed CFD data (like your Cp field)
    torch.manual_seed(42)
    n = 100_000
    # Mix of Gaussian + extreme outliers (mimics CFD)
    cp_raw = torch.randn(n) * 0.5
    cp_raw[0] = 27.0    # extreme positive (like your max)
    cp_raw[1] = -15.0   # extreme negative
    cp_raw[2] = 10.0

    wss_raw = torch.randn(n, 3) * 0.3
    wss_raw[0, 0] = 50.0   # extreme WSS
    wss_raw[1, 1] = -33.0

    # Flatten wss for normalise_fields (it expects 1D for simplicity)
    # We'll test with the magnitude
    wss_mag = wss_raw.norm(dim=1)

    print(f"\nBefore symlog:")
    print(f"  Cp  range: [{cp_raw.min():.1f}, {cp_raw.max():.1f}]  std={cp_raw.std():.3f}")
    print(f"  WSS range: [{wss_mag.min():.1f}, {wss_mag.max():.1f}]  std={wss_mag.std():.3f}")

    cp_sl = symlog(cp_raw)
    wss_sl = symlog(wss_mag)

    print(f"\nAfter symlog (before standardising):")
    print(f"  Cp  range: [{cp_sl.min():.2f}, {cp_sl.max():.2f}]  std={cp_sl.std():.3f}")
    print(f"  WSS range: [{wss_sl.min():.2f}, {wss_sl.max():.2f}]  std={wss_sl.std():.3f}")

    # Verify invertibility
    cp_back = inv_symlog(cp_sl)
    max_err = (cp_raw - cp_back).abs().max().item()
    print(f"\n  Symlog invertibility error: {max_err:.2e}  {'✓ OK' if max_err < 1e-5 else '✗ PROBLEM'}")

    print()


def test_head_init():
    """Verify model output std is now ~1.0, not ~0.05."""
    print("=" * 60)
    print("TEST 2: Head output scale at init")
    print("=" * 60)

    try:
        from models.f1_net import F1AeroNet
    except ImportError:
        print("  Cannot import F1AeroNet — skipping (run from project root)")
        return

    # Build model with the tiny layer specs from your config
    model = F1AeroNet(
        in_channels=4,
        layer_specs=[(8,2), (8,2), (16,2), (16,2), (8,1), (8,1)],
        N_nonlin=5,
        head_hidden=128,
        head_dropout=0.0,  # disable dropout for deterministic test
        break_symmetry_final=True,
    )
    model.eval()

    # Count parameters
    params = model.count_parameters()
    print(f"\n  Total parameters: {params['total']:,}")

    # Feed random input with std ≈ 1.0 (like normalised mesh data)
    torch.manual_seed(0)
    V = 5000
    x = torch.randn(V, 4)  # [xyz normalised, U_inf normalised]

    # Need edge_index, angles, transporters
    # Create a simple ring graph for testing
    src = torch.arange(V)
    dst = (src + 1) % V
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])
    E = edge_index.shape[1]
    angles = torch.rand(E) * 2 * 3.14159
    transporters = torch.rand(E) * 2 * 3.14159

    with torch.no_grad():
        pred = model(x, edge_index, angles, transporters)

    print(f"\n  Input x:         std={x.std():.4f}")
    print(f"\n  Prediction stats at init (WANT std ≈ 0.5 - 2.0):")
    print(f"    Cp  std={pred['cp'].std():.4f}  min={pred['cp'].min():.4f}  max={pred['cp'].max():.4f}")

    wss_std = pred['wss'].std()
    print(f"    WSS std={wss_std:.4f}  min={pred['wss'].min():.4f}  max={pred['wss'].max():.4f}")
    print(f"    Cd  val={pred['cd'].item():.4f}")
    print(f"    Cl  val={pred['cl'].item():.4f}")

    # Verdict
    cp_ok  = 0.2 < pred['cp'].std().item() < 5.0
    wss_ok = 0.2 < wss_std.item() < 5.0
    print(f"\n  Cp  scale: {'✓ GOOD' if cp_ok else '✗ TOO SMALL — head init not working'}")
    print(f"  WSS scale: {'✓ GOOD' if wss_ok else '✗ TOO SMALL — head init not working'}")
    print()


def test_huber_vs_mse():
    """Show how Huber protects against outlier gradients."""
    print("=" * 60)
    print("TEST 3: Huber vs MSE gradient comparison")
    print("=" * 60)

    import torch.nn.functional as Fu

    # Simulate: prediction=0, target has outlier at 27
    pred = torch.tensor([0.0], requires_grad=True)
    target = torch.tensor([27.0])

    # MSE gradient
    loss_mse = Fu.mse_loss(pred, target)
    loss_mse.backward()
    grad_mse = pred.grad.item()

    pred2 = torch.tensor([0.0], requires_grad=True)
    loss_hub = Fu.huber_loss(pred2, target, delta=1.0)
    loss_hub.backward()
    grad_hub = pred2.grad.item()

    print(f"\n  pred=0, target=27 (extreme Cp vertex):")
    print(f"    MSE   loss={loss_mse.item():.1f}   grad={grad_mse:.1f}")
    print(f"    Huber loss={loss_hub.item():.1f}  grad={grad_hub:.1f}")
    print(f"    Gradient ratio: MSE/Huber = {abs(grad_mse/grad_hub):.0f}×")
    print(f"\n  → Huber caps the gradient at ±delta, preventing outlier domination")
    print()


if __name__ == '__main__':
    test_symlog()
    test_head_init()
    test_huber_vs_mse()
    print("=" * 60)
    print("All checks done. If Cp/WSS std ≈ 0.5-2.0, you're ready to train.")
    print("Remember: re-init from scratch (don't resume old checkpoint).")
    print("Remember: clear PyG cache (rm -rf data/drivaernet_tiny/processed/)")
    print("=" * 60)