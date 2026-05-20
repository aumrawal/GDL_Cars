# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

F1AeroNet — a Gauge Equivariant Mesh CNN (GEM-CNN) that predicts aerodynamic surface fields (Cp, WSS, Cd) on 3D car meshes, replacing CFD solvers. Implements de Haan et al. 2020. Trained on DrivAerNet++ (200 cars). Runs on Kaggle T4 or locally on Apple M4 (MPS).

## Commands

```bash
# Train from scratch
python -m train.trainer --config configs/f1_base.yaml

# Resume training
python -m train.trainer --config configs/f1_base.yaml --resume runs_200/last.pt

# Evaluate on test set
python -m eval.evaluator --config configs/f1_base.yaml --checkpoint runs_200/best.pt

# Export VTP + Cp plots for one sample
python -m eval.visualize --config configs/f1_base.yaml --checkpoint runs_200/best.pt --out outputs/

# Prepare raw DrivAerNet++ VTP files (run once after download)
python prepare_data.py
```

**Before training locally**, update `data.data_root` in `configs/f1_base.yaml` — it defaults to `/kaggle/working/drivaernet_real`. If the mesh directory is missing the trainer automatically falls back to a synthetic dataset for pipeline testing.

**Kaggle training** uses `F1_Training_Kaggle.ipynb` with `configs/f1_kaggle.yaml` (written by cell-05 at runtime). The notebook owns its own inline training loop (cell-07) rather than calling `train()` from `trainer.py` — this is intentional so OOM guards and `empty_cache()` can be controlled per-batch.

## Architecture

### Data flow

```
Raw VTP (DrivAerNet++)
  → prepare_data.py  (merges p, wallShearStress, cd, cl into one VTP)
  → DrivAerNetDataset.get()
       - load_merged_vtp()     reads vertices, faces, pressure, WSS, Cd from VTP via PyVista
       - mesh_to_pyg_data()    builds PyG Data with 9 input features per vertex
       - precompute_geometry() computes GEM edge attributes (angles, transporters)
       - cached as .pt in data_root/processed/{split}/  (CACHE_VERSION=2 — bump on format change)
  → DataLoader  (batch_size=1, num_workers=0 for MPS/Kaggle)
  → F1AeroNet
```

### Input features (9 per vertex)

`[x, y, z, U_inf/100, x_frac, z_frac, nx, ny, nz]` where x_frac/z_frac are normalised streamwise/vertical positions and n is the area-weighted vertex normal.

### Model (`models/f1_net.py`)

```
input_embed  : Linear(9 → first_dim)
GEMBlock × 5 : equivariant conv + RegularNonlinearity
               + global context injection (pool scalars → broadcast back)
sym_break_proj: if break_symmetry_final=True, collapse to scalar channels only
Heads:
  cp_head  : ScalarHead → (V,)   per-vertex Cp
  wss_head : VectorHead → (V,3)  per-vertex WSS
  cd_head  : GlobalHead → (B,)   global Cd (mean-pool → MLP)
  cl_head  : None  (DrivAerNet WWS_WM subset has no Cl)
```

Each head's last Linear is weight-boosted ×10 at init to prevent mean collapse (see `_boost_last_layer`). Global context injection pools only scalar (ρ₀) channels to preserve gauge equivariance — pooling vector/tensor channels across frames is not meaningful.

### GEM internals (`models/gem_conv.py`, `models/irreps.py`, `data/mesh_geometry.py`)

Each GEM layer carries SO(2) irrep features (order 0 = scalars, order 1 = 2D vectors, order 2 = tensors). Before aggregation, neighbour features are parallel-transported into the target vertex's frame using the discrete Levi-Civita connection angle (`edge_transporters`). Neighbour angles (`edge_angles`) encode anisotropic kernel direction. Both are precomputed once in `data/mesh_geometry.py` and stored as edge attributes.

**OOM hotspot:** `EquivariantKernelBasis.eval_neigh()` in `models/irreps.py` allocates a dense `(E, dim_out, dim_in)` tensor per forward pass. At E=150k edges, a `(16,2)→(24,2)` block produces a ~1.72 GB tensor that can OOM a T4. The safe Kaggle architecture is `[[16,2],[16,2],[16,2],[24,1],[16,1]]` (set in `f1_kaggle.yaml`). `f1_base.yaml` keeps `[[16,2],[16,2],[24,2],[32,1],[16,1]]` for local use where memory is less constrained.

### Normalisation (critical — get this wrong and training diverges)

| Field | Pipeline |
|-------|----------|
| Cp    | raw pressure → dimensionless Cp → `symlog()` → z-score (mean/std stored per sample as `cp_sl_mean`, `cp_sl_std`) → ±5σ clip |
| WSS   | divide by `τ_ref = μ·U_inf/L_ref` → `symlog()` → **per-component** z-score (axis=0) → ±5σ clip |
| Cd    | z-score across training split; stats saved to `data_root/cd_stats.json`; auto-loaded by dataset |

`symlog(x) = sign(x) * log(1 + |x|)` — compresses heavy tails. Defined in `data/transforms.py`.

WSS uses per-component (axis=0) normalisation — a scalar std across all V×3 elements distorts the x/y/z components because streamwise shear is 10–100× larger than lateral.

### Training details

- Optimizer: AdamW, CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- Losses: Huber (δ=1.0) for Cp/WSS, MSE for Cd — all weights 1.0 (equal, since all targets are z-scored to std≈1)
- GEMBlock uses `torch.utils.checkpoint` during training — halves activation memory at ~30% speed cost
- OOM guard: batches with >150k edges (Kaggle) / >200k edges (local) are skipped
- Checkpoints: `last.pt` every epoch, `best.pt` on new val minimum, `checkpoint_epoch_NNNN.pt` every `save_every` epochs
- CSV log: `training_log.csv` in checkpoint dir
- Best run at 200 epochs lives in `runs_200/`

### Cd normalisation workflow

`cd_stats.json` must exist in `data_root` before training. To generate it:
1. First run of `DrivAerNetDataset(split='train', normalize_cd=False)` processes all VTPs into cache
2. `compute_cd_stats_from_cache()` scans the cache and saves mean/std to `cd_stats.json`
3. All subsequent dataset loads auto-apply z-score from this file

On Kaggle, run cell-06 once; the file is copied to `/kaggle/working/output/` by cell-08 and can be restored in future sessions by attaching the output dataset as input.

## Key Files

| File | Role |
|------|------|
| `configs/f1_base.yaml` | Local training config (wider architecture, paths to local data) |
| `configs/f1_kaggle.yaml` | Kaggle config — written at runtime by notebook cell-05 |
| `models/f1_net.py` | Top-level model, output heads, `from_config()` constructor |
| `models/gem_conv.py` | GEMConv layer + RegularNonlinearity + GEMBlock (with grad checkpoint) |
| `models/irreps.py` | SO(2) irrep math, `EquivariantKernelBasis`, `eval_neigh()` OOM hotspot |
| `data/mesh_geometry.py` | Discrete log map, parallel transporters, `precompute_geometry()` |
| `data/drivaernet_dataset.py` | Dataset, `mesh_to_pyg_data()`, `compute_cd_stats_from_cache()` |
| `train/trainer.py` | Training loop for local runs; exports `load_datasets`, `save_checkpoint`, `load_checkpoint` used by notebook |
| `train/losses.py` | `F1AeroLoss` — weighted sum of per-field losses |
| `eval/evaluator.py` | RMSE/MAE/R²/angular-error metrics, `evaluate()` |
| `CHANGES.md` | Full rationale for all architectural changes made to date |

## Environment Notes

- PyTorch Geometric required (`torch_geometric`, `torch_scatter` optional — pure-PyTorch fallback in `gem_conv.py`)
- PyVista + VTK required for VTP I/O
- `num_workers=0` everywhere — MPS and Kaggle do not support forked DataLoader workers
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` on Kaggle to reduce fragmentation from repeated large K_neigh allocations
