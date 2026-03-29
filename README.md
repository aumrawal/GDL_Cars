# F1 Aero GEM-CNN

Predicts CFD outputs (surface pressure Cp, wall shear stress WSS, drag Cd, downforce Cl)
on F1 car meshes using Gauge Equivariant Mesh CNNs — replacing Navier-Stokes solvers
for rapid aerodynamic evaluation.

## Install

```bash
# macOS M4 — MPS-accelerated
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install pyvista vtk numpy scipy pyyaml tqdm matplotlib
```

## Project Structure

```
f1_aero_gem/
├── data/
│   ├── drivaernet_dataset.py   # VTP parsing + CFD field extraction
│   ├── mesh_geometry.py        # Tangent frames, log map, parallel transporters
│   └── transforms.py           # Normalisation + augmentation
├── models/
│   ├── irreps.py               # SO(2) irrep basis kernel builder (Table 1, GEM paper)
│   ├── gem_conv.py             # GEM-CNN layer: anisotropic gauge-equivariant kernels
│   └── f1_net.py               # Full network with Cp / WSS / Cd / Cl heads
├── train/
│   ├── losses.py               # MSE + L1 losses, gradient-weighted surface loss
│   └── trainer.py              # Training loop, LR scheduling, checkpointing
├── eval/
│   ├── evaluator.py            # Per-vertex errors, Cd/Cl force integration
│   └── visualize.py            # VTK/ParaView output
└── configs/
    └── f1_base.yaml            # All hyperparameters
```

## Quick Start

```bash
# 1. Download DrivAerNet++ dataset (see https://github.com/Mohamedelrefaie/DrivAerNet)
# 2. Edit configs/f1_base.yaml — set data_root to your dataset path
# 3. Train
python -m train.trainer --config configs/f1_base.yaml

# 4. Evaluate
python -m eval.evaluator --config configs/f1_base.yaml --checkpoint runs/best.pt

# 5. Visualise in ParaView
python -m eval.visualize --config configs/f1_base.yaml --checkpoint runs/best.pt --out output.vtp
```

## Key Design Decisions

### Why GEM-CNN over plain GCN?
Standard GCNs apply *isotropic* kernels — they cannot distinguish a front-wing leading
edge from a trailing edge because both look like "a vertex with neighbours". GEM-CNN uses
*anisotropic* kernels that depend on the angular position θ_pq of each neighbour in the
local tangent plane, making the model sensitive to flow direction and surface curvature.

### Feature types (SO(2) irreps)
- **ρ₀** — scalars (pressure Cp, speed magnitude): invariant under gauge change
- **ρ₁** — tangent vectors (WSS τ, surface normals): rotate with gauge
- **ρ₂** — 2nd-order tensors (strain rate at surface): rotate at 2× frequency

Intermediate layers use mixed types e.g. `16ρ₀ ⊕ 16ρ₁ ⊕ 8ρ₂` to capture all
geometric information relevant to aerodynamic loading.

### Parallel transport
When accumulating neighbour features across curved surfaces (e.g. the tight radius of
an F1 floor edge), raw feature vectors expressed in different local frames cannot simply
be summed. The discrete Levi-Civita connection g_{q→p} rotates neighbour features into
the target frame before aggregation — critical for correct WSS vector prediction.
