# CHANGES — F1AeroNet Improvements

All changes target the Kaggle T4 GPU (16 GB VRAM) training run on DrivAerNet++ (300 meshes).

---

## Fix A — Cd Mean Collapse: Z-score Normalisation

**Files:** `data/drivaernet_dataset.py`, `train/trainer.py`

**Problem:** The model predicted ~0.24 for every sample (the dataset mean). Root cause: raw Cd values sit in [0.20, 0.29] (range ≈ 0.09), so the unscaled MSE loss was ~0.001 — roughly 0.7% of the total gradient signal. The network learned to ignore Cd entirely and predict the mean.

**Fix:** Added `compute_cd_stats_from_cache()` in the dataset to compute training-split mean and std. The trainer calls this after building the train dataset, saves results to `cd_stats.json`, and injects them into all splits via `set_cd_stats()`. The dataset then z-scores `y_cd` on-the-fly at load time, giving targets with std ≈ 1 — the same scale as Cp and WSS.

**Why z-score and not min-max:** Z-score is robust to outliers and makes the loss scale comparable across all three targets without requiring knowledge of the physical range in advance.

---

## Fix B — Cd Mean Collapse: Loss Type MSE Instead of L1

**Files:** `configs/f1_base.yaml`, `F1_Training_Kaggle.ipynb` (cell-05)

**Problem:** L1 loss has a constant gradient of ±1 regardless of error magnitude. When predictions are stuck near the mean (small errors), L1 provides no stronger gradient than when they are wildly off — it does not push the model out of the collapsed state.

**Fix:** Changed `loss_type.cd` from `l1` to `mse`. MSE gradient is `2 × error`, so larger errors receive proportionally stronger feedback, which breaks the mean-collapse attractor.

---

## Fix C — Cd Mean Collapse: Equal Loss Weight

**Files:** `configs/f1_base.yaml`, `train/losses.py`, `F1_Training_Kaggle.ipynb` (cell-05)

**Problem:** `loss_weight.cd` was set to 0.1 in earlier experiments and then briefly inflated to 5.0. The 0.1 weight made Cd contribute less than 1% of the total gradient. The 5.0 weight over-penalised Cd at the expense of Cp and WSS.

**Fix:** After Fix A (z-score normalisation), all three targets have loss magnitudes on the same order. Set `loss_weight.cd = 1.0` (equal to Cp and WSS). No output-head dominates gradient flow.

---

## Fix 1 — Per-Component WSS Normalisation

**Files:** `data/drivaernet_dataset.py`

**Problem:** Wall shear stress was normalised with a single scalar mean and std computed over all V×3 elements. The streamwise (x) component is typically 10–100× larger than the lateral (y) and vertical (z) components, so a single scalar std inflated the small components and compressed the large one. The network saw a distorted target distribution.

**Fix:** Changed `wss_new.mean()` and `wss_new.std()` to `mean(axis=0)` and `std(axis=0)`, producing per-component statistics of shape `(3,)`. Each component is now independently z-scored to zero mean, unit variance. The `wss_sl_mean` and `wss_sl_std` attributes stored in the `Data` object are now `(3,)` tensors (needed for denormalisation at eval time).

---

## Fix 2 — Vertex Normals as Input Features

**Files:** `data/drivaernet_dataset.py`, `models/f1_net.py`, `configs/f1_base.yaml`

**Problem:** The input feature vector `x` had 6 channels: 3 normalised coordinates, 1 velocity ratio (U_inf/U_ref), and 2 positional fractions (x_frac, z_frac). The network had no direct access to local surface orientation, forcing it to infer which way each face points from neighbour geometry alone.

**Fix:** `precompute_geometry` already returns `normals` (unit outward vertex normals, shape V×3). Appended these to `x`, giving 9 input channels. Updated `in_channels` from 6 → 9 in `F1AeroNet.__init__`, `F1AeroNet.from_config`, `configs/f1_base.yaml`, and the Kaggle notebook config.

Surface orientation is critical for distinguishing high-pressure stagnation faces from low-pressure suction surfaces — providing it explicitly avoids the network having to learn geometry from message passing alone.

**Cache:** Feature format changed; `CACHE_VERSION` bumped from 1 → 2. Stale `.pt` files are auto-detected and regenerated on next load (see also notebook cell-06).

---

## Fix 3 — Global Context Injection

**Files:** `models/f1_net.py`

**Problem:** GEM-CNN is a local message-passing network. With 5 layers and typical mesh edge lengths of ~5–10 mm on a 4.6 m car, the receptive field covers roughly 3–5 cm. Global aerodynamic features (front/rear pressure balance, stagnation point location) require information from across the entire surface. As a result, tail and underbody pressure predictions failed because those vertices could not "see" the high-pressure nose.

**Fix:** After each GEM block, pool only the scalar (ρ₀) channels via `global_mean_pool` to produce a per-graph summary vector of shape `(B, s_dim)`. A zero-initialised linear projection (so it starts as a no-op) mixes this global context, which is then broadcast back and added to the scalar channels of every vertex.

Only scalar channels are pooled and injected. Pooling vector (ρ₁) or tensor (ρ₂) channels would break gauge equivariance because vectors and tensors depend on the local frame of each vertex — averaging across frames is not meaningful. Scalars are frame-independent and safe to pool.

Zero-initialising the projection weights ensures the injection starts as an identity passthrough, so the global context is learned gradually without disrupting early training.

---

## Fix 4 — Increased Network Capacity

**Files:** `models/f1_net.py`, `configs/f1_base.yaml`, `F1_Training_Kaggle.ipynb` (cell-05)

**Problem:** The original architecture `[(8,2),(8,2),(16,2),(32,1),(8,1)]` has a bottleneck at the final `(8,1)` layer — only 8 scalar channels flow into the prediction heads. This is insufficient to represent the complex spatial structure of surface Cp and WSS fields.

**Fix:** Updated default `layer_specs` to `[(16,2),(16,2),(24,2),(32,1),(16,1)]`. The intermediate layers are uniformly wider and the output layer doubles from 8 to 16 multiplicity. Peak GPU memory was verified to be safe on Kaggle T4 (16 GB): the largest intermediate tensor `K_neigh` reaches ~9.2 GB with E=200k edges, leaving headroom for activations and optimizer state.

---

## Fix 6 — Cosine Annealing Learning Rate Schedule

**Files:** `train/trainer.py`, `configs/f1_base.yaml`, `F1_Training_Kaggle.ipynb` (cell-05)

**Problem:** `ReduceLROnPlateau` (patience=15, factor=0.5) only decays the learning rate when validation loss stops improving. On a 300-mesh dataset, validation loss is noisy and the scheduler frequently stalls — keeping the learning rate high when it should be decaying, or decaying prematurely on a bad batch. It also has no mechanism to escape local minima.

**Fix:** Replaced with `CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)`. The learning rate follows a smooth cosine curve from `lr` down to `eta_min` over T_0 epochs, then restarts. Each successive cycle is twice as long (T_mult=2), providing a progressively finer exploration-exploitation tradeoff. Warm restarts are particularly effective for GNN training where loss landscapes are non-convex.

Config keys `lr_patience` and `lr_factor` have been removed; `T_0` and `T_mult` replace them.

---

## Cache Version Note

Any time `mesh_to_pyg_data` changes the feature format or normalisation (Fixes 1 and 2 in this batch), the `CACHE_VERSION` constant in `drivaernet_dataset.py` must be incremented. The `get()` method checks this version and auto-deletes and regenerates stale `.pt` files. Notebook cell-06 additionally performs an explicit sweep to remove stale files before starting the training run.
