# CLAUDE.md — F1AeroNet Demo Redesign

## Project Overview

This project is **F1AeroNet** — a Gauge Equivariant Mesh CNN (GEM-CNN) that predicts
aerodynamic fields (Cp, WSS, Cd) on 3D car surfaces without running a CFD solver.
The goal is to redesign `demo.html` into a **business-facing portfolio showcase** that
communicates the value of the technology to non-technical stakeholders (investors,
automotive OEMs, racing teams, simulation software companies).

## What You Are Building

Replace the current `demo.html` (which asks users to upload a mesh and runs fake JS
inference) with a **polished, self-contained showcase** that:

1. Loads **three pre-computed VTP prediction files** (real GEM-CNN outputs)
2. Visualises the predicted aerodynamic fields (Cp, WSS magnitude) on the 3D car surface
3. Shows **business-relevant metrics** (Cd, Cl, speedup vs CFD, cost savings)
4. Includes **charts and graphs** (training loss curve, Cd scatter, field error bars)
5. Requires **zero user interaction** to look impressive — auto-plays through the cars
6. Is deployable as a single `demo.html` file on GitHub Pages (no backend)

## Input Data

You have **three VTP files** with the following per-vertex arrays:

| Array name in VTP | Meaning | Shape |
|---|---|---|
| `Points` | (x, y, z) vertex coordinates | (N, 3) |
| `Pressure` | Raw pressure in Pa | (N,) |
| `WallShearStress` | WSS vector in Pa | (N, 3) |
| `Cp` | Predicted pressure coefficient (dimensionless) | (N,) |
| `Cd_total` | Drag coefficient (scalar per mesh, in field_data) | scalar |
| `Cl_total` | Lift/downforce coefficient | scalar |

VTP files are XML-based. Parse them in JavaScript using the DOMParser and DataView
APIs — do NOT assume a backend is available. All parsing must happen client-side.

### VTP Parsing Strategy

VTP files store data in one of two formats:
- **ASCII**: values as space-separated text inside `<DataArray>` tags
- **Binary appended**: base64-encoded binary blobs

Use the following approach:
```javascript
// 1. Fetch the .vtp file as text
// 2. Parse XML with DOMParser
// 3. Find <PointData> → <DataArray Name="Cp"> for per-vertex fields
// 4. Find <FieldData> → <DataArray Name="Cd_total"> for scalars
// 5. Parse the text content as Float32Array
```

If the VTP files are binary/appended format, fall back to hardcoded JSON data
extracted from the files (see Fallback Data section below).

## File Structure

```
GDL_Cars/
├── demo.html          ← THE FILE TO REWRITE
├── assets/
│   ├── car1.vtp       ← Predicted outputs, car design 1
│   ├── car2.vtp       ← Predicted outputs, car design 2
│   └── car3.vtp       ← Predicted outputs, car design 3
└── CLAUDE.md          ← This file
```

## Design Direction

### Audience
Business executives, automotive engineers, F1 team technical directors,
investors in simulation/ML companies. They care about:
- **Speed**: How much faster than CFD?
- **Cost**: What does this save in compute?
- **Accuracy**: Is it close enough to be useful?
- **Generalisability**: Does it work on new unseen car shapes?

### Aesthetic
**Premium dark-mode dashboard** — think Bloomberg Terminal meets Formula 1 data
wall. Not a toy, not a research demo. A product.

- Dark background (#0a0e17 or similar deep navy/charcoal)
- Sharp accent colour: electric cyan (#00e5ff) or racing red (#ff1e00) or gold (#f5c518)
- Typography: a strong geometric sans (Bebas Neue or Orbitron for headers,
  DM Sans or Syne for body) — NOT Inter
- Subtle carbon-fibre texture or diagonal grid pattern in backgrounds
- Smooth transitions between car selections
- Numbers that count up on load (Cd, speedup factor, cost savings)

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: F1AeroNet logo | tagline | "Powered by GEM-CNN"   │
├──────────────────┬──────────────────────────────────────────┤
│                  │                                          │
│  3D VIEWPORT     │   METRICS PANEL                         │
│  (Three.js mesh  │   ┌─────────┬─────────┬─────────┐      │
│   with Cp        │   │  Cd     │  Cl     │Speedup  │      │
│   colourmap)     │   │ 0.312   │ -2.41   │ 4800×   │      │
│                  │   └─────────┴─────────┴─────────┘      │
│  [Car 1][Car 2]  │                                          │
│  [Car 3] tabs    │   FIELD SELECTOR: [Cp] [WSS Mag]        │
│                  │                                          │
│                  │   CHARTS:                               │
│                  │   • Cp distribution histogram           │
│                  │   • Predicted vs CFD Cd scatter         │
│                  │   • Training loss curve                 │
├──────────────────┴──────────────────────────────────────────┤
│  BUSINESS VALUE BAR: "Sub-second inference · $0.003/query  │
│  · 200 training cars · Gauge-equivariant by construction"  │
└─────────────────────────────────────────────────────────────┘
```

## Charts to Include

### 1. Cp Distribution Histogram
- X-axis: Cp value (−2 to +1.5)
- Y-axis: Vertex count
- Show distribution for the currently selected car
- Use a gradient fill matching the Cp colourmap (blue→white→red)
- Library: Chart.js (CDN)

### 2. Predicted vs CFD Cd Scatter
- Show 20 data points (use the hardcoded values below from the actual val set)
- X-axis: CFD Cd (ground truth)
- Y-axis: Predicted Cd
- Perfect prediction line (y=x dashed)
- Highlight the 3 demo cars
- Show MAE annotation

### 3. Training Loss Curve
- X-axis: Epoch (0–50)
- Y-axis: Validation loss
- Show dramatic drop from ~2.0 → ~0.19
- Mark the best checkpoint

### 4. Colourbar Legend
- Vertical gradient bar (blue→white→red for Cp, black→red→yellow for WSS)
- Min/max values updating per car

## Hardcoded Fallback Data

If VTP parsing fails or files are not present, use this hardcoded data extracted
from actual model outputs (replace with real values after running inference):

```javascript
const DEMO_CARS = [
  {
    id: 'car1',
    name: 'Fastback Sedan — Design 001',
    config: 'F_S_WWS_WM',
    cd: 0.312,
    cl: -2.41,
    cpMin: -1.73,
    cpMax: 0.98,
    wssMax: 1.24,
    inferenceMs: 180,
    cfgSimHours: 14.2,
    // Per-vertex Cp/WSS arrays loaded from VTP
    // Fallback: generate synthetic data matching these stats
  },
  {
    id: 'car2',
    name: 'Estate Wagon — Design 047',
    config: 'F_S_WWS_WM',
    cd: 0.341,
    cl: -1.87,
    cpMin: -1.51,
    cpMax: 1.02,
    wssMax: 1.09,
    inferenceMs: 195,
    cfgSimHours: 15.8,
  },
  {
    id: 'car3',
    name: 'Notchback Coupe — Design 089',
    config: 'F_S_WWS_WM',
    cd: 0.289,
    cl: -2.68,
    cpMin: -1.89,
    cpMax: 0.91,
    wssMax: 1.41,
    inferenceMs: 172,
    cfgSimHours: 13.5,
  },
];
```

## Cd Scatter Data (real validation set values)

Use these 20 real (Cd_CFD, Cd_predicted) pairs for the scatter chart:

```javascript
const CD_SCATTER = [
  [0.289, 0.301], [0.312, 0.318], [0.341, 0.335], [0.298, 0.291],
  [0.325, 0.330], [0.278, 0.285], [0.356, 0.348], [0.301, 0.308],
  [0.334, 0.327], [0.267, 0.272], [0.318, 0.324], [0.345, 0.339],
  [0.292, 0.299], [0.361, 0.354], [0.279, 0.283], [0.308, 0.315],
  [0.349, 0.343], [0.285, 0.290], [0.322, 0.317], [0.337, 0.331],
];
// MAE = 0.00619 (replace with actual computed value)
```

## Training Loss Data

```javascript
const LOSS_CURVE = {
  epochs: [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50],
  valLoss: [2.01,1.74,1.43,1.12,0.89,0.72,0.58,0.47,0.39,0.33,0.28,0.25,0.23,0.21,
            0.205,0.200,0.197,0.194,0.192,0.191,0.190,0.190,0.189,0.189,0.189,0.189],
};
```

## Business Value Metrics to Show

Compute and display dynamically:

| Metric | Value | How to compute |
|---|---|---|
| Inference speedup | ~4800× | CFD hours / inference seconds |
| Cost per query | ~$0.003 | Kaggle T4 cost × inference time |
| Training dataset | 200 cars | Hardcoded |
| Parameters | 138k | Hardcoded |
| Gauge invariance | 100% | Always exact by construction |
| Cp MAE | 0.121 | From actual results |
| Cd relative error | 6.87% | From actual results |

## Technical Constraints

- **Single file**: Everything in `demo.html` — no separate JS/CSS files
- **CDN only**: Three.js r128, Chart.js 4.x, Google Fonts — all from CDN
- **No build step**: Must work by opening `demo.html` directly or via GitHub Pages
- **No backend**: All data either parsed from VTP or hardcoded as JS constants
- **Performance**: Must render smoothly on a MacBook — keep vertex count ≤ 50k per mesh
  (subsample if VTP has more vertices)

## Colourmap Specification

### Cp (diverging blue → white → red)
```
t=0.0 → rgb(0, 32, 255)   — most negative Cp (suction)
t=0.5 → rgb(255, 255, 255) — Cp = 0
t=1.0 → rgb(255, 20, 20)  — most positive Cp (stagnation)
```

### WSS Magnitude (sequential heat)
```
t=0.0 → rgb(5, 5, 30)     — zero shear
t=0.33 → rgb(120, 0, 180)
t=0.66 → rgb(255, 80, 0)
t=1.0 → rgb(255, 240, 50) — peak shear
```

## Mesh Rendering Notes

- Use `THREE.BufferGeometry` with per-vertex colour attribute
- Material: `THREE.MeshPhongMaterial({ vertexColors: true, shininess: 60 })`
- Add subtle edge highlight with a second wireframe pass at low opacity
- If VTP parsing works: use real triangle connectivity from `<Polys>`
- If fallback: generate a parametric car silhouette geometry in JS
  (elongated ellipsoid with roof bump and wing surfaces — see existing demo.html
  `generateDemoMesh()` function for reference)

## Key Messages for Business Audience

Prominently display these statements somewhere on the page:

1. **"Sub-second aerodynamic prediction — replacing 14-hour CFD simulations"**
2. **"Gauge-equivariant by construction — predictions are rotationally consistent,
   not learned"**
3. **"Trained on 200 car configurations · Generalises to unseen geometries"**
4. **"138,000 parameters · Runs on a single GPU · $0.003 per car evaluation"**

## What NOT To Do

- Do not show a file upload button (remove it entirely)
- Do not show fake progress bars pretending to run inference
- Do not use Inter or Roboto fonts
- Do not use purple gradients on white backgrounds
- Do not make it look like a research notebook or Jupyter output
- Do not hide the fact that predictions are pre-computed — label them
  "Pre-computed GEM-CNN predictions on held-out test set"

## Success Criteria

A business executive with no ML background should look at this page and understand:
1. What problem this solves (CFD is slow and expensive)
2. How fast the alternative is (sub-second)
3. That the predictions look physically reasonable (colour fields on 3D car)
4. That someone serious built this (professional design, real metrics)

A technical reviewer (F1 CFD engineer, NVIDIA Modulus team) should see:
1. Correct Cp sign convention (stagnation = positive, suction = negative)
2. Gauge-equivariance mentioned and explained briefly
3. Honest error metrics (Cp R² = 0.08 is shown, not hidden)
4. Real training details (DrivAerNet++, 200 cars, Kaggle T4)
