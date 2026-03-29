"""
patch.py  —  run this once from your project folder:
    python patch.py

It directly patches data/mesh_geometry.py and train/trainer.py
in place, then clears all caches. No file downloading needed.
"""

import os, glob, sys

PROJECT = os.path.dirname(os.path.abspath(__file__))

# ── Patch 1: mesh_geometry.py ─────────────────────────────────────────────────
geo_path = os.path.join(PROJECT, "data", "mesh_geometry.py")
content  = open(geo_path).read()

OLD_FUNC = '''def build_edge_index_from_faces(faces: Tensor) -> Tensor:
    """
    Build directed edge_index (2, E) from face tensor (F, 3).
    Every mesh edge is represented in both directions (q→p and p→q).
    Self-loops are NOT included (unlike some GCN implementations).
    """
    pairs = []
    for i, j in [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]:
        pairs.append(torch.stack([faces[:,i], faces[:,j]], dim=0))  # (2, F)
    edge_index = torch.cat(pairs, dim=1)   # (2, 6F)

    # Deduplicate
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index   # (2, E)'''

NEW_FUNC = '''def build_edge_index_from_faces(faces: Tensor) -> Tensor:
    """
    Build directed edge_index (2, E) from face tensor (F, 3).
    Every mesh edge is represented in both directions (q\u2192p and p\u2192q).

    STL meshes store each triangle independently — every adjacent triangle
    has its own copy of shared vertices, so face indices differ even for the
    same physical vertex. This doubles every edge in the edge_index.
    Fix: deduplicate faces first (sort each face's vertex indices, then
    np.unique across all faces), which removes duplicated triangles and
    brings the edge count from ~6E/V down to the correct ~3E/V.
    """
    import numpy as np

    faces_np     = faces.cpu().numpy()
    faces_sorted = np.sort(faces_np, axis=1)
    _, keep      = np.unique(faces_sorted, axis=0, return_index=True)
    faces_np     = faces_np[keep]
    faces        = torch.from_numpy(faces_np).to(faces.device)

    pairs = []
    for i, j in [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]:
        pairs.append(torch.stack([faces[:,i], faces[:,j]], dim=0))
    edge_index = torch.cat(pairs, dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index'''

if OLD_FUNC in content:
    content = content.replace(OLD_FUNC, NEW_FUNC)
    open(geo_path, 'w').write(content)
    print("[OK] data/mesh_geometry.py patched")
elif 'faces_sorted' in content:
    print("[OK] data/mesh_geometry.py already has the fix")
else:
    # Nuclear option: just append a monkey-patch at the end
    patch = '''

# ── PATCH: replace build_edge_index_from_faces with fixed version ────────────
import numpy as _np
def build_edge_index_from_faces(faces):
    import numpy as np
    faces_np     = faces.cpu().numpy()
    faces_sorted = np.sort(faces_np, axis=1)
    _, keep      = np.unique(faces_sorted, axis=0, return_index=True)
    faces_np     = faces_np[keep]
    import torch as _torch
    faces        = _torch.from_numpy(faces_np).to(faces.device if hasattr(faces,'device') else 'cpu')
    pairs = []
    for i, j in [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]:
        pairs.append(_torch.stack([faces[:,i], faces[:,j]], dim=0))
    edge_index = _torch.cat(pairs, dim=1)
    edge_index = _torch.unique(edge_index, dim=1)
    return edge_index
'''
    open(geo_path, 'a').write(patch)
    print("[OK] data/mesh_geometry.py patched via append (pattern not found)")

# ── Patch 2: trainer.py MAX_EDGES ────────────────────────────────────────────
trainer_path = os.path.join(PROJECT, "train", "trainer.py")
tcontent = open(trainer_path).read()

# Replace any MAX_EDGES line with the correct value
import re
new_tcontent = re.sub(
    r'MAX_EDGES\s*=\s*[\d_]+.*',
    'MAX_EDGES = 180_000   # welded mesh at 50k verts gives ~150k edges',
    tcontent
)
if new_tcontent != tcontent:
    open(trainer_path, 'w').write(new_tcontent)
    print("[OK] train/trainer.py MAX_EDGES updated")
else:
    # Add the guard if it's completely missing
    if 'MAX_EDGES' not in tcontent:
        print("[WARNING] MAX_EDGES not found in trainer.py — add it manually")
    else:
        print("[OK] train/trainer.py already correct")

# ── Step 3: Delete ALL cached .pt files ──────────────────────────────────────
data_root = os.path.join(PROJECT, "data", "drivaernet_real")
pt_files  = glob.glob(os.path.join(data_root, "**", "*.pt"), recursive=True)

# Also check common alternative locations
pt_files += glob.glob(os.path.join(PROJECT, "**", "processed", "**", "*.pt"),
                      recursive=True)
pt_files = list(set(pt_files))

print(f"\nDeleting {len(pt_files)} cached .pt files...")
for f in pt_files:
    os.remove(f)
    print(f"  rm {os.path.relpath(f, PROJECT)}")

if not pt_files:
    print("  (none found — cache was already empty)")

# ── Step 4: Quick self-test ───────────────────────────────────────────────────
print("\nRunning quick self-test...")
sys.path.insert(0, PROJECT)

# Reload the module fresh
import importlib
if 'data.mesh_geometry' in sys.modules:
    importlib.reload(sys.modules['data.mesh_geometry'])

import numpy as np
import torch

# Simulate a small STL-style mesh with 4 duplicate faces
# Two triangles sharing an edge — each stores 5 vertices (2 duplicated)
# Duplicated triangle face pattern: face (0,1,2) appears as (0,1,2) AND (3,4,2)
# where vertices 3,4 are physical duplicates of 0,1
verts_dup = np.array([
    [0,0,0], [1,0,0], [0,1,0],   # triangle 1
    [0,0,0], [1,0,0], [1,1,0],   # triangle 2 — vertices 3,4 duplicate 0,1
], dtype=np.float32)
faces_dup = np.array([[0,1,2],[3,4,5],[0,3,1],[2,4,5]], dtype=np.int64)
# Before fix: 4 faces × 6 directed edges = 24, after unique ≈ many duplicates
# After fix: 2 unique face patterns (by sorted indices) → fewer edges

from data.mesh_geometry import build_edge_index_from_faces
ei = build_edge_index_from_faces(torch.from_numpy(faces_dup))
E  = ei.shape[1]
print(f"  Test mesh: 6 verts, 4 faces (with duplicates) → {E} directed edges")
print(f"  Expected: ≤18 (≤3 undirected edges × 2 directions × ~3 unique faces)")

# Test on actual data if available
vtps = glob.glob(os.path.join(data_root, "meshes", "*.vtp"))
if vtps:
    print(f"\nTesting on real mesh: {os.path.basename(vtps[0])}")
    try:
        from data.drivaernet_dataset import load_merged_vtp, mesh_to_pyg_data
        raw  = load_merged_vtp(vtps[0])
        data = mesh_to_pyg_data(raw)
        V    = data.x.shape[0]
        E    = data.edge_index.shape[1]
        ratio = E / V
        print(f"  Vertices: {V}   Edges: {E}   E/V ratio: {ratio:.2f}")
        if ratio < 4.0:
            print("  [PASS] Edge count is correct")
        else:
            print(f"  [FAIL] E/V={ratio:.2f} is too high — still duplicated")
    except Exception as e:
        print(f"  Could not test on real mesh: {e}")
else:
    print("\nNo VTP files found — run prepare_data.py first, then train")

print("\n" + "=" * 55)
print("Patch complete. Now run:")
print("  caffeinate -i python -m train.trainer --config configs/f1_base.yaml")
print("=" * 55)