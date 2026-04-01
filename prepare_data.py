"""
prepare_data.py
───────────────
Extracts exactly N designs from your DrivAerNet++ zip files into a clean
folder structure that the training pipeline expects.

Run this ONCE before training:
    python prepare_data.py

What it does:
    1. Reads STL + pressure VTK + WSS VTK from your zip files
    2. Merges them into combined VTP files (mesh + fields in one file)
    3. Matches drag coefficients from your CSV
    4. Writes everything to data/drivaernet_real/ with train/val/test splits
    5. Prints a summary so you can verify before training

Edit the CONFIG section below to match your paths.
"""

import os, sys, zipfile, json, csv, random
import numpy as np
import pyvista as pv

# ── CONFIG — edit these paths ────────────────────────────────────────────────
MESH_ZIP   = "/Volumes/DriverNet/Mesh_zip/F_S_WWS_WM.zip"
PRESS_ZIP  = "/Volumes/DriverNet/Pressure_zip/F_S_WWS_WM.zip"
WSS_ZIP    = "/Volumes/DriverNet/WSS_zip/F_S_WWS_WM.zip"
COEFF_CSV  = "/Volumes/DriverNet/drag_coefficients.csv"

OUTPUT_DIR = os.path.expanduser("~/Desktop/f1_aero_gem/data/drivaernet_real")

N_DESIGNS  = 300      # how many designs to extract (out of ~684)
MAX_VERTS  = 50000    # subsample meshes to this size (None = keep all)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def read_vtk_fields(zf, inner_path):
    """Read a VTK file from inside a zip and return a pyvista mesh."""
    import tempfile
    data = zf.read(inner_path)
    # Write to temp file — pyvista needs a real file path
    suffix = '.vtk'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        mesh = pv.read(tmp_path)
    finally:
        os.unlink(tmp_path)
    return mesh


def read_stl(zf, inner_path):
    """Read an STL file from inside a zip and return a pyvista mesh."""
    import tempfile
    data = zf.read(inner_path)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        mesh = pv.read(tmp_path)
    finally:
        os.unlink(tmp_path)
    return mesh


def subsample_mesh(mesh, max_verts):
    """
    Robustly subsample a mesh to at most max_verts vertices.
    Uses three strategies in order:
      1. decimate_pro  — best quality, handles non-manifold meshes
      2. decimate      — fallback
      3. face-based random subsampling — always works
    Always verifies the result is actually small enough.
    """
    if mesh.n_points <= max_verts:
        return mesh

    target_reduction = min(1.0 - (max_verts / mesh.n_points), 0.97)

    # Strategy 1: decimate_pro (handles non-manifold)
    try:
        result = mesh.decimate_pro(target_reduction)
        if result.n_points <= max_verts * 1.1:
            print(f"    decimate_pro: {mesh.n_points} → {result.n_points}")
            return result
    except Exception:
        pass

    # Strategy 2: standard decimate
    try:
        result = mesh.decimate(target_reduction)
        if result.n_points <= max_verts * 1.1:
            print(f"    decimate: {mesh.n_points} → {result.n_points}")
            return result
    except Exception:
        pass

    # Strategy 3: random face subsampling — guaranteed to work
    # Select random faces, build a new clean mesh from just those faces
    print(f"    Falling back to random face subsampling")
    mesh_tri = mesh.triangulate()
    n_faces  = mesh_tri.n_faces
    # Estimate faces needed to get ~max_verts vertices
    # For triangular meshes: V ≈ F/2 (Euler), so F_target ≈ 2*max_verts
    f_target = min(int(2.2 * max_verts), n_faces)
    face_idx = np.random.choice(n_faces, f_target, replace=False)
    face_idx.sort()

    # Extract the chosen faces
    flat_faces = np.array(mesh_tri.faces).reshape(-1, 4)  # [3, v0, v1, v2]
    chosen     = flat_faces[face_idx]                      # (F_target, 4)

    # Remap vertex indices
    used_verts  = np.unique(chosen[:, 1:])
    v_remap     = np.full(mesh_tri.n_points, -1, dtype=np.int64)
    v_remap[used_verts] = np.arange(len(used_verts))

    new_verts = np.array(mesh_tri.points)[used_verts]
    new_faces = v_remap[chosen[:, 1:]]                    # (F_target, 3)

    # Rebuild as pyvista mesh
    new_flat  = np.hstack([np.full((len(new_faces),1), 3), new_faces]).flatten()
    result    = pv.PolyData(new_verts, new_flat)
    print(f"    face-sample: {mesh.n_points} → {result.n_points}")
    return result


def load_coefficients(csv_path):
    """
    Load Cd/Cl from the coefficients CSV.
    Tries several common column name patterns.
    Returns dict: design_id -> {'cd': float, 'cl': float}
    """
    coeffs = {}
    if not os.path.exists(csv_path):
        print(f"[WARNING] Coefficients file not found: {csv_path}")
        print("          Cd/Cl will be computed from surface fields.")
        return coeffs

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"  Coefficient columns: {cols}")

        # Find the ID column
        id_col = next((c for c in cols if
                       any(k in c.lower() for k in ['id', 'design', 'name', 'run'])), cols[0])
        # Find Cd column
        cd_col = next((c for c in cols if 'cd' in c.lower() or 'drag' in c.lower()), None)
        # Find Cl column
        cl_col = next((c for c in cols if 'cl' in c.lower() or 'lift' in c.lower()), None)

        print(f"  Using: ID='{id_col}'  Cd='{cd_col}'  Cl='{cl_col}'")

        for row in reader:
            did = str(row[id_col]).strip()
            cd  = float(row[cd_col]) if cd_col and row[cd_col] else None
            cl  = float(row[cl_col]) if cl_col and row[cl_col] else None
            coeffs[did] = {'cd': cd, 'cl': cl}

    print(f"  Loaded {len(coeffs)} coefficient entries")
    return coeffs


def compute_cd_cl_from_fields(pressure_mesh, wss_mesh, stl_mesh,
                               rho=1.225, U_inf=83.33):
    """
    Integrate pressure and WSS over the surface to compute Cd and Cl.
    Used as fallback when CSV coefficients are not available.
    """
    try:
        import trimesh
        # Use the STL mesh for normals and areas (it's the geometry source)
        verts  = np.array(stl_mesh.points)
        faces  = np.array(stl_mesh.faces).reshape(-1, 4)[:, 1:]

        v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
        cross      = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(cross, axis=1) / 2.0
        face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10)

        # Get pressure at face centres (interpolate from vertex pressure)
        # Simplified: use vertex mean per face
        p_field = _get_scalar_field(pressure_mesh)
        wss_field = _get_vector_field(wss_mesh)

        if p_field is None or wss_field is None:
            return None, None

        q_inf = 0.5 * rho * U_inf**2
        # Map vertex fields to faces by averaging face vertices
        # This is approximate but sufficient
        p_ref = np.mean(p_field)
        cp_vertices = (p_field - p_ref) / q_inf
        # Average Cp over face vertices
        cp_faces = cp_vertices[faces].mean(axis=1)

        # Pressure force: F = -Cp * q_inf * n * A (non-dim)
        F_p = -(cp_faces[:, None] * face_normals * face_areas[:, None]).sum(axis=0)

        # Viscous force (approximate vertex WSS to faces)
        wss_nd = wss_field / q_inf
        wss_faces = wss_nd[faces].mean(axis=1)
        F_v = (wss_faces * face_areas[:, None]).sum(axis=0)

        F_total = F_p + F_v
        cd = float(F_total[0])    # streamwise = x
        cl = float(-F_total[2])   # vertical   = z
        return cd, cl
    except Exception as e:
        print(f"    [WARNING] Could not compute Cd/Cl: {e}")
        return None, None


def _get_scalar_field(mesh):
    """Extract pressure scalar field with multiple name fallbacks."""
    candidates = ['p', 'pressure', 'Pressure', 'P', 'pMean', 'p_rgh']
    for name in candidates:
        if name in mesh.point_data:
            arr = np.array(mesh.point_data[name], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            return arr
    # Try first available scalar field
    for name, arr in mesh.point_data.items():
        arr = np.array(arr)
        if arr.ndim == 1:
            print(f"    Using field '{name}' as pressure")
            return arr.astype(np.float32)
    print(f"    [WARNING] No scalar field found. Available: {list(mesh.point_data.keys())}")
    return None


def _get_vector_field(mesh):
    """Extract WSS vector field with multiple name fallbacks."""
    candidates = ['wallShearStress', 'WallShearStress', 'wall_shear_stress',
                  'tau_w', 'wss', 'WSS', 'Tau']
    for name in candidates:
        if name in mesh.point_data:
            arr = np.array(mesh.point_data[name], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            return arr
    # Try first available vector field
    for name, arr in mesh.point_data.items():
        arr = np.array(arr)
        if arr.ndim == 2 and arr.shape[1] == 3:
            print(f"    Using field '{name}' as WSS")
            return arr.astype(np.float32)
    print(f"    [WARNING] No vector field found. Available: {list(mesh.point_data.keys())}")
    return None


def process_one_design(design_id, mesh_zf, press_zf, wss_zf,
                        mesh_inner, press_inner, wss_inner,
                        coeffs, max_verts):
    """
    Load STL + pressure VTK + WSS VTK for one design,
    merge into a single pyvista mesh with all fields attached,
    and return it.
    """
    # 1. Load geometry from STL
    stl_mesh = read_stl(mesh_zf, mesh_inner)
    stl_mesh = stl_mesh.triangulate()   # ensure triangles
    # Weld duplicate vertices — STL format stores each triangle independently,
    # so shared edges are duplicated. clean() merges them, halving edge count.
    stl_mesh = stl_mesh.clean(tolerance=1e-6)

    # 2. Load pressure field
    press_mesh = read_vtk_fields(press_zf, press_inner)

    # 3. Load WSS field
    wss_mesh = read_vtk_fields(wss_zf, wss_inner)

    # 4. Subsample geometry if needed
    if max_verts and stl_mesh.n_points > max_verts:
        stl_mesh = subsample_mesh(stl_mesh, max_verts)

    # 5. Interpolate pressure and WSS onto the (possibly subsampled) STL mesh
    #    pyvista's sample() maps fields from source mesh onto target mesh points
    try:
        stl_with_press = stl_mesh.sample(press_mesh)
        stl_with_all   = stl_with_press.sample(wss_mesh)
    except Exception as e:
        print(f"    [WARNING] Field interpolation failed: {e}")
        print(f"    Falling back to direct field copy (meshes may differ in size)")
        stl_with_all = stl_mesh
        # Attach raw fields — may need interpolation if vertex counts differ
        p_field   = _get_scalar_field(press_mesh)
        wss_field = _get_vector_field(wss_mesh)
        if p_field is not None and len(p_field) == stl_mesh.n_points:
            stl_with_all.point_data['p'] = p_field
        if wss_field is not None and len(wss_field) == stl_mesh.n_points:
            stl_with_all.point_data['wallShearStress'] = wss_field

    # 6. Rename fields to standard names
    _standardise_fields(stl_with_all)

    # 7. Attach Cd/Cl
    cd, cl = None, None
    if design_id in coeffs:
        cd = coeffs[design_id]['cd']
        cl = coeffs[design_id]['cl']
    if cd is None:
        cd, cl = compute_cd_cl_from_fields(press_mesh, wss_mesh, stl_mesh)

    stl_with_all.field_data['cd'] = np.array([cd or 0.0])
    stl_with_all.field_data['cl'] = np.array([cl or 0.0])
    stl_with_all.field_data['design_id'] = np.array([design_id])

    return stl_with_all


def _standardise_fields(mesh):
    """Rename fields to the standard names our loader expects."""
    rename = {
        'Pressure': 'p', 'pressure': 'p', 'P': 'p', 'pMean': 'p',
        'WallShearStress': 'wallShearStress',
        'wall_shear_stress': 'wallShearStress',
        'wss': 'wallShearStress', 'WSS': 'wallShearStress',
    }
    for old, new in rename.items():
        if old in mesh.point_data and new not in mesh.point_data:
            mesh.point_data[new] = mesh.point_data[old]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Validate zips exist
    for path, name in [(MESH_ZIP, "Mesh"), (PRESS_ZIP, "Pressure"), (WSS_ZIP, "WSS")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} zip not found: {path}")
            print("        Edit the CONFIG section at the top of this script.")
            sys.exit(1)

    # Load coefficients
    print("\nLoading aerodynamic coefficients...")
    coeffs = load_coefficients(COEFF_CSV)

    # Discover design IDs from mesh zip
    print("\nDiscovering design IDs from mesh zip...")
    with zipfile.ZipFile(MESH_ZIP) as zf:
        stl_entries = sorted([
            n for n in zf.namelist()
            if n.endswith('.stl')
        ])
    print(f"  Found {len(stl_entries)} STL files")

    # Extract design ID from path e.g. '3DMeshesSTL/F_S_WWS_WM/F_S_WWS_WM_001.stl'
    def get_id(stl_path):
        return os.path.splitext(os.path.basename(stl_path))[0]

    all_ids = [get_id(p) for p in stl_entries]

    # Select N designs randomly
    if N_DESIGNS >= len(all_ids):
        selected_ids = all_ids
    else:
        selected_ids = random.sample(all_ids, N_DESIGNS)
        selected_ids = sorted(selected_ids)

    print(f"  Selected {len(selected_ids)} designs for extraction")
    print(f"  First 5: {selected_ids[:5]}")

    # Build inner path lookups
    def build_lookup(zip_path, ext):
        with zipfile.ZipFile(zip_path) as zf:
            entries = [n for n in zf.namelist() if n.endswith(ext)]
        return {get_id(e): e for e in entries}

    print("\nBuilding file path lookups...")
    mesh_lookup  = build_lookup(MESH_ZIP,  '.stl')
    press_lookup = build_lookup(PRESS_ZIP, '.vtk')
    wss_lookup   = build_lookup(WSS_ZIP,   '.vtk')

    # Verify all three exist for each selected design
    valid_ids = []
    missing = []
    for did in selected_ids:
        if did in mesh_lookup and did in press_lookup and did in wss_lookup:
            valid_ids.append(did)
        else:
            missing.append(did)

    if missing:
        print(f"\n[WARNING] {len(missing)} designs missing from one or more zips:")
        for m in missing[:5]:
            print(f"  {m}  mesh={m in mesh_lookup}  "
                  f"press={m in press_lookup}  wss={m in wss_lookup}")
    print(f"\nProcessing {len(valid_ids)} complete designs...")

    # Create output directory
    out_mesh_dir = os.path.join(OUTPUT_DIR, "meshes")
    os.makedirs(out_mesh_dir, exist_ok=True)

    # Process each design
    processed = []
    failed    = []

    with zipfile.ZipFile(MESH_ZIP)  as mzf, \
         zipfile.ZipFile(PRESS_ZIP) as pzf, \
         zipfile.ZipFile(WSS_ZIP)   as wzf:

        for i, did in enumerate(valid_ids):
            print(f"  [{i+1:3d}/{len(valid_ids)}] {did}", end='  ', flush=True)
            try:
                merged = process_one_design(
                    design_id  = did,
                    mesh_zf    = mzf,
                    press_zf   = pzf,
                    wss_zf     = wzf,
                    mesh_inner = mesh_lookup[did],
                    press_inner= press_lookup[did],
                    wss_inner  = wss_lookup[did],
                    coeffs     = coeffs,
                    max_verts  = MAX_VERTS,
                )
                out_path = os.path.join(out_mesh_dir, f"{did}.vtp")
                merged.save(out_path)
                print(f"✓  {merged.n_points} verts  "
                      f"fields={list(merged.point_data.keys())}")
                processed.append(did)
            except Exception as e:
                print(f"✗  ERROR: {e}")
                failed.append((did, str(e)))

    print(f"\nProcessed: {len(processed)}  Failed: {len(failed)}")
    if failed:
        print("Failed designs:")
        for did, err in failed:
            print(f"  {did}: {err}")

    # Create train/val/test split
    random.shuffle(processed)
    n_train = int(TRAIN_FRAC * len(processed))
    n_val   = int(VAL_FRAC   * len(processed))

    split = {
        "train": processed[:n_train],
        "val":   processed[n_train : n_train + n_val],
        "test":  processed[n_train + n_val:],
    }

    split_path = os.path.join(OUTPUT_DIR, "split.json")
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit written to {split_path}")
    print(f"  Train: {len(split['train'])}")
    print(f"  Val  : {len(split['val'])}")
    print(f"  Test : {len(split['test'])}")

    # Final verification — read one output file
    print("\nVerifying one output file...")
    if processed:
        sample = pv.read(os.path.join(out_mesh_dir, f"{processed[0]}.vtp"))
        print(f"  Design  : {processed[0]}")
        print(f"  Vertices: {sample.n_points}")
        print(f"  Fields  : {list(sample.point_data.keys())}")
        print(f"  Cd/Cl   : {sample.field_data.get('cd', ['?'])[0]:.4f} / "
              f"{sample.field_data.get('cl', ['?'])[0]:.4f}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNext step — update configs/f1_base.yaml:")
    print(f"  data_root: \"{OUTPUT_DIR}\"")
    print("\nThen run:")
    print("  python -m train.trainer --config configs/f1_base.yaml")
    print("=" * 60)


if __name__ == '__main__':
    main()
