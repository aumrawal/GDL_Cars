# data/drivaernet_dataset.py
"""
DrivAerNet++ dataset loader — updated for actual file format:
  - Merged VTP files (produced by prepare_data.py)
  - Fields: 'p' (pressure scalar), 'wallShearStress' (vector)
  - Cd/Cl in field_data per mesh
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import Optional, List

from data.mesh_geometry import (
    precompute_geometry,
    build_edge_index_from_faces,
)
from data.transforms import normalise_mesh, symlog

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARNING] pyvista not found. pip install pyvista vtk")


def load_merged_vtp(filepath: str) -> dict:
    if not HAS_PYVISTA:
        raise RuntimeError("pip install pyvista vtk")
    mesh = pv.read(filepath)
    mesh_tri = mesh.triangulate()

    # Weld duplicate vertices — STL files store each triangle independently
    # so shared edges appear twice with slightly different coordinates.
    # clean() merges vertices within tolerance, halving the edge count.
    mesh_tri = mesh_tri.clean(tolerance=1e-6)

    vertices = np.array(mesh_tri.points, dtype=np.float32)
    faces_flat = np.array(mesh_tri.faces)
    faces = faces_flat.reshape(-1, 4)[:, 1:].astype(np.int64)

    pressure = None
    for name in ['p', 'pressure', 'Pressure', 'pMean']:
        if name in mesh_tri.point_data:
            arr = np.array(mesh_tri.point_data[name], dtype=np.float32)
            pressure = arr[:, 0] if arr.ndim > 1 else arr
            break

    wss = None
    for name in ['wallShearStress', 'WallShearStress', 'wss', 'WSS']:
        if name in mesh_tri.point_data:
            arr = np.array(mesh_tri.point_data[name], dtype=np.float32)
            wss = arr.reshape(-1, 3) if arr.ndim == 1 else arr
            break

    cd = float(mesh_tri.field_data['cd'][0]) if 'cd' in mesh_tri.field_data else None
    cl = float(mesh_tri.field_data['cl'][0]) if 'cl' in mesh_tri.field_data else None

    return {'vertices': vertices, 'faces': faces,
            'pressure': pressure, 'wss': wss,
            'cd_total': cd, 'cl_total': cl}


# def mesh_to_pyg_data(raw, rho=1.225, U_inf=83.33, design_id="") -> Data:
    vertices = raw['vertices']
    faces    = raw['faces']
    pressure = raw['pressure']
    wss      = raw['wss']

    verts_t = torch.from_numpy(vertices)
    faces_t = torch.from_numpy(faces).long()
    U_ref = 100
    U_col = torch.full((verts_t.shape[0], 1), U_inf/U_ref)
    x = normalise_mesh(torch.cat([verts_t, U_col], dim=-1))

    q_inf = 0.5 * rho * U_inf ** 2
    P_INF = 0.0
    # if pressure is not None:
        # p_ref = float(np.mean(pressure))
        # cp = torch.from_numpy((pressure - p_ref) / q_inf).float()
          # freestream static pressure — 0 Pa in DrivAerNet (gauge pressure)

    if pressure is not None:
        cp = torch.from_numpy((pressure - P_INF) / q_inf).float()
    else:
        cp = torch.zeros(verts_t.shape[0])


    mu= 1.81e-5   # dynamic viscosity of air at 20°C, Pa·s
    L_ref = 4.6       # car length, metres

    if wss is not None:
        tau_ref  = mu * U_inf / L_ref          # physical reference scale ~3.2e-4 Pa
        wss_new  = wss / tau_ref               # rescale to order ~100-500

        wss_mean = wss_new.mean()              # scalar — mean over all V×3 elements
        wss_std  = wss_new.std().clip(1e-8)    # scalar std, clip instead of clamp (numpy)

        wss_nd   = torch.from_numpy(
                    ((wss_new - wss_mean) / wss_std).astype(np.float32)
                )                           # (V, 3), std=1.0
    else:
        wss_nd = torch.zeros(verts_t.shape[0], 3)
        cd = torch.tensor([raw['cd_total'] or 0.0])
        cl = torch.tensor([raw['cl_total'] or 0.0])

    edge_index = build_edge_index_from_faces(faces_t)
    geo = precompute_geometry(verts_t, faces_t, edge_index)

    return Data(x=x, edge_index=edge_index, pos=verts_t, face=faces_t.T,
                edge_angles=geo['angles'], edge_transporters=geo['transporters'],
                y_cp=cp, y_wss=wss_nd, wss_mean = wss_mean, y_cd=cd, y_cl=cl,
                design_id=design_id, num_nodes=verts_t.shape[0])


def mesh_to_pyg_data(raw, rho=1.225, U_inf=83.33, design_id="") -> Data:
    vertices = raw['vertices']
    faces    = raw['faces']
    pressure = raw['pressure']
    wss      = raw['wss']

    verts_t = torch.from_numpy(vertices)
    faces_t = torch.from_numpy(faces).long()
    U_ref = 100.0
    U_col = torch.full((verts_t.shape[0], 1), U_inf/U_ref)  # normalise U_inf to order 1 for better training stability
    x = normalise_mesh(torch.cat([verts_t, U_col], dim=-1))

    # ── Cp ────────────────────────────────────────────────────────────────
    q_inf = 0.5 * rho * U_inf ** 2
    P_INF = 0.0

    if pressure is not None:
        cp_raw  = (pressure - P_INF) / q_inf        # dimensionless
        cp_sl   = symlog(cp_raw)                    # compress heavy tails
        cp_mean = cp_sl.mean()
        cp_std  = cp_sl.std().clip(1e-8)
        cp_nd   = torch.from_numpy(
                    np.clip((cp_sl - cp_mean) / cp_std, -5.0, 5.0).astype(np.float32)
                  )                                       # ±5σ Winsorisation
    else:
        cp_nd = torch.zeros(verts_t.shape[0])

    # ── WSS ───────────────────────────────────────────────────────────────
    mu    = 1.81e-5   # dynamic viscosity of air, Pa·s
    L_ref = 4.6       # car length, metres

    if wss is not None:
        tau_ref  = mu * U_inf / L_ref
        wss_new  = wss / tau_ref
        wss_new  = symlog(wss_new)              # compress heavy tails
        wss_mean = wss_new.mean()
        wss_std  = wss_new.std().clip(1e-8)
        wss_nd   = torch.from_numpy(
                    np.clip((wss_new - wss_mean) / wss_std, -5.0, 5.0).astype(np.float32)
                )                           # ±5σ Winsorisation
    else:
        wss_nd   = torch.zeros(verts_t.shape[0], 3)
        wss_mean = 0.0
        wss_std  = 1.0

    # ── Cd / Cl ───────────────────────────────────────────────────────────
    # defined here, outside all if blocks — always assigned
    cd = torch.tensor([raw['cd_total'] or 0.0], dtype=torch.float32)
    cl = torch.tensor([raw['cl_total'] or 0.0], dtype=torch.float32)

    # ── Graph geometry ────────────────────────────────────────────────────
    edge_index = build_edge_index_from_faces(faces_t)
    geo = precompute_geometry(verts_t, faces_t, edge_index)
    # In mesh_to_pyg_data, add to input features:
    x_frac = (verts_t[:, 0] - verts_t[:, 0].min()) / (verts_t[:, 0].max() - verts_t[:, 0].min())  # 0=front, 1=rear
    z_frac = (verts_t[:, 2] - verts_t[:, 2].min()) / (verts_t[:, 2].max() - verts_t[:, 2].min())  # 0=bottom, 1=top
    # Now the model KNOWS a vertex is on the floor vs the roof
    return Data(
        x                 = x,
        edge_index        = edge_index,
        pos               = verts_t,
        face              = faces_t.T,
        edge_angles       = geo['angles'],
        edge_transporters = geo['transporters'],
        y_cp              = cp_nd,
        y_wss             = wss_nd,
        wss_mean          = wss_mean,
        wss_std           = wss_std,
        y_cd              = cd,
        y_cl              = cl,
        design_id         = design_id,
        num_nodes         = verts_t.shape[0],
    )

class DrivAerNetDataset(Dataset):
    def __init__(self, data_root, split='train', rho=1.225, U_inf=83.33,
                 force_reload=False, max_vertices=None):
        self.data_root    = data_root
        self.split        = split
        self.rho          = rho
        self.U_inf        = U_inf
        self.force_reload = force_reload

        split_file = os.path.join(data_root, 'split.json')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.design_ids = json.load(f)[split]
        else:
            all_vtps = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(data_root, 'meshes'))
                if f.endswith('.vtp')
            ])
            n = len(all_vtps)
            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)
            if split == 'train':
                self.design_ids = all_vtps[:n_train]
            elif split == 'val':
                self.design_ids = all_vtps[n_train:n_train + n_val]
            else:
                self.design_ids = all_vtps[n_train + n_val:]

        self.cache_dir = os.path.join(data_root, 'processed', split)
        os.makedirs(self.cache_dir, exist_ok=True)
        super().__init__(root=None)

    def len(self):
        return len(self.design_ids)

    def get(self, idx):
        did = self.design_ids[idx]
        cache_path = os.path.join(self.cache_dir, f"{did}.pt")
        if os.path.exists(cache_path) and not self.force_reload:
            return torch.load(cache_path, weights_only=False)
        vtp_path = os.path.join(self.data_root, 'meshes', f"{did}.vtp")
        if not os.path.exists(vtp_path):
            raise FileNotFoundError(f"Missing: {vtp_path}")
        print(f"  Processing {did}...")
        raw  = load_merged_vtp(vtp_path)
        data = mesh_to_pyg_data(raw, rho=self.rho, U_inf=self.U_inf,
                                  design_id=did)
        try:
            torch.save(data, cache_path)
        except Exception as e:
            print(f'    Cache save skipped: {e}')
        return data


def make_synthetic_dataset(n_meshes=8, n_vertices=500,
                            U_inf=83.33, rho=1.225):
    from scipy.spatial import ConvexHull
    dataset = []
    for i in range(n_meshes):
        theta = np.random.uniform(0, np.pi, n_vertices)
        phi   = np.random.uniform(0, 2 * np.pi, n_vertices)
        r     = 1.0 + 0.3 * np.random.randn(n_vertices)
        verts = np.stack([
            2.5 * r * np.sin(theta) * np.cos(phi),
            0.8 * r * np.sin(theta) * np.sin(phi),
            0.6 * r * np.cos(theta)
        ], axis=1).astype(np.float32)
        faces = ConvexHull(verts).simplices.astype(np.int64)
        q_inf = 0.5 * rho * U_inf ** 2
        raw = {
            'vertices': verts, 'faces': faces,
            'pressure': (-0.5*np.cos(theta)+0.1*np.random.randn(n_vertices)).astype(np.float32)*q_inf,
            'wss': np.stack([0.01*np.sin(phi), 0.01*np.cos(theta),
                             0.005*np.ones(n_vertices)], axis=1).astype(np.float32)*q_inf,
            'cd_total': 0.3 + 0.05*np.random.randn(),
            'cl_total': -1.5 + 0.1*np.random.randn(),
        }
        dataset.append(mesh_to_pyg_data(raw, rho=rho, U_inf=U_inf,
                                         design_id=f"synthetic_{i:04d}"))
    return dataset
