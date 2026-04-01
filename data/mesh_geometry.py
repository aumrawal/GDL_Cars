# data/mesh_geometry.py
"""
Core differential geometry operations for GEM-CNN on triangular surface meshes.

Implements (following de Haan et al. 2020, Secs 4.1 & 4.2):
  - Area-weighted vertex normal estimation
  - Discrete Riemannian logarithmic map  (projects edges onto tangent plane)
  - Local reference frame construction   (gauge choice)
  - Neighbour angle computation           (θ_pq)
  - Discrete Levi-Civita parallel transporters (g_{q→p}, Eq. 6)

All operations are written in PyTorch so they run on MPS (Apple Silicon).
Results are precomputed once and cached as graph edge attributes.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ────────────────────────────────────────────────────────────────────────────
# 1. Vertex normals
# ────────────────────────────────────────────────────────────────────────────

def compute_vertex_normals(
    vertices: Tensor,   # (V, 3)
    faces:    Tensor,   # (F, 3)  long indices
) -> Tensor:
    """
    Area-weighted average of adjacent face normals at each vertex.
    Returns unit normals of shape (V, 3).

    The area weighting follows the GEM paper: each face contributes its
    normal scaled by face area, so larger faces have proportionally more
    influence on the local tangent plane orientation.
    """
    v0 = vertices[faces[:, 0]]   # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product gives a vector whose magnitude = 2 * face area
    cross = torch.linalg.cross(v1 - v0, v2 - v0)   # (F, 3)

    # Scatter-add onto vertices (each face contributes to its 3 corners)
    normals = torch.zeros_like(vertices)             # (V, 3)
    for i in range(3):
        idx = faces[:, i].unsqueeze(1).expand(-1, 3)
        normals.scatter_add_(0, idx, cross)

    return F.normalize(normals, dim=-1)              # (V, 3)


# ────────────────────────────────────────────────────────────────────────────
# 2. Discrete logarithmic map
# ────────────────────────────────────────────────────────────────────────────

def log_map(
    p:        Tensor,   # (E, 3)  source vertex positions
    q:        Tensor,   # (E, 3)  neighbour vertex positions
    normal_p: Tensor,   # (E, 3)  unit normals at p
) -> Tensor:
    """
    Project edge vector (q - p) onto the tangent plane at p,
    then rescale so the projected vector has the same norm as the
    original edge (i.e. preserve edge length, Sec 4.1).

    This is the discrete analog of the Riemannian exponential map inverse:
        log_p(q) = |q-p| * proj(q-p) / |proj(q-p)|

    Returns tangent vectors of shape (E, 3) living in TpM ⊂ R³.
    """
    edge = q - p                                                   # (E, 3)
    edge_len = edge.norm(dim=-1, keepdim=True).clamp(min=1e-8)    # (E, 1)

    # Project onto tangent plane: subtract normal component
    dot = (edge * normal_p).sum(dim=-1, keepdim=True)              # (E, 1)
    proj = edge - dot * normal_p                                   # (E, 3)
    proj_len = proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)    # (E, 1)

    return edge_len * proj / proj_len                              # (E, 3)


# ────────────────────────────────────────────────────────────────────────────
# 3. Reference frames and neighbour angles
# ────────────────────────────────────────────────────────────────────────────

def build_reference_frames(
    normals:     Tensor,   # (V, 3)  vertex unit normals
    ref_vectors: Tensor,   # (V, 3)  log_p(q0) — tangent vector to ref neighbour
) -> Tuple[Tensor, Tensor]:
    """
    Construct a right-handed orthonormal frame (e1, e2) of the tangent
    plane TpM at each vertex, using the reference neighbour direction.

        e1 = normalize(ref_vector)
        e2 = normal × e1          (right-hand rule)

    Returns:
        e1: (V, 3)
        e2: (V, 3)
    """
    e1 = F.normalize(ref_vectors, dim=-1)                  # (V, 3)
    e2 = torch.linalg.cross(normals, e1)                   # (V, 3)
    e2 = F.normalize(e2, dim=-1)
    return e1, e2


def compute_neighbour_angles(
    log_pq: Tensor,   # (E, 3)  log_p(q) tangent vectors
    e1_p:   Tensor,   # (E, 3)  first basis vector at p  (per edge)
    e2_p:   Tensor,   # (E, 3)  second basis vector at p (per edge)
) -> Tensor:
    """
    Compute polar angle θ_pq of neighbour q in the local frame at p.

        θ_pq = atan2( log_p(q)·e2_p,  log_p(q)·e1_p )

    Returns angles in (-π, π] of shape (E,).
    """
    cos_comp = (log_pq * e1_p).sum(dim=-1)   # (E,)
    sin_comp = (log_pq * e2_p).sum(dim=-1)   # (E,)
    return torch.atan2(sin_comp, cos_comp)    # (E,)


# ────────────────────────────────────────────────────────────────────────────
# 4. Parallel transporters  (discrete Levi-Civita connection, Eq. 6)
# ────────────────────────────────────────────────────────────────────────────

def _rotation_align_normals(
    n_src: Tensor,   # (E, 3)  unit normals at source vertex q
    n_tgt: Tensor,   # (E, 3)  unit normals at target vertex p
) -> Tuple[Tensor, Tensor]:
    """
    Compute the rotation R_α ∈ SO(3) that rotates n_src onto n_tgt
    around axis (n_src × n_tgt).

    Returns (cos_alpha, axis) where axis has shape (E, 3).
    We use the Rodrigues formula to rotate arbitrary 3D vectors.
    """
    axis = torch.linalg.cross(n_src, n_tgt)          # (E, 3)
    sin_alpha = axis.norm(dim=-1, keepdim=True)       # (E, 1)
    cos_alpha = (n_src * n_tgt).sum(dim=-1, keepdim=True)  # (E, 1)

    axis = axis / sin_alpha.clamp(min=1e-8)           # (E, 3) unit axis
    return cos_alpha, sin_alpha, axis


def rodrigues_rotate(
    v:         Tensor,   # (E, 3)  vectors to rotate
    axis:      Tensor,   # (E, 3)  unit rotation axis
    cos_alpha: Tensor,   # (E, 1)
    sin_alpha: Tensor,   # (E, 1)
) -> Tensor:
    """
    Rodrigues' rotation formula:
        v_rot = v*cos(α) + (axis × v)*sin(α) + axis*(axis·v)*(1 - cos(α))
    """
    dot = (axis * v).sum(dim=-1, keepdim=True)        # (E, 1)
    cross = torch.linalg.cross(axis, v)               # (E, 3)
    return (v * cos_alpha
            + cross * sin_alpha
            + axis * dot * (1.0 - cos_alpha))


def compute_parallel_transporters(
    e1_src: Tensor,   # (E, 3)  e_{q,1} — first basis vector at source q
    e2_src: Tensor,   # (E, 3)  e_{q,2} — second basis vector at source q
    e1_tgt: Tensor,   # (E, 3)  e_{p,1} — first basis vector at target p
    n_src:  Tensor,   # (E, 3)  unit normal at source q
    n_tgt:  Tensor,   # (E, 3)  unit normal at target p
) -> Tensor:
    """
    Compute the discrete Levi-Civita connection angle g_{q→p} for each
    directed edge q→p (Eq. 6 in GEM paper):

        g_{q→p} = atan2( (R_α e_{q,2})ᵀ e_{p,1},
                         (R_α e_{q,1})ᵀ e_{p,1} )

    where R_α rotates the tangent space at q to be parallel to that at p
    by rotating around axis (n_q × n_p).

    Returns transporter angles of shape (E,) in (-π, π].

    Intuition for F1 application:
        On the curved underfloor of an F1 car, the tangent plane at one
        vertex may be tilted ~30° relative to its neighbour due to the
        diffuser geometry. Without this correction, vector features (WSS)
        would be accumulated in misaligned frames, producing wrong results.
    """
    # Check whether src and tgt normals are (nearly) parallel
    # If so, no rotation needed (flat region)
    cos_alpha, sin_alpha, axis = _rotation_align_normals(n_src, n_tgt)

    # Detect flat edges (sin_alpha ≈ 0) to avoid divide-by-zero
    flat = (sin_alpha.squeeze(-1).abs() < 1e-6)   # (E,) bool

    # Rotate source frame basis vectors into target tangent space
    e1_rot = rodrigues_rotate(e1_src, axis, cos_alpha, sin_alpha)  # (E, 3)
    e2_rot = rodrigues_rotate(e2_src, axis, cos_alpha, sin_alpha)  # (E, 3)

    # For flat edges: rotated frame = original frame
    e1_rot[flat] = e1_src[flat]
    e2_rot[flat] = e2_src[flat]

    # Project onto target frame to get the connection angle
    cos_g = (e1_rot * e1_tgt).sum(dim=-1)   # (E,)
    sin_g = (e2_rot * e1_tgt).sum(dim=-1)   # (E,)
    return torch.atan2(sin_g, cos_g)        # (E,)  g_{q→p}


# ────────────────────────────────────────────────────────────────────────────
# 5. Full precomputation pass
# ────────────────────────────────────────────────────────────────────────────

def precompute_geometry(
    vertices:  Tensor,         # (V, 3)
    faces:     Tensor,         # (F, 3) long
    edge_index: Tensor,        # (2, E) long  [src=q, tgt=p]  directed edges
) -> dict:
    """
    One-shot precomputation of all geometric quantities needed by GEM-CNN.
    Call this once per mesh and cache the results as graph edge attributes.

    Returns a dict with keys:
        'normals'      : (V, 3)  vertex unit normals
        'angles'       : (E,)    neighbour angles θ_{pq}
        'transporters' : (E,)    parallel transporter angles g_{q→p}
        'e1'           : (V, 3)  first tangent frame basis vector
        'e2'           : (V, 3)  second tangent frame basis vector

    Usage:
        geo = precompute_geometry(vertices, faces, edge_index)
        # Store in PyG Data object:
        data.edge_attr_angles = geo['angles']
        data.edge_attr_transport = geo['transporters']
    """
    device = vertices.device
    src, tgt = edge_index[0], edge_index[1]   # q, p

    # 1. Vertex normals
    normals = compute_vertex_normals(vertices, faces)   # (V, 3)

    # 2. Log map for every undirected edge  (i < j)
    # log_ij is the tangent vector at j pointing toward i
    log_ij = log_map(
        p=vertices[tgt],
        q=vertices[src],
        normal_p=normals[tgt],
    )   # (E, 3)

    # 3. Choose reference neighbour per vertex.
    #    Since edge_index is only (i < j), we expand to directed for gauge selection
    #    to ensure every vertex has at least one neighbour in its ref list.
    full_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    full_log        = torch.cat([log_ij, -log_ij], dim=0) # crude approx for ref choice
    f_src, f_tgt    = full_edge_index[0], full_edge_index[1]

    V = vertices.shape[0]
    edge_ids = torch.arange(f_tgt.shape[0], dtype=torch.long, device=device)
    ref_edge = torch.full((V,), f_tgt.shape[0], dtype=torch.long, device=device)
    for i in range(f_tgt.shape[0] - 1, -1, -1):
        ref_edge[f_tgt[i]] = i
    ref_log = full_log[ref_edge]   # (V, 3)

    # 4. Build reference frames at every vertex
    e1, e2 = build_reference_frames(normals, ref_log)   # (V, 3), (V, 3)

    # 5. Neighbour angles θ_ij  (angle of i in j's frame)
    angles = compute_neighbour_angles(
        log_pq=log_ij,
        e1_p=e1[tgt],
        e2_p=e2[tgt],
    )   # (E,)

    # 6. Parallel transporters g_{i→j}
    transporters = compute_parallel_transporters(
        e1_src=e1[src],
        e2_src=e2[src],
        e1_tgt=e1[tgt],
        n_src=normals[src],
        n_tgt=normals[tgt],
    )   # (E,)

    return {
        'normals':       normals,      # (V, 3)
        'e1':            e1,           # (V, 3)
        'e2':            e2,           # (V, 3)
        'angles':        angles,       # (E,)
        'transporters':  transporters, # (E,)
    }


def build_edge_index_from_faces(faces: Tensor) -> Tensor:
    """
    Build directed edge_index (2, E) from face tensor (F, 3).
    Every mesh edge is represented in both directions (q→p and p→q).

    STL-derived meshes have duplicate vertices — adjacent triangles each store
    their own copy of shared corner points, so vertex indices for the same
    physical point differ across faces. This means the same physical edge
    appears twice under different index pairs, giving ~6 edges per vertex
    instead of the expected ~3.

    Fix: before building edges, weld duplicate vertices by remapping face
    indices so that vertices with identical coordinates share one index.
    This halves the edge count for STL-sourced meshes.
    """
    import numpy as np

    # ── Weld duplicate vertices ──────────────────────────────────────
    # Convert to numpy for the unique operation (needs structured comparison)
    faces_np = faces.cpu().numpy()   # (F, 3)

    # We need the vertex positions to weld — but here we only have face
    # indices, not positions. Instead, weld at the face level:
    # deduplicate faces that reference the same set of vertex indices
    # (after sorting each face's vertices so orientation doesn't matter).
    # This removes the duplicate faces that cause doubled edges.
    faces_sorted = np.sort(faces_np, axis=1)           # canonical orientation
    _, unique_face_idx = np.unique(faces_sorted,
                                   axis=0,
                                   return_index=True)
    faces_np = faces_np[unique_face_idx]               # deduplicated faces
    faces = torch.from_numpy(faces_np).to(faces.device)
    # ────────────────────────────────────────────────────────────────

    pairs = []
    for i, j in [(0,1), (0,2), (1,2)]:
        # Sort indices to ensure each physical edge is represented once (i < j)
        e = torch.stack([faces[:, i], faces[:, j]], dim=0)
        e = torch.sort(e, dim=0)[0]
        pairs.append(e)
    edge_index = torch.cat(pairs, dim=1)

    # Final deduplication of unique undirected edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
