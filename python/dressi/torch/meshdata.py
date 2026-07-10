"""Per-topology cached data derived from the `tri` tensor."""

from __future__ import annotations

import numpy as np
import torch

from .. import _C
from .convert import tri_int32


class MeshData:
    """Static-topology arrays for one triangle list.

    Dressi wants faces in two encodings (IVEC3 for the rasterizers, float
    VEC3 textures for the gather ops) plus adjacency textures and the
    unweld gather index; all are computed once per distinct `tri`.
    """

    def __init__(self, tri: torch.Tensor):
        # Pin the source tensor: the cache key contains its data_ptr, and
        # holding it prevents a freed tri's address from being reused by a
        # different topology (a real, allocation-order-dependent bug)
        self.src = tri
        tri = tri_int32(tri)
        self.n_faces = int(tri.shape[0])
        self.n_verts = int(tri.max().item()) + 1 if self.n_faces else 0
        # (1, F, 3) float32 -- both the IVEC3 upload (converted to uint at
        # sendImg) and the float faces texture use this layout
        self.faces_f32 = tri.to(torch.float32).unsqueeze(0).contiguous().numpy()
        # Unweld gather: vertex-array index per unwelded corner (3F,)
        self.unweld = tri.reshape(-1).long()
        # Sequential faces for unwelded geometry: (1, F, 3) = (3i, 3i+1, 3i+2)
        self.faces_seq_f32 = (
            np.arange(self.n_faces * 3, dtype=np.float32).reshape(1, -1, 3)
        )
        # Vertex -> incident-face adjacency texture (max_deg, V, 1)
        self.vtx_faces_tex = _C.vertex_faces_tex(self.faces_f32, self.n_verts)
        self.max_deg = int(self.vtx_faces_tex.shape[0])
        # Per-corner face id for unwelded soft geometry (1, 3F, 1)
        self.corner_face_id = np.repeat(
            np.arange(self.n_faces, dtype=np.float32), 3
        ).reshape(1, -1, 1)


def mesh_key(tri: torch.Tensor) -> tuple:
    return (tri.data_ptr(), tri._version, tuple(tri.shape))
