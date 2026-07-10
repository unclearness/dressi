"""Side table attaching producer metadata to tensors we return.

nvdiffrast's signatures drop information the Dressi backward ops need
(e.g. `interpolate(attr, rast, tri)` has no `pos`, but the exact FaceFetch
backward wants the corner clip positions). `rasterize` therefore records
its inputs against the returned `rast` tensor, and `interpolate` records
against the returned attribute image for `texture`'s inverse-UV table.
Entries live exactly as long as the tensor object does.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

_table: dict[int, tuple[weakref.ref, Any]] = {}


@dataclass
class RastInfo:
    pos: Any  # CPU float32 clip positions [N, V, 4] (GL convention)
    tri: Any  # int32 [F, 3]
    resolution: tuple[int, int]  # (H, W)
    extra: dict = field(default_factory=dict)


@dataclass
class InterpInfo:
    attr: Any  # the vertex attribute tensor given to interpolate
    rast_info: RastInfo


def attach(tensor, info) -> None:
    key = id(tensor)

    def _drop(_ref, _key=key):
        _table.pop(_key, None)

    _table[key] = (weakref.ref(tensor, _drop), info)


def lookup(tensor, kind: str):
    entry = _table.get(id(tensor))
    if entry is None or entry[0]() is not tensor:
        raise RuntimeError(
            f"dressi.torch: this tensor has no {kind} provenance -- it "
            "must be produced by dressi.torch (e.g. `rast` from "
            "rasterize(), `uv` from interpolate()) within the same process."
        )
    return entry[1]
