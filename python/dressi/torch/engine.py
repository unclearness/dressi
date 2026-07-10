"""Engine: one cached DressiAD graph per shape signature.

Dressi is define-and-run: a graph is built once (packing + GLSL compile +
Vulkan objects, expensive) and then executed as a cached command buffer
where only leaf VALUES change (cheap). The torch layer reconciles this
with eager execution by keying engines on the call's shape signature and
reusing them across calls -- the same trick nvdiffrast plays with its
per-context shape-keyed buffers.

External gradients enter through the surrogate loss `output * seed`:
BuildBackward seeds gradient 1 everywhere (the implicit sum), so the leaf
gradients equal the exact VJP for grad_output == seed.
"""

from __future__ import annotations

import numpy as np

from .. import _C


class Engine:
    """A DressiAD graph with named leaves, one output, optional gradients."""

    def __init__(self, ctx_native):
        self.ad = _C.DressiAD(ctx_native)
        # Disable reactive pruning: fwd uploads inputs, bwd uploads only the
        # seed -- the alternating dirty pattern would thrash SUBSTAGE repacks.
        self.ad.set_rebuild_counts(0, 0)
        self.leaves: dict[str, _C.Variable] = {}
        self.outputs: list = []
        self._grad_vars: dict[str, object] | None = None
        self._grad_leaf_names: list[str] = []
        self._uploaded: dict[str, tuple] = {}
        self._executed = False

    # ------------------------------- build ---------------------------------
    def leaf(self, name: str, vtype, w: int, h: int):
        var = _C.Variable(vtype, w, h)
        self.leaves[name] = var
        return var

    def finalize(self, outputs, grad_leaves: list[str] = ()):  # noqa: B006
        """Set the output(s) (and their surrogate-loss backward when
        grad_leaves is non-empty). Must be called once, after graph
        construction. Multiple outputs must share one vtype/size (their
        seeded products are combined pixel-wise into a single loss)."""
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self._grad_leaf_names = list(grad_leaves)
        if grad_leaves:
            prods = []
            for i, out in enumerate(self.outputs):
                seed = self.leaf(self.seed_name(i), out.vtype, out.width,
                                 out.height)
                prods.append(out * seed)
            loss = prods[0] if len(prods) == 1 else _C.f.sum_pixel_wise(prods)
            for name in grad_leaves:
                self.leaves[name].set_requires_grad_recursively(True)
            self.ad.set_loss_var(loss)
            self.ad.set_grad_outputs_enabled(True)
        else:
            self.ad.set_loss_var(self.outputs[0])
        for out in self.outputs:
            self.ad.mark_output(out)

    @staticmethod
    def seed_name(i: int = 0) -> str:
        return "__seed__" if i == 0 else f"__seed{i}__"

    # ------------------------------ transfer -------------------------------
    def upload(self, name: str, array: np.ndarray, token=None, ref=None):
        """sendImg with dirty-skip: identical (token) uploads are elided so
        static data crosses PCIe exactly once. Tokens contain data_ptr, so
        the skip is only sound while the pointed-at memory cannot be reused
        by a new tensor: `ref` (the source tensor the token was derived
        from) is retained alongside the token to pin it."""
        if token is not None:
            prev = self._uploaded.get(name)
            if prev is not None and prev[0] == token:
                return
        array = np.ascontiguousarray(array)
        self.ad.send_img(self.leaves[name], array)
        if token is not None:
            self._uploaded[name] = (token, ref)

    def run(self):
        self.ad.exec_step()
        self._executed = True

    def read_output(self, i: int = 0) -> np.ndarray:
        return self.ad.recv_img(self.outputs[i])

    def read_grad(self, name: str) -> np.ndarray:
        return self.ad.recv_img(self._grad_var(name))

    # ------------------------------ internal -------------------------------
    def _grad_var(self, name: str):
        if self._grad_vars is None:
            assert self._executed, "backward before the first exec_step"
            by_id = {var.id: var for var in self.leaves.values()}
            names = {var.id: n for n, var in self.leaves.items()}
            self._grad_vars = {}
            for leaf, grad in self.ad.input_grads():
                if leaf.id in by_id:
                    self._grad_vars[names[leaf.id]] = grad
        return self._grad_vars[name]


def upload_token(t) -> tuple:
    """Cheap change-detection token for a torch tensor: in-place edits bump
    _version, replaced tensors change data_ptr."""
    return (t.data_ptr(), t._version, tuple(t.shape))
