"""Pre-defined test networks. Ports of SSDI-1's ``networks/`` directory."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["tnet9a", "tnet9b", "tnet9c", "tnet9d", "tneter", "tnetmod", "tnet243"]


def tnet9a() -> npt.NDArray[np.intp]:
    """9-node test network (a). Port of ``tnet9a.m``."""
    return np.array(
        [
            [1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
        ],
        dtype=np.intp,
    )


def tnet9b() -> npt.NDArray[np.intp]:
    """9-node test network (b). Port of ``tnet9b.m``."""
    return np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.intp,
    )


def tnet9c() -> npt.NDArray[np.intp]:
    """9-node test network (c). Port of ``tnet9c.m``."""
    return np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=np.intp,
    )


def tnet9d() -> npt.NDArray[np.intp]:
    """9-node test network (d) — fully connected ring with self-loops."""
    n = 9
    G = np.eye(n, dtype=np.intp)
    for i in range(n):
        G[i, (i + 1) % n] = 1
    return G


def tneter(n: int, p: float, rng: np.random.Generator | None = None) -> npt.NDArray[np.intp]:
    """Erdős-Rényi random directed graph (with self-loops).

    Port of ``tneter.m``. Returns binary adjacency of shape ``(n, n)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    G = np.eye(n, dtype=np.intp)
    mask = rng.random((n, n)) < p
    G[mask] = 1
    return G


def tnet243() -> npt.NDArray[np.intp]:
    """9-node test network with a 2 + 4 + 3 module split.

    Layout (rows = target, columns = source, matching MVGC2 convention):

    - **2-module** (nodes 0-1): fully intra-connected, sends to the 4-module.
    - **4-module** (nodes 2-5): fully intra-connected, *receives* from the
      2-module.
    - **3-module** (nodes 6-8): fully intra-connected, **no incoming
      connections** from any other module.

    Because the 3-module is dynamically isolated and the 2+4-module union
    is dynamically closed (its dynamics depend only on its own past), one
    expects two natural emergent macros in any innovations-form SS fit:

    - ``m = 3``: aligned with the 3-module (nodes 6, 7, 8).
    - ``m = 6``: aligned with the 2+4-module union (nodes 0-5).
    """
    return tnetmod([2, 4, 3], cons=np.array([[1, 0]], dtype=np.intp))


def tnetmod(
    smod: list[int] | npt.NDArray[np.intp],
    cons: npt.NDArray[np.intp] | None = None,
) -> npt.NDArray[np.intp]:
    """Modular network with intra-module fully-connected and configurable
    inter-module connections.

    Port of ``tnetmod.m``. ``smod`` is a list of module sizes; ``cons`` is an
    optional connection specification:

    - ``shape (k, 2)``: each row ``(i, j)`` fully inter-connects modules ``i``
      and ``j`` (0-indexed in Python).
    - ``shape (k, 4)``: each row ``(i, a, j, b)`` adds a single edge from node
      ``a`` of module ``i`` to node ``b`` of module ``j``.
    """
    smod = np.asarray(smod, dtype=np.intp)
    nmods = smod.size
    nnodes = int(smod.sum())
    G = np.eye(nnodes, dtype=np.intp)

    starts = np.zeros(nmods, dtype=np.intp)
    starts[1:] = np.cumsum(smod[:-1])
    slices = [slice(int(starts[m]), int(starts[m] + smod[m])) for m in range(nmods)]

    for m in range(nmods):
        G[slices[m], slices[m]] = 1

    if cons is not None:
        cons = np.asarray(cons, dtype=np.intp)
        if cons.ndim == 1:
            cons = cons[None, :]
        if cons.shape[1] == 2:
            for i, j in cons:
                G[slices[i], slices[j]] = 1
        elif cons.shape[1] == 4:
            for i, a, j, b in cons:
                G[starts[i] + a, starts[j] + b] = 1
        else:
            raise ValueError("cons must have 2 or 4 columns")
    return G
