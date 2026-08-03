"""Clustering of local DD optima on the Grassmannian.

Port of SSDI-1's ``utils/Lcluster.m`` and ``utils/Lhcluster.m`` (single-link
clustering by Grassmannian distance and hierarchical clustering wrappers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ._grassmann import gmetrics

__all__ = ["ClusterResult", "Lcluster", "Lhcluster"]


@dataclass
class ClusterResult:
    """Cluster representatives and sizes."""

    uidx: npt.NDArray[np.intp]
    usiz: npt.NDArray[np.intp]
    nruns: int


def Lcluster(distance: npt.NDArray[np.floating], tol: float) -> ClusterResult:
    """Greedy single-linkage clustering on a distance matrix.

    Port of ``Lcluster.m``. Input ``distance`` must be sorted by DD (ascending)
    so that the first index of each cluster is its representative.
    """
    distance = np.asarray(distance, dtype=float)
    R = distance.shape[0]
    available = np.ones(R, dtype=bool)
    uidx_list: list[int] = []
    usiz_list: list[int] = []
    for i in range(R):
        if available[i]:
            uidx_list.append(i)
            size = 1
            available[i] = False
            for j in range(R):
                if available[j] and distance[i, j] < tol:
                    available[j] = False
                    size += 1
            usiz_list.append(size)
    return ClusterResult(
        uidx=np.array(uidx_list, dtype=np.intp),
        usiz=np.array(usiz_list, dtype=np.intp),
        nruns=len(uidx_list),
    )


def Lhcluster(
    L: npt.NDArray[np.floating],
    threshold: float,
    method: str = "average",
    max_angle: bool = True,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """Hierarchical clustering of projections by Grassmannian distance.

    Port of ``Lhcluster.m`` (simplified — no Gnuplot output). Returns
    ``(cluster_assignments, linkage_matrix)``.
    """
    D = gmetrics(L, max_angle=max_angle)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method=method)
    clusters = fcluster(Z, t=threshold, criterion="distance").astype(np.intp)
    return clusters, Z
