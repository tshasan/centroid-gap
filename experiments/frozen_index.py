"""Read a frozen IVF index staged by ivf_workload_index.py, and score its centroids.

A run directory RUN/<workload>/ holds
    owner.npy    list of each base vector (int, one entry per base id)
    sizes.npy    vectors per list
    centers.npy  the stored Faiss centroids
    <split>.npz  `queries` and their exact top-10 `truth` ids, for split in fit, val, test (and image on t2i)
The Gate A and B router studies (tip_router, power_router, power_router_v2, reanalyse_power_router,
list_effective_dimension) read only these files; nothing is re-clustered.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKLOADS = ("deep", "t2i")
TOP_SPREAD = 32  # score_spread compares each query's best and 32nd-best centroid scores


def metric_for(workload):
    """Native metric of a workload: inner product for Text-to-Image, squared L2 for DEEP."""
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; expected one of {WORKLOADS}")
    return "ip" if workload == "t2i" else "l2"


@dataclass(frozen=True)
class FrozenIndex:
    owner: np.ndarray
    sizes: np.ndarray
    centers: np.ndarray

    @property
    def nlist(self):
        return len(self.sizes)


def load_frozen_index(directory):
    """owner, sizes and centers from `directory`, checked against each other."""
    directory = Path(directory)
    owner = np.load(directory / "owner.npy")
    sizes = np.load(directory / "sizes.npy")
    centers = np.load(directory / "centers.npy")
    if len(centers) != len(sizes):
        raise ValueError(f"{directory}: {len(centers)} centers but {len(sizes)} list sizes")
    if not np.array_equal(np.bincount(owner, minlength=len(sizes)), sizes):
        raise ValueError(f"{directory}: sizes.npy does not count the lists in owner.npy")
    return FrozenIndex(owner, sizes, centers)


def load_split(path):
    """(queries, truth) from one staged .npz split."""
    with np.load(path) as z:
        queries, truth = z["queries"], z["truth"]
    if len(queries) != len(truth):
        raise ValueError(f"{path}: {len(queries)} queries but {len(truth)} truth rows")
    return queries, truth


def centroid_scores(queries, centers, metric):
    """Routing score of every list, higher is better.

    ip: <q, c>.  l2: 2<q, c> - |c|^2, which is -|q - c|^2 up to a per-query constant."""
    s = queries @ centers.T
    return s if metric == "ip" else 2 * s - (centers ** 2).sum(axis=1)[None, :]


def score_spread(scores):
    """Median gap between each query's best and TOP_SPREAD-th best list score: the unit for temperatures."""
    top = np.sort(scores, axis=1)[:, -TOP_SPREAD:]
    return float(np.median(top[:, -1] - top[:, 0]))


def members_by_list(owner, nlist):
    """Sorted base ids of each list."""
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(nlist + 1))
    return [np.sort(order[bounds[c]:bounds[c + 1]]) for c in range(nlist)]
