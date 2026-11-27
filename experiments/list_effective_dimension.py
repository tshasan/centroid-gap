"""Measure the effective dimension of each frozen IVF list (before Gate A; paper Section 7).

Best-tip routing scores a list by the support function of m tips. Polytope
approximation (Dudley; Bronshteyn-Ivanov) gives worst-case relative error of order
m^(-2/(d-1)), tight for round bodies, so tips can only help where lists are
low-dimensional. This script reports, per list, the PCA dimension holding 90% of
the variance and the participation ratio (sum λ)^2 / sum λ^2, then the worst-case
factor m^(-2/(d-1)) at the median of each.

Run after experiments/ivf_workload_index.py:
    python experiments/list_effective_dimension.py DATA RUN deep t2i --output results/list-effective-dimension.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .frozen_index import members_by_list
else:
    from frozen_index import members_by_list

TIPS = (1, 2, 4, 8, 16, 64, 256)
VARIANCE = 0.90


def spectra(base, owner, nlist):
    """Eigenvalues (descending) of each list's covariance."""
    out = np.empty((nlist, base.shape[1]))
    for label, ids in enumerate(members_by_list(owner, nlist)):
        rows = np.asarray(base[ids], dtype=np.float64)
        rows -= rows.mean(axis=0)
        out[label] = np.linalg.eigvalsh(rows.T @ rows / max(len(rows) - 1, 1))[::-1].clip(min=0)
    return out


def summarize(eig):
    total = eig.sum(axis=1)
    d90 = (np.cumsum(eig, axis=1) < VARIANCE * total[:, None]).sum(axis=1) + 1
    pr = total**2 / (eig**2).sum(axis=1)
    pooled = eig.mean(axis=0)
    def q(x):
        return {p: float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)}
    def factor(dim):
        return {m: float(m ** (-2 / (dim - 1))) if dim > 1 else 0.0 for m in TIPS}
    return {
        "ambient_dim": int(eig.shape[1]),
        "d90_percentiles": q(d90),
        "participation_ratio_percentiles": q(pr),
        "pooled_within_list_participation_ratio": float(pooled.sum()**2 / (pooled**2).sum()),
        "worst_case_error_factor_at_median_d90": factor(float(np.median(d90))),
        "worst_case_error_factor_at_median_pr": factor(float(np.median(pr))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("workloads", nargs="+", choices=["deep", "t2i"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    for w in args.workloads:
        base = np.load(args.data / f"{w}-base.npy", mmap_mode="r")
        owner = np.load(args.run / w / "owner.npy")
        nlist = int(owner.max()) + 1
        if len(owner) != len(base):
            raise ValueError(f"{w}: owner has {len(owner)} rows, base has {len(base)}")
        result[w] = summarize(spectra(base, owner, nlist))
        # Routing only needs the support function in query directions, so also
        # report the dimension of the training-query distribution.
        q = np.asarray(np.load(args.data / f"{w}-fit.npy", mmap_mode="r"), dtype=np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        lam = np.linalg.eigvalsh(np.cov(q, rowvar=False))[::-1].clip(min=0)
        second = (q.T @ q / len(q))
        mu = np.linalg.eigvalsh(second)[::-1].clip(min=0)
        result[w]["fit_query_direction_participation_ratio"] = {
            "centered": float(lam.sum()**2 / (lam**2).sum()),
            "uncentered": float(mu.sum()**2 / (mu**2).sum()),
            "d90_uncentered": int((np.cumsum(mu) < VARIANCE * mu.sum()).sum() + 1),
        }
        print(w, json.dumps(result[w], indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
