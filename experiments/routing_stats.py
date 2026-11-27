"""Recall and work of a router, read from the list ranks of the true neighbors (no vectors scanned).

In IVF-Flat the scan inside a list is exact, so recall@k at p probes is the share of true neighbors
whose list ranks in the top p. Work per query = routing scores + the sizes of the probed lists.

Two readings of "work at 95% recall":
  * whole probe: the first whole probe count that reaches the target. It moves in steps of about
    one list (roughly 3% of the work on the million-vector indexes), so two routers compared this
    way can differ by a step that says nothing about the routers;
  * interpolated: work linearly interpolated to exactly the target.
`compare` reports both, with paired bootstrap intervals over queries (both arms resampled together).
"""

import numpy as np

MAX_PROBES = 120


def _truth_ranks(ranked, truth, owner):
    """Zero-based rank of each true neighbor's list, given each query's lists in routing order."""
    pos = np.empty_like(ranked)
    pos[np.arange(len(ranked))[:, None], ranked] = np.arange(ranked.shape[1])[None, :]
    return np.take_along_axis(pos, owner[truth].astype(np.int64), axis=1)


def mean_curve(scores, truth, owner, sizes, nscores):
    """Recall and mean work at p = 1..nlist probes. scores: (n, nlist), higher is better.

    nscores is the routing cost per query (number of centroids or tips scored)."""
    ranked = np.argsort(-scores, axis=1)
    rank = _truth_ranks(ranked, truth, owner)
    recall = np.bincount(rank.ravel(), minlength=scores.shape[1]).cumsum() / truth.size
    work = nscores + sizes[ranked].cumsum(axis=1).mean(axis=0)
    return recall, work


def whole_probe_summary(scores, truth, owner, sizes, nscores, target):
    """Work at the first whole probe count reaching `target` recall, with the probe count and recall."""
    return summarize_curve(*mean_curve(scores, truth, owner, sizes, nscores), nscores, target)


def summarize_curve(recall, work, nscores, target):
    """whole_probe_summary from a precomputed mean_curve."""
    p = int(np.searchsorted(recall, target)) + 1
    if p > len(recall):
        raise ValueError(f"recall never reaches {target:g} (best {recall[-1]:.4f})")
    return {"probes": p, "recall": float(recall[p - 1]), "work": float(work[p - 1]),
            "scanned": float(work[p - 1] - nscores), "tip_scores": int(nscores),
            "recall_at_1_probe": float(recall[0])}


def per_query(scores, truth, owner, sizes, max_probes=MAX_PROBES):
    """Per-query hit counts and work for p = 1..max_probes probes.

    scores: (n, nlist), higher is better. truth: (n, k) stored ids. owner[id] = list of that id.
    Returns hits (n, P) = true neighbors in the top-p lists, work (n, P) = nlist + scanned vectors."""
    n, nlist = scores.shape
    P = min(max_probes, nlist)
    ranked = np.argsort(-scores, axis=1, kind="stable")
    rank = _truth_ranks(ranked, truth, owner)
    hits = np.stack([(rank < p).sum(axis=1) for p in range(1, P + 1)], axis=1)
    work = nlist + sizes[ranked[:, :P]].cumsum(axis=1)
    return hits, work


def mean_recall_work(hits, work, k):
    """Recall@k and mean work at each probe count, from per_query output."""
    return hits.sum(axis=0) / (len(hits) * k), work.mean(axis=0)


def interp(recall, work, target, nlist):
    """Mean work linearly interpolated to exactly `target` recall; nan if the curve never gets there."""
    r = np.concatenate([[0.0], recall])
    w = np.concatenate([[float(nlist)], work])
    p = int(np.searchsorted(r, target))
    if p >= len(r):
        return float("nan")
    f = (target - r[p - 1]) / max(r[p] - r[p - 1], 1e-12)
    return float(w[p - 1] + f * (w[p] - w[p - 1]))


def step(recall, work, target):
    """(probes, work) at the first whole probe count reaching `target`."""
    p = int(np.searchsorted(recall, target))
    return p + 1, float(work[p]) if p < len(work) else float("nan")


def compare(base, other, k, target, nlist, boots=1000, seed=0):
    """Interpolated and whole-probe change of `other` against `base`, with paired bootstrap 95% intervals.

    base, other: (hits, work) from per_query on the same queries."""
    rng = np.random.default_rng(seed)
    (h0, w0), (h1, w1) = base, other
    if h0.shape != h1.shape:
        raise ValueError(f"base and other cover different queries: {h0.shape} vs {h1.shape}")
    r0, c0 = mean_recall_work(h0, w0, k)
    r1, c1 = mean_recall_work(h1, w1, k)
    i0, i1 = interp(r0, c0, target, nlist), interp(r1, c1, target, nlist)
    (p0, s0), (p1, s1) = step(r0, c0, target), step(r1, c1, target)
    n = len(h0)
    di, ds = [], []
    for _ in range(boots):
        i = rng.integers(n, size=n)
        a0, b0 = mean_recall_work(h0[i], w0[i], k)
        a1, b1 = mean_recall_work(h1[i], w1[i], k)
        di.append(interp(a1, b1, target, nlist) / interp(a0, b0, target, nlist) - 1)
        ds.append(step(a1, b1, target)[1] / step(a0, b0, target)[1] - 1)
    lo_i, hi_i = np.nanpercentile(di, [2.5, 97.5])
    lo_s, hi_s = np.nanpercentile(ds, [2.5, 97.5])
    return {
        "queries": n,
        "interpolated": {"base": i0, "other": i1, "change": i1 / i0 - 1, "ci95": [float(lo_i), float(hi_i)]},
        "whole_probe": {"base_probes": p0, "base": s0, "other_probes": p1, "other": s1,
                        "change": s1 / s0 - 1, "ci95": [float(lo_s), float(hi_s)]},
    }
