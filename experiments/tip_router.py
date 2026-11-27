"""m-tip router on the frozen 1M IVF lists (Gate A, paper Section 7).

Each list keeps m database-only tips and is scored by its best tip: the largest
inner product (t2i) or the smallest squared distance (deep). Lists, answer keys and
splits are those written by experiments/ivf_workload_index.py; nothing is
re-clustered. Recall@10 at p probes follows exactly from the list ranks of the
true neighbors, so no vectors are scanned. Work per query = nlist·m tip scores +
the sizes of the p probed lists.

Tip kinds:
  centroid  the stored Faiss centers (m = 1 baseline)
  sub       m k-means centers fitted inside each list
  far       the list center plus m-1 farthest-point picks from the list

    python experiments/tip_router.py DATA RUN t2i --tips 1 2 4 8 --output results/tip-router-t2i.json
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

if __package__:
    from .frozen_index import load_frozen_index, load_split, members_by_list, metric_for
    from .routing_stats import whole_probe_summary
else:
    from frozen_index import load_frozen_index, load_split, members_by_list, metric_for
    from routing_stats import whole_probe_summary

TARGET = 0.95
TIP_KINDS = ("sub", "far")


def make_tips(base, members, centers, kind, m, seed):
    """(tips, list label of each tip) for `m` tips of the given kind per list."""
    if kind == "centroid" or m == 1:
        return centers.astype(np.float32), np.arange(len(centers))
    if kind not in TIP_KINDS:
        raise ValueError(f"unknown tip kind {kind!r}; expected centroid or one of {TIP_KINDS}")
    tips, label = [], []
    for c, ids in enumerate(members):
        x = np.asarray(base[ids], dtype=np.float32)
        if kind == "sub":
            km = faiss.Kmeans(x.shape[1], min(m, len(x)), niter=20, seed=seed + c, verbose=False)
            km.train(x)
            t = km.centroids
        else:  # far: greedy farthest-point traversal seeded at the stored center
            chosen = [centers[c].astype(np.float32)]
            d2 = ((x - chosen[0]) ** 2).sum(axis=1)
            for _ in range(min(m, len(x) + 1) - 1):
                i = int(np.argmax(d2))
                chosen.append(x[i])
                d2 = np.minimum(d2, ((x - x[i]) ** 2).sum(axis=1))
            t = np.stack(chosen)
        tips.append(t)
        label.extend([c] * len(t))
    return np.concatenate(tips).astype(np.float32), np.asarray(label)


def list_scores(queries, tips, label, nlist, metric):
    """Score of each list = score of its best tip (higher is better)."""
    s = queries @ tips.T
    if metric == "l2":
        s = 2 * s - (tips ** 2).sum(axis=1)[None, :]  # larger is closer
    order = np.argsort(label, kind="stable")
    starts = np.searchsorted(label[order], np.arange(nlist))
    return np.maximum.reduceat(s[:, order], starts, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("workload", choices=["deep", "t2i"])
    parser.add_argument("--tips", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--kinds", nargs="+", choices=TIP_KINDS, default=list(TIP_KINDS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tips) < 1:
        parser.error("--tips must be positive")
    w = args.workload
    metric = metric_for(w)
    index = load_frozen_index(args.run / w)
    base = np.load(args.data / f"{w}-base.npy", mmap_mode="r")
    if len(base) != len(index.owner):
        raise ValueError(f"{w}: base has {len(base)} rows but owner.npy has {len(index.owner)}")
    members = members_by_list(index.owner, index.nlist)
    splits = {s: load_split(args.run / w / f"{s}.npz") for s in ("val", "test")}
    result = {"workload": w, "metric": metric, "target_recall_at_10": TARGET, "rows": []}
    arms = [("centroid", 1)] + [(k, m) for k in args.kinds for m in args.tips if m > 1]
    for kind, m in arms:
        tips, label = make_tips(base, members, index.centers, kind, m, args.seed)
        row = {"kind": kind, "m": m}
        for s, (q, truth) in splits.items():
            scores = list_scores(q.astype(np.float32), tips, label, index.nlist, metric)
            row[s] = whole_probe_summary(scores, truth, index.owner, index.sizes, len(tips), TARGET)
        result["rows"].append(row)
        print(f"{w} {kind:8s} m={m:<3d} val {row['val']['probes']:4d} probes {row['val']['work']:9.0f} | "
              f"test {row['test']['probes']:4d} probes {row['test']['work']:9.0f} work", flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
