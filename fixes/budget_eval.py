#!/usr/bin/env python3
"""Headroom from per-query probe budgets, and a simple stopping rule.

budget oracle: each query gets its own probe count, chosen to minimize mean work subject
to mean recall@10 >= 0.95 (Lagrangian sweep over the per-query recall/work curves; this is
the lower convex hull, an upper bound on what any stopping rule can save).
score gap rule: probe every list whose centroid score is within tau of the best one
(relative for L2), capped at PMAX, with tau tuned on validation to reach 0.95.

    python fixes/budget_eval.py deep t2i --out fixes/budget.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spill_eval import PMAX, TARGET, Layout, load, query_guided, scores


def curves(layout, queries, truth, centers, metric):
    s = scores(queries, centers, metric)
    order = np.argsort(-s, axis=1)
    rankpos = np.empty_like(order)
    np.put_along_axis(rankpos, order, np.arange(order.shape[1])[None, :], 1)
    cover = layout.first_cover(truth, rankpos)                    # (n, 10)
    p = np.arange(1, PMAX + 1)
    rec = (cover[:, :, None] <= p[None, None, :]).mean(1)          # (n, PMAX)
    work = layout.nlist + np.cumsum(layout.sizes[order[:, :PMAX]], axis=1)
    sorted_s = np.take_along_axis(s, order[:, :PMAX], 1)
    return rec, work, sorted_s


def budget_oracle(rec, work):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2
        pick = (rec - lam * work).argmax(1)
        r = rec[np.arange(len(rec)), pick].mean()
        if r >= TARGET:
            lo = lam
        else:
            hi = lam
    pick = (rec - lo * work).argmax(1)
    rows = np.arange(len(rec))
    return dict(recall=float(rec[rows, pick].mean()), work=float(work[rows, pick].mean()),
                mean_probes=float((pick + 1).mean()))


def gap_rule(rec, work, ss, metric, tau):
    if metric == "l2":
        # scores are 2<q,c> - |c|^2 = |q|^2 - d^2 up to the query constant; use gaps directly
        keep = ss >= ss[:, :1] - tau
    else:
        keep = ss >= ss[:, :1] - tau
    pick = keep.sum(1) - 1
    rows = np.arange(len(rec))
    return float(rec[rows, pick].mean()), float(work[rows, pick].mean()), float((pick + 1).mean())


def tune_gap(val, test, metric):
    rv, wv, sv = val
    spread = float(np.median(sv[:, 0] - sv[:, 50]))
    lo, hi = 0.0, 4 * spread
    for _ in range(50):
        mid = (lo + hi) / 2
        if gap_rule(rv, wv, sv, metric, mid)[0] >= TARGET:
            hi = mid
        else:
            lo = mid
    r, w, p = gap_rule(*test, metric, hi)
    return dict(tau=hi, recall=r, work=w, mean_probes=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="+")
    ap.add_argument("--out", default="fixes/budget.json")
    args = ap.parse_args()
    out = {}
    for w in args.workloads:
        d = load(w)
        nlist = len(d["centers"])
        layouts = {"centroids": Layout(d["owner"], nlist)}
        v, l, _ = query_guided(d, 0.25, 8)
        layouts["query_guided"] = Layout(d["owner"], nlist, v, l)
        res = {}
        for name, L in layouts.items():
            test = curves(L, d["test"]["queries"], d["test"]["truth"], d["centers"], d["metric"])
            val = curves(L, d["val"]["queries"], d["val"]["truth"], d["centers"], d["metric"])
            rec, work, _ = test
            p = np.nonzero(rec.mean(0) >= TARGET)[0][0]
            res[name] = dict(memory=L.memory, fixed=dict(probes=int(p + 1), work=float(work[:, p].mean())),
                             budget_oracle=budget_oracle(rec, work),
                             gap_rule=tune_gap(val, test, d["metric"]))
            print(w, name, json.dumps(res[name]), flush=True)
        out[w] = res
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
