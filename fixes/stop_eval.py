#!/usr/bin/env python3
"""Query-time stopping rules on the cached indexes (centroid probe order unchanged).

ball rule: after each probed list, compute for every unprobed list the best score any of
its members could have, using the list radius scaled by beta, and stop once none can beat
the current 10th-best found score. beta is tuned on validation to reach 95% recall@10.
Also reports the per-query recall spread under the fixed budget and the budget oracle.

    python fixes/stop_eval.py deep t2i --out fixes/stop.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from budget_eval import budget_oracle, curves
from spill_eval import TARGET, Layout, load, scores


def radii(d, q=1.0, chunk=200_000):
    N, nlist = len(d["owner"]), len(d["centers"])
    dist = np.empty(N, np.float32)
    for i in range(0, N, chunk):
        x = np.asarray(d["base"][i:i + chunk], np.float32)
        dist[i:i + chunk] = np.linalg.norm(x - d["centers"][d["owner"][i:i + chunk]], axis=1)
    order = np.argsort(d["owner"], kind="stable")
    bounds = np.searchsorted(d["owner"][order], np.arange(nlist + 1))
    return np.array([np.quantile(dist[order[bounds[j]:bounds[j + 1]]], q) for j in range(nlist)], np.float32)


def stop_profiles(d, split, R, pmax):
    """For each query: scores of all lists, order, and the running 10th-best found score."""
    qs = d[split]["queries"]
    owner_order = np.argsort(d["owner"], kind="stable")
    nlist = len(d["centers"])
    bounds = np.searchsorted(d["owner"][owner_order], np.arange(nlist + 1))
    S = scores(qs, d["centers"], d["metric"])
    order = np.argsort(-S, axis=1)[:, :pmax]
    kth = np.empty((len(qs), pmax), np.float32)
    base = d["base"]
    for i, q in enumerate(qs):
        ids = np.concatenate([owner_order[bounds[j]:bounds[j + 1]] for j in order[i]])
        x = base[ids]
        sc = -((x - q) ** 2).sum(1) if d["metric"] == "l2" else x @ q
        best = np.full(10, -np.inf, np.float32)
        start = 0
        for p, j in enumerate(order[i]):
            n = bounds[j + 1] - bounds[j]
            best = np.sort(np.concatenate([best, sc[start:start + n]]))[-10:]
            start += n
            kth[i, p] = best[0]
    return S, order, kth


def ball_stop(S, order, kth, R, metric, qnorm, beta):
    """First probe count p at which no unprobed list can beat the current 10th best."""
    n, pmax = kth.shape
    pick = np.full(n, pmax - 1)
    for i in range(n):
        if metric == "l2":
            # S = |q|^2 - d^2 ; best member distance >= max(0, d - beta R)
            d = np.sqrt(np.maximum(qnorm[i] ** 2 - S[i], 0))
            best_possible = -(np.maximum(d - beta * R, 0) ** 2)     # best possible score -dist^2
        else:
            best_possible = S[i] + beta * qnorm[i] * R
        a = best_possible[order[i]]
        suffix = np.maximum.accumulate(a[::-1])[::-1]          # max(a[p:])
        rest_max = np.append(suffix[1:], -np.inf)              # max(a[p+1:])
        tail = np.ones(len(best_possible), bool)
        tail[order[i]] = False
        tail_max = best_possible[tail].max() if tail.any() else -np.inf
        ok = np.nonzero(np.maximum(rest_max, tail_max) <= kth[i])[0]
        if len(ok):
            pick[i] = ok[0]
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="+")
    ap.add_argument("--out", default="fixes/stop.json")
    args = ap.parse_args()
    out = {}
    for w in args.workloads:
        d = load(w)
        d["base"] = np.ascontiguousarray(d["base"], dtype=np.float32)
        nlist = len(d["centers"])
        L = Layout(d["owner"], nlist)
        pmax = 80 if w == "t2i" else 50
        res = {}
        rec_t, work_t, _ = curves(L, d["test"]["queries"], d["test"]["truth"], d["centers"], d["metric"])
        p = np.nonzero(rec_t.mean(0) >= TARGET)[0][0]
        fixed = rec_t[:, p]
        bo = budget_oracle(rec_t, work_t)
        res["fixed_recall_spread"] = dict(p10=float(np.quantile(fixed, .1)), share_below_half=float((fixed < .5).mean()))
        lam_pick = None
        # recompute budget-oracle picks to report spread
        lo, hi = 0.0, 1.0
        for _ in range(60):
            lam = (lo + hi) / 2
            pk = (rec_t - lam * work_t).argmax(1)
            lo, hi = (lam, hi) if rec_t[np.arange(len(pk)), pk].mean() >= TARGET else (lo, lam)
        pk = (rec_t - lo * work_t).argmax(1)
        r_bo = rec_t[np.arange(len(pk)), pk]
        res["budget_oracle"] = dict(bo, p10=float(np.quantile(r_bo, .1)), share_below_half=float((r_bo < .5).mean()))
        print(w, "spread", res["fixed_recall_spread"], res["budget_oracle"], flush=True)
        for q in (0.9, 1.0):
            R = radii(d, q)
            prof = {s: stop_profiles(d, s, R, pmax) for s in ("val", "test")}
            qn = {s: np.linalg.norm(d[s]["queries"], axis=1) for s in ("val", "test")}
            rec_v, work_v, _ = curves(L, d["val"]["queries"], d["val"]["truth"], d["centers"], d["metric"])
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                pk = ball_stop(*prof["val"], R, d["metric"], qn["val"], mid)
                lo, hi = (lo, mid) if rec_v[np.arange(len(pk)), pk].mean() >= TARGET else (mid, hi)
            beta = hi
            pk = ball_stop(*prof["test"], R, d["metric"], qn["test"], beta)
            rows = np.arange(len(pk))
            r = rec_t[rows, pk]
            res[f"ball_q{q}"] = dict(beta=beta, recall=float(r.mean()), work=float(work_t[rows, pk].mean()),
                                     mean_probes=float((pk + 1).mean()), p10=float(np.quantile(r, .1)),
                                     share_below_half=float((r < .5).mean()))
            print(w, f"ball_q{q}", res[f"ball_q{q}"], flush=True)
        out[w] = res
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
