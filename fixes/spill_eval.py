#!/usr/bin/env python3
"""Evaluate storage fixes for IVF routing on the cached million-vector indexes.

Work = nlist routing scores + every scanned stored copy (duplicates count).
Recall@10 counts a true neighbor once if any of its copies sits in a probed list.
All variants keep the k-means lists and centroids of the cached index; they only
add copies of vectors to extra lists.

    python fixes/spill_eval.py deep t2i --out fixes/results.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache/centroid-gap"
TARGET = 0.95
PMAX = 200


def load(workload):
    run, data = CACHE / "run" / workload, CACHE / "data"
    meta = json.loads((run / "index.json").read_text())
    base = np.load(data / f"{workload}-base.npy", mmap_mode="r")
    split = lambda name: np.load(run / f"{name}.npz")
    extra = np.load(CACHE / "gateb" / workload / "extra.npz")
    return dict(metric=meta["metric"], base=base, centers=np.load(run / "centers.npy"),
                owner=np.load(run / "owner.npy"), test=split("test"), val=split("val"),
                train=dict(queries=extra["queries"], truth=extra["truth"]))


def scores(x, centers, metric):
    s = x @ centers.T
    if metric == "l2":
        s = 2 * s - (centers ** 2).sum(1)[None, :]  # ranks like -||x - c||^2
    return s


def top_lists(x, centers, metric, n, chunk=50_000):
    out = np.empty((len(x), n), dtype=np.int32)
    for i in range(0, len(x), chunk):
        s = scores(np.asarray(x[i:i + chunk], dtype=np.float32), centers, metric)
        part = np.argpartition(-s, n, axis=1)[:, :n]
        order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
        out[i:i + chunk] = np.take_along_axis(part, order, 1)
    return out


def soar_second(x, centers, owner, lam=1.0, cand=20, chunk=10_000):
    """SOAR: second list minimizes ||x-c'||^2 + lam * <x-c1, x-c'>^2 / ||x-c1||^2."""
    out = np.empty(len(x), dtype=np.int32)
    for i in range(0, len(x), chunk):
        xb = np.asarray(x[i:i + chunk], dtype=np.float32)
        top = top_lists(xb, centers, "l2", cand)
        r = xb - centers[owner[i:i + chunk]]
        rn = (r ** 2).sum(1, keepdims=True) + 1e-12
        rp = xb[:, None, :] - centers[top]                       # (b, cand, d)
        loss = (rp ** 2).sum(2) + lam * np.einsum("bd,bcd->bc", r, rp) ** 2 / rn
        loss[top == owner[i:i + chunk, None]] = np.inf
        out[i:i + chunk] = top[np.arange(len(xb)), loss.argmin(1)]
    return out


class Layout:
    """Primary owner of every vector plus extra (vector, list) copies."""

    def __init__(self, owner, nlist, extra_vec=None, extra_list=None):
        self.owner, self.nlist = owner, nlist
        self.ev = np.zeros(0, np.int64) if extra_vec is None else np.asarray(extra_vec, np.int64)
        self.el = np.zeros(0, np.int32) if extra_list is None else np.asarray(extra_list, np.int32)
        self.sizes = np.bincount(owner, minlength=nlist) + np.bincount(self.el, minlength=nlist)
        self.memory = (len(owner) + len(self.ev)) / len(owner)

    def first_cover(self, truth, rankpos):
        """Rank (1-based) at which each true neighbor is first covered, per query."""
        cover = np.take_along_axis(rankpos, self.owner[truth], 1) + 1
        if len(self.ev):
            order = np.argsort(self.ev)
            ev, el = self.ev[order], self.el[order]
            lo = np.searchsorted(ev, truth, "left")
            hi = np.searchsorted(ev, truth, "right")
            for q, t in zip(*np.nonzero(hi > lo)):
                lists = el[lo[q, t]:hi[q, t]]
                cover[q, t] = min(cover[q, t], rankpos[q, lists].min() + 1)
        return cover


def evaluate(layout, queries, truth, centers, metric):
    s = scores(queries, centers, metric)
    order = np.argsort(-s, axis=1)
    rankpos = np.empty_like(order)
    np.put_along_axis(rankpos, order, np.arange(order.shape[1])[None, :], 1)
    cover = layout.first_cover(truth, rankpos)
    p = np.arange(1, PMAX + 1)
    recall = np.array([(cover <= pp).mean() for pp in p])
    work = layout.nlist + np.cumsum(layout.sizes[order[:, :PMAX]], axis=1).mean(0)
    hit = np.nonzero(recall >= TARGET)[0]
    if not len(hit):
        return dict(probes=None, work=None, work_interp=None)
    i = hit[0]
    if i == 0:
        interp = work[0]
    else:
        f = (TARGET - recall[i - 1]) / (recall[i] - recall[i - 1])
        interp = work[i - 1] + f * (work[i] - work[i - 1])
    return dict(probes=int(p[i]), recall=float(recall[i]), work=float(work[i]), work_interp=float(interp))


def oracle(layout, queries, truth, centers, metric):
    """Oracle: open lists by number of true neighbors held (primary layout only)."""
    held = np.zeros((len(queries), layout.nlist))
    np.add.at(held, (np.repeat(np.arange(len(queries)), truth.shape[1]), layout.owner[truth].ravel()), 1)
    order = np.lexsort((layout.sizes[None, :].repeat(len(queries), 0), -held), axis=1)
    rankpos = np.empty_like(order)
    np.put_along_axis(rankpos, order, np.arange(order.shape[1])[None, :], 1)
    cover = np.take_along_axis(rankpos, layout.owner[truth], 1) + 1
    p = np.arange(1, PMAX + 1)
    recall = np.array([(cover <= pp).mean() for pp in p])
    work = layout.nlist + np.cumsum(layout.sizes[order[:, :PMAX]], axis=1).mean(0)
    i = np.nonzero(recall >= TARGET)[0][0]
    return dict(probes=int(p[i]), recall=float(recall[i]), work=float(work[i]))


def query_guided(d, budget, p0, train_rows=None):
    """Copy each missed training neighbor into the best list of the training query that
    missed it. Keep the most-voted (vector, list) pairs up to budget * N copies."""
    q, t = d["train"]["queries"], d["train"]["truth"]
    if train_rows is not None:
        q, t = q[train_rows], t[train_rows]
    tops = top_lists(q, d["centers"], d["metric"], p0)
    own = d["owner"][t]                                          # (n, 10)
    covered = (own[:, :, None] == tops[:, None, :]).any(2)
    qi, ti = np.nonzero(~covered)
    vec, lst = t[qi, ti], tops[qi, 0]
    key = vec.astype(np.int64) * 4096 + lst
    keys, votes = np.unique(key, return_counts=True)
    keep = keys[np.argsort(-votes, kind="stable")[: int(budget * len(d["owner"]))]]
    return keep // 4096, (keep % 4096).astype(np.int32), int(len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="+")
    ap.add_argument("--out", default="fixes/results.json")
    args = ap.parse_args()
    results = {}
    for w in args.workloads:
        t0 = time.time()
        d = load(w)
        N, nlist = len(d["owner"]), len(d["centers"])
        test, val = d["test"], d["val"]
        ev = lambda L, s=test: evaluate(L, s["queries"], s["truth"], d["centers"], d["metric"])
        res = {}
        base = Layout(d["owner"], nlist)
        res["centroids"] = dict(memory=1.0, **ev(base))
        res["oracle"] = dict(memory=1.0, **oracle(base, test["queries"], test["truth"], d["centers"], d["metric"]))
        print(w, "baseline", res["centroids"], "oracle", res["oracle"], flush=True)

        top2 = top_lists(d["base"], d["centers"], d["metric"], 2)
        second = np.where(top2[:, 0] == d["owner"], top2[:, 1], top2[:, 0])
        s2 = None
        # margin between best and second centroid score, for selective spilling
        idx = np.arange(N)
        sc = np.empty(N, np.float32)
        for i in range(0, N, 100_000):
            xb = np.asarray(d["base"][i:i + 100_000], np.float32)
            s = scores(xb, d["centers"], d["metric"])
            sc[i:i + 100_000] = s[np.arange(len(xb)), d["owner"][i:i + 100_000]] - s[np.arange(len(xb)), second[i:i + 100_000]]
        res["spill_all"] = dict(memory=2.0, **ev(Layout(d["owner"], nlist, idx, second)))
        print(w, "spill_all", res["spill_all"], flush=True)
        for f in (0.1, 0.25, 0.5):
            pick = np.argsort(sc)[: int(f * N)]
            L = Layout(d["owner"], nlist, pick, second[pick])
            res[f"spill_margin_{f}"] = dict(memory=L.memory, **ev(L))
            print(w, f"spill_margin_{f}", res[f"spill_margin_{f}"], flush=True)
        soar = soar_second(d["base"], d["centers"], d["owner"])
        res["soar_all"] = dict(memory=2.0, **ev(Layout(d["owner"], nlist, idx, soar)))
        print(w, "soar_all", res["soar_all"], flush=True)
        for p0 in (4, 8):
            for f in (0.05, 0.1, 0.25):
                v, l, cand = query_guided(d, f, p0)
                L = Layout(d["owner"], nlist, v, l)
                r = dict(memory=L.memory, candidates=cand, **ev(L))
                r["val"] = ev(L, val)
                res[f"query_guided_p{p0}_{f}"] = r
                print(w, f"query_guided_p{p0}_{f}", r, flush=True)
        res["seconds"] = time.time() - t0
        results[w] = res
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
