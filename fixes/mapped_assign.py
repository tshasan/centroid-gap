#!/usr/bin/env python3
"""Store each vector where its typical query would look.

Fit a ridge map W from a stored vector to the training queries whose top-10 contain it,
then place (move or copy) every vector into the list that its predicted query W x opens
under the unchanged centroid router. Evaluated on held-out test queries.

    python fixes/mapped_assign.py deep t2i --out fixes/mapped.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spill_eval import Layout, evaluate, load, scores, top_lists


def fit_map(d, lam=1.0, chunk=200_000):
    q, t = d["train"]["queries"].astype(np.float64), d["train"]["truth"]
    dim = q.shape[1]
    xtx, xty = np.zeros((dim, dim)), np.zeros((dim, dim))
    flat_t = t.ravel()
    flat_q = np.repeat(np.arange(len(q)), t.shape[1])
    for i in range(0, len(flat_t), chunk):
        x = np.asarray(d["base"][np.sort(flat_t[i:i + chunk])], np.float64)
        order = np.argsort(flat_t[i:i + chunk], kind="stable")
        y = q[flat_q[i:i + chunk][order]]
        xtx += x.T @ x
        xty += x.T @ y
    scale = np.trace(xtx) / dim
    return np.linalg.solve(xtx + lam * scale * 1e-3 * np.eye(dim), xty).astype(np.float32)


def predicted_lists(d, W, n=2, chunk=100_000):
    N = len(d["owner"])
    top = np.empty((N, n), np.int32)
    gap = np.empty(N, np.float32)          # score(predicted best) - score(owner) under W x
    for i in range(0, N, chunk):
        xh = np.asarray(d["base"][i:i + chunk], np.float32) @ W
        s = scores(xh, d["centers"], d["metric"])
        part = np.argpartition(-s, n, axis=1)[:, :n]
        order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
        top[i:i + chunk] = np.take_along_axis(part, order, 1)
        rows = np.arange(len(xh))
        gap[i:i + chunk] = s[rows, top[i:i + chunk, 0]] - s[rows, d["owner"][i:i + chunk]]
    return top, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="+")
    ap.add_argument("--out", default="fixes/mapped.json")
    args = ap.parse_args()
    out = {}
    for w in args.workloads:
        d = load(w)
        N, nlist = len(d["owner"]), len(d["centers"])
        ev = lambda L, s="test": evaluate(L, d[s]["queries"], d[s]["truth"], d["centers"], d["metric"])
        res = {"centroids": dict(memory=1.0, **ev(Layout(d["owner"], nlist)))}
        print(w, "centroids", res["centroids"], flush=True)
        W = fit_map(d)
        top, gap = predicted_lists(d, W)
        differ = top[:, 0] != d["owner"]
        print(w, "fraction whose predicted list differs", float(differ.mean()), flush=True)
        L = Layout(top[:, 0].astype(d["owner"].dtype), nlist)
        res["move"] = dict(memory=1.0, **ev(L), val=ev(L, "val"))
        print(w, "move", res["move"], flush=True)
        idx = np.nonzero(differ)[0]
        L = Layout(d["owner"], nlist, idx, top[idx, 0])
        res["copy_all"] = dict(memory=L.memory, **ev(L), val=ev(L, "val"))
        print(w, "copy_all", res["copy_all"], flush=True)
        for f in (0.05, 0.1, 0.25):
            pick = idx[np.argsort(-gap[idx])[: int(f * N)]]
            L = Layout(d["owner"], nlist, pick, top[pick, 0])
            res[f"copy_confident_{f}"] = dict(memory=L.memory, **ev(L), val=ev(L, "val"))
            print(w, f"copy_confident_{f}", res[f"copy_confident_{f}"], flush=True)
        out[w] = res
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
