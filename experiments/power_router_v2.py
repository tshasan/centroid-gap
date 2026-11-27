"""Gate B v2: power-diagram router with more data, a cost-aware loss and an image-shift guard.

Same router as experiments/power_router.py: the stored centers plus one scalar per list,
    score_c(q) = <q, c> + b_c            (t2i)
    score_c(q) = 2<q, c> - |c|^2 + b_c   (deep)
so serving adds no vector work. Work at 95% recall@10 = 1,000 center scores + the
sizes of the probed lists, at the smallest fixed probe count reaching 95%
(routing_stats.whole_probe_summary).

Loss. For fit query i with label weights T_ic (the share of its exact top-10 held by
list c) and logits z_c = (score_c + b_c) / tau,
    L_i = -sum_c T_ic z_c + log sum_c w_c exp(z_c),   w_c = (size_c / mean size)^gamma.
gamma = 0 is the softmax cross-entropy of power_router.py. gamma = 1 makes the
denominator a smoothed count of the vectors in lists that outrank the labels, so each
list is charged its size; at the optimum the biases shift by -tau*gamma*log(size), which
ranks lists by fitted probability per scanned vector. An L2 penalty (lam / tau) |b|^2/2
pulls the biases toward zero. Image queries (t2i only) can be mixed into the fit with a
total weight share alpha. The softmax runs over the top 96 lists by stored-center score
plus the label lists; the rest carry negligible mass at these temperatures.
tools/ivf_tune ports this loss.

Selection. Every arm is scored on text validation. On t2i an arm is eligible only if its
work on 3,000 held-out image-validation queries is no higher than the centroids'. The
pick is reported on test and on the 1,993 image-shift queries, which play no part in
fitting or selection. Two selection keys are reported: "work", the repo standard (work at
the smallest fixed probe count reaching 95%, a step function of the probe count), and
"work_interp", the same work linearly interpolated to exactly 95% recall@10, which
removes the one-probe jumps between validation and test. "work" is primary.

Arm grids: "full" (learning curve, gamma, lam, image mixing and fixed size priors) and
"quick" (16 arms on the full fit set; the settings the full grid settled on seed 2026).

    python experiments/power_router_v2.py ~/.cache/centroid-gap/run ~/.cache/centroid-gap/gateb \
        t2i --seed 2026 --output results/power-router-v2-t2i.json
"""

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

if __package__:
    from .frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from .routing_stats import interp, mean_curve, summarize_curve
else:
    from frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from routing_stats import interp, mean_curve, summarize_curve

CAND = 96
IMAGE_VAL = 3000
TARGET = 0.95
SPREAD_ROWS = 16384  # temperatures are measured on the original fit queries only
TAUS = (0.02, 0.05, 0.1, 0.2)
GRID_TAUS = TAUS + (0.5,)
QUICK_TAUS = (0.1, 0.2, 0.3, 0.5)
ARM_KEYS = ("kind", "n", "tau_frac", "gamma", "lam", "alpha")


class Grid(str, Enum):
    FULL = "full"
    QUICK = "quick"


@dataclass
class Candidates:
    """Per fit query: sorted unique candidate lists, their raw scores and their label weights."""
    cand: np.ndarray
    raw: np.ndarray
    T: np.ndarray


@dataclass
class FitData:
    """Everything an arm needs, built once by prepare() and shared read-only with worker processes."""
    owner: np.ndarray
    sizes: np.ndarray
    spread: float
    n_orig: int
    fit: Candidates
    evals: dict                         # split -> (scores, truth)
    image_fit: Optional[Candidates] = None

    @property
    def n_fit(self):
        return len(self.fit.cand)


def candidates(S, labels):
    """Sorted unique candidate lists per query, their raw scores and label weights."""
    n = len(S)
    top = np.argpartition(-S, CAND, axis=1)[:, :CAND]
    cand = np.sort(np.concatenate([top, labels], axis=1), axis=1)
    first = np.ones(cand.shape, dtype=bool)
    first[:, 1:] = cand[:, 1:] != cand[:, :-1]
    raw = np.where(first, np.take_along_axis(S, cand, axis=1), -np.inf).astype(np.float32)
    T = np.zeros(cand.shape, dtype=np.float32)
    for s in range(0, n, 16384):
        e = slice(s, s + 16384)
        T[e] = (cand[e, :, None] == labels[e, None, :]).sum(-1) / labels.shape[1]
    T *= first
    return Candidates(cand.astype(np.int32), raw, T)


def fit(data, cfg):
    """Full-batch Adam on the weighted-softmax loss; returns biases with mean zero."""
    n = cfg["n"]
    cand, raw, T = data.fit.cand[:n], data.fit.raw[:n], data.fit.T[:n]
    qw = np.ones(len(cand), dtype=np.float32)
    if cfg["alpha"] > 0:
        img = data.image_fit
        cand, raw, T = np.concatenate([cand, img.cand]), np.concatenate([raw, img.raw]), np.concatenate([T, img.T])
        qw = np.concatenate([qw * (1 - cfg["alpha"]) / qw.sum(),
                             np.full(len(img.cand), cfg["alpha"] / len(img.cand), dtype=np.float32)])
    qw = (qw / qw.sum()).astype(np.float32)[:, None]
    tau = cfg["tau_frac"] * data.spread
    logw = (cfg["gamma"] * np.log(data.sizes / data.sizes.mean())).astype(np.float32)
    k = len(data.sizes)
    b = np.zeros(k)
    m = np.zeros(k)
    v = np.zeros(k)
    lr, l2 = 0.05 * tau, cfg["lam"] / tau
    flat = cand.ravel()
    for t in range(1, cfg["steps"] + 1):
        z = (raw + b[cand].astype(np.float32)) / tau + logw[cand]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = np.bincount(flat, weights=((p - T) * qw).ravel(), minlength=k) / tau + l2 * b
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        b = b - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
    return b - b.mean()


def summarize(data, scores, truth):
    """Whole-probe summary at 95% recall@10, plus work interpolated to exactly 95%."""
    k = len(data.sizes)
    recall, work = mean_curve(scores, truth, data.owner, data.sizes, k)
    out = summarize_curve(recall, work, k, TARGET)
    out["work_interp"] = interp(recall, work, TARGET, k)
    return out


def evaluate(data, b):
    return {s: summarize(data, S + b, truth) for s, (S, truth) in data.evals.items()}


def run_arm(data, cfg):
    if cfg["kind"] == "size_prior":  # no learning: b = -tau*gamma*log(size)
        tau = cfg["tau_frac"] * data.spread
        b = -tau * cfg["gamma"] * np.log(data.sizes / data.sizes.mean())
    else:
        b = fit(data, cfg)
    row = dict(cfg, bias_std_over_spread=float(b.std() / data.spread), **evaluate(data, b))
    print(" ".join(f"{k}={row[k]}" for k in ARM_KEYS),
          " ".join(f"{s} {row[s]['work']:.0f}" for s in data.evals), flush=True)
    return row


_WORKER_DATA = None  # set once in each worker process by _init_worker


def _init_worker(data):
    global _WORKER_DATA
    _WORKER_DATA = data


def _run_arm_in_worker(cfg):
    return run_arm(_WORKER_DATA, cfg)


def prepare(run_dir, gate_dir, w, index_dir):
    index = load_frozen_index(index_dir)
    owner = index.owner
    sizes = index.sizes.astype(np.float64)
    centers = index.centers.astype(np.float64)
    metric = metric_for(w)
    sets = {s: load_split(run_dir / w / f"{s}.npz") for s in ("fit", "val", "test")}
    eq, et = load_split(gate_dir / w / "extra.npz")
    fq = np.concatenate([sets["fit"][0], eq])
    ft = np.concatenate([sets["fit"][1], et])

    def score(q):
        return centroid_scores(np.asarray(q, dtype=np.float64), centers, metric).astype(np.float32)

    S_fit = score(fq)
    spread = score_spread(S_fit[:SPREAD_ROWS])
    fit_cands = candidates(S_fit, owner[ft])
    del S_fit
    evals = {s: (score(sets[s][0]), sets[s][1]) for s in ("val", "test")}
    image_fit = None
    if w == "t2i":
        iq, it = load_split(gate_dir / w / "image-extra.npz")
        if len(iq) <= IMAGE_VAL:
            raise ValueError(f"image-extra.npz needs more than {IMAGE_VAL} rows; got {len(iq)}")
        evals["image_val"] = (score(iq[:IMAGE_VAL]), it[:IMAGE_VAL])
        image_fit = candidates(score(iq[IMAGE_VAL:]), owner[it[IMAGE_VAL:]])
        q, t = load_split(run_dir / w / "image.npz")
        evals["image"] = (score(q), t)
    return FitData(owner=owner, sizes=sizes, spread=spread, n_orig=len(sets["fit"][0]),
                   fit=fit_cands, evals=evals, image_fit=image_fit)


def arm_grid(grid, workload, data, steps):
    """(fit-set sizes for the learning curve, arm configs) for the chosen grid."""
    base = dict(kind="fit", steps=steps, alpha=0.0, lam=1e-4)
    if grid is Grid.QUICK:  # learning curve and gamma=0 / image-mixing arms settled on seed 2026
        arms = [dict(base, n=data.n_fit, tau_frac=t, gamma=g, lam=l)
                for t in QUICK_TAUS for g in (0.5, 1.0) for l in (1e-4, 1e-3)]
        return [data.n_fit], arms
    sizes_n = sorted({data.n_orig, 50_000, 100_000, data.n_fit})
    arms = [dict(base, n=n, tau_frac=t, gamma=g) for n in sizes_n for g in (0.0, 1.0) for t in TAUS]
    full = dict(base, n=data.n_fit)
    arms += [dict(full, tau_frac=t, gamma=g, lam=l, alpha=al)
             for t in GRID_TAUS for g in (0.0, 0.5, 1.0) for l in (1e-4, 1e-3, 1e-2)
             for al in ((0.0, 0.1, 0.3) if workload == "t2i" else (0.0,))
             if not (l == 1e-4 and al == 0.0 and g in (0.0, 1.0))]
    arms += [dict(kind="size_prior", n=0, tau_frac=t, gamma=g, lam=0.0, alpha=0.0, steps=0)
             for t in TAUS for g in (0.5, 1.0)]
    return sizes_n, arms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("gateb", type=Path)
    p.add_argument("workload", choices=["deep", "t2i"])
    p.add_argument("--seed", type=int, default=2026, choices=[2026, 2027])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--grid", type=Grid, choices=list(Grid), default=Grid.FULL,
                   help="full: every arm; quick: 16 arms on the full fit set")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    w = a.workload
    index_dir = a.run / w if a.seed == 2026 else a.gateb / w / f"seed{a.seed}"
    data = prepare(a.run, a.gateb, w, index_dir)
    print(f"{w} seed {a.seed}: {data.n_fit:,} fit queries ({data.n_orig:,} original)", flush=True)
    sizes_n, arms = arm_grid(a.grid, w, data, a.steps)
    baseline = evaluate(data, np.zeros(len(data.sizes)))
    print("centroids", {s: round(baseline[s]["work"]) for s in baseline}, flush=True)
    with Pool(a.workers, initializer=_init_worker, initargs=(data,)) as pool:
        rows = pool.map(_run_arm_in_worker, arms, chunksize=1)

    def pick(rs, key):
        if w == "t2i":  # image-shift guard, on image-validation queries only
            rs = [r for r in rs if r["image_val"][key] <= baseline["image_val"][key]]
        r = min(rs, key=lambda r: r["val"][key]) if rs else None
        if r is None or r["val"][key] >= baseline["val"][key]:
            return {"kind": "centroids", **baseline}  # validation keeps the centroids
        return r

    def summary(r):
        out = {"arm": {k: r.get(k) for k in ARM_KEYS}}
        for s in data.evals:
            out[s] = {"work": r[s]["work"], "probes": r[s]["probes"], "work_interp": r[s]["work_interp"],
                      "change": r[s]["work"] / baseline[s]["work"] - 1,
                      "change_interp": r[s]["work_interp"] / baseline[s]["work_interp"] - 1}
        return out

    picks = {key: summary(pick(rows, key)) for key in ("work", "work_interp")}
    curve_rows = []
    for g in (0.0, 1.0):
        for n in sizes_n:
            arms_n = [r for r in rows if r["kind"] == "fit" and r["n"] == n and r["gamma"] == g
                      and r["lam"] == 1e-4 and r["alpha"] == 0.0]
            curve_rows.append({"n": n, "gamma": g, **{key: summary(pick(arms_n, key))
                                                      for key in ("work", "work_interp")}})
    result = {
        "workload": w, "index_seed": a.seed, "score_spread": data.spread, "candidates": CAND,
        "fit_queries": data.n_fit, "original_fit_queries": data.n_orig,
        "image_fit_queries": None if data.image_fit is None else len(data.image_fit.cand),
        "centroids": baseline, "validation_picks": picks, "learning_curve": curve_rows,
        "unconstrained_validation_pick": summary(min(rows, key=lambda r: r["val"]["work"])),
        "rows": rows,
    }
    for key, sm in picks.items():
        print(f"{w} seed {a.seed} pick by val {key}:", sm["arm"],
              " ".join(f"{s} {100 * sm[s]['change']:+.2f}% (interp {100 * sm[s]['change_interp']:+.2f}%)"
                       for s in data.evals), flush=True)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
