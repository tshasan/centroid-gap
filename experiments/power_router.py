"""Power-diagram router on the frozen 1M IVF lists (Gate B, paper Section 7).

Keeps the stored centers and adds one learned scalar per list:
    score_c(q) = <q, c> + b_c            (t2i, inner product)
    score_c(q) = 2<q, c> - |c|^2 + b_c   (deep, i.e. -|q-c|^2 + b_c up to a constant)
For balanced lists this is the power-diagram form of a size-constrained
least-squares assignment (Aurenhammer, Hoffmann and Aronov 1998).

The biases minimise softmax cross-entropy of the lists holding each fit query's
true top-10, with logits (score + b)/tau. Each tau and the zero-bias baseline are
scored on validation work at 95% recall@10; the validation pick is reported on test.

    python experiments/power_router.py DATA RUN t2i --output results/power-router-t2i.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from .routing_stats import whole_probe_summary
else:
    from frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from routing_stats import whole_probe_summary

TARGET = 0.95
TAU_FRACS = (0.02, 0.05, 0.1, 0.2, 0.5)


def fit_bias(scores, labels, tau, steps, lr, l2):
    """Full-batch Adam on softmax cross-entropy of the label lists; returns biases with mean zero."""
    n, k = scores.shape
    target = np.zeros((n, k))
    np.add.at(target, (np.repeat(np.arange(n), labels.shape[1]), labels.ravel()), 1.0 / labels.shape[1])
    b = np.zeros(k)
    m = v = np.zeros(k)
    for t in range(1, steps + 1):
        z = (scores + b) / tau
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - target).mean(axis=0) / tau + l2 * b
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        b = b - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
    return b - b.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="unused; kept so the command line matches tip_router.py")
    parser.add_argument("run", type=Path)
    parser.add_argument("workload", choices=["deep", "t2i"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    w = args.workload
    metric = metric_for(w)
    index = load_frozen_index(args.run / w)
    owner, sizes, nlist = index.owner, index.sizes, index.nlist
    centers = index.centers.astype(np.float64)
    z = {s: load_split(args.run / w / f"{s}.npz") for s in ("fit", "val", "test")}
    S = {s: centroid_scores(q.astype(np.float64), centers, metric) for s, (q, _) in z.items()}
    truth = {s: t for s, (_, t) in z.items()}
    labels = owner[truth["fit"]]
    spread = score_spread(S["fit"])  # temperatures are fractions of this

    def summary(s, b):
        return whole_probe_summary(S[s] + b, truth[s], owner, sizes, nlist, TARGET)

    rows = [{"tau": None, "val": summary("val", 0), "test": summary("test", 0)}]
    print(f"{w} baseline        val {rows[0]['val']['work']:8.0f} test {rows[0]['test']['work']:8.0f}", flush=True)
    for frac in TAU_FRACS:
        tau = frac * spread
        b = fit_bias(S["fit"], labels, tau, args.steps, lr=0.05 * tau, l2=1e-4 / tau)
        row = {"tau": tau, "tau_over_spread": frac, "bias_std_over_spread": float(b.std() / spread),
               "val": summary("val", b), "test": summary("test", b)}
        rows.append(row)
        print(f"{w} tau={frac:4.2f}·spread val {row['val']['work']:8.0f} test {row['test']['work']:8.0f} "
              f"probes {row['test']['probes']}", flush=True)
    pick = min(rows, key=lambda r: r["val"]["work"])
    result = {"workload": w, "metric": metric, "score_spread": spread, "rows": rows,
              "validation_pick": pick,
              "test_change_vs_centroid": pick["test"]["work"] / rows[0]["test"]["work"] - 1}
    print(f"{w} pick tau={pick.get('tau_over_spread')} test change "
          f"{100 * result['test_change_vs_centroid']:+.2f}%", flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
