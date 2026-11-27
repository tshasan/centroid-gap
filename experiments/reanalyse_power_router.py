"""Re-read Gate B (power-diagram router) without one-probe steps, with intervals and a free baseline.

Refits the Gate B biases exactly as experiments/power_router.py does (same fit split, temperatures
and selection on validation work), then reports on the test split and, for Text-to-Image, the
image-shift queries:

  * work interpolated to exactly 95% recall@10 and at the first whole probe count, each with a
    paired query bootstrap 95% interval (experiments/routing_stats.py);
  * work with the probe count frozen at the value chosen on validation;
  * on DEEP (L2), the zero-parameter half-scale rule, i.e. a power router with bias |c|^2 / 2.

    python experiments/reanalyse_power_router.py ~/.cache/centroid-gap/run \
        --output results/power-router-reanalysis.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from .power_router import fit_bias
    from .routing_stats import compare, mean_recall_work, per_query
else:
    from frozen_index import centroid_scores, load_frozen_index, load_split, metric_for, score_spread
    from power_router import fit_bias
    from routing_stats import compare, mean_recall_work, per_query

TARGET = 0.95
FRACS = (0.02, 0.05, 0.1, 0.2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = {"target": TARGET, "boots": 1000}
    for w in ("t2i", "deep"):
        metric = metric_for(w)
        index = load_frozen_index(a.run / w)
        owner, sizes, nlist = index.owner, index.sizes, index.nlist
        C = index.centers.astype(np.float64)
        splits = ["fit", "val", "test"] + (["image"] if w == "t2i" else [])
        z = {s: load_split(a.run / w / f"{s}.npz") for s in splits}
        S = {s: centroid_scores(q.astype(np.float64), C, metric) for s, (q, _) in z.items()}
        truth = {s: t for s, (_, t) in z.items()}
        k = truth["test"].shape[1]
        labels = owner[truth["fit"]]
        spread = score_spread(S["fit"])

        def pq(s, b):
            return per_query(S[s] + b, truth[s], owner, sizes)

        zero = np.zeros(nlist)
        arms = {}
        for frac in FRACS:
            tau = frac * spread
            b = fit_bias(S["fit"], labels, tau, 400, lr=0.05 * tau, l2=1e-4 / tau)
            r, c = mean_recall_work(*pq("val", b), k)
            arms[frac] = (b, float(c[int(np.searchsorted(r, TARGET))]))
        pick = min(arms, key=lambda f: arms[f][1])
        bias = arms[pick][0]
        res = {"picked_tau_frac": pick, "val_whole_probe_work": {str(f): arms[f][1] for f in arms}}
        for s in [x for x in splits if x not in ("fit", "val")]:
            base, tuned = pq(s, zero), pq(s, bias)
            res[s] = compare(base, tuned, k, TARGET, nlist)
            if s == "test":
                # probe counts frozen on validation
                (rv0, _), (rv1, _) = mean_recall_work(*pq("val", zero), k), mean_recall_work(*pq("val", bias), k)
                p0, p1 = int(np.searchsorted(rv0, TARGET)) + 1, int(np.searchsorted(rv1, TARGET)) + 1
                (r0, c0), (r1, c1) = mean_recall_work(*base, k), mean_recall_work(*tuned, k)
                res["test_val_locked"] = {"base_probes": p0, "base_recall": float(r0[p0 - 1]), "base_work": float(c0[p0 - 1]),
                                          "other_probes": p1, "other_recall": float(r1[p1 - 1]), "other_work": float(c1[p1 - 1]),
                                          "change": float(c1[p1 - 1] / c0[p0 - 1] - 1)}
                if metric == "l2":
                    half = 0.5 * (C ** 2).sum(axis=1)  # 2<q,c> - |c|^2 + |c|^2/2 ranks like the half-scale centroids
                    res["test_half_scale"] = compare(base, pq(s, half), k, TARGET, nlist)
                    res["test_learned_vs_half_scale"] = compare(pq(s, half), tuned, k, TARGET, nlist)
        out[w] = res
        for s, r in res.items():
            if isinstance(r, dict) and "interpolated" in r:
                i, d = r["interpolated"], r["whole_probe"]
                print(f"{w} {s}: interpolated {100 * i['change']:+.2f}% [{100 * i['ci95'][0]:+.2f}, {100 * i['ci95'][1]:+.2f}] "
                      f"({i['base']:,.0f} -> {i['other']:,.0f}); whole-probe {100 * d['change']:+.2f}% "
                      f"[{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] ({d['base_probes']} -> {d['other_probes']} probes)",
                      flush=True)
        if "test_val_locked" in res:
            v = res["test_val_locked"]
            print(f"{w} test, probes frozen on validation: {v['base_probes']} -> {v['other_probes']}, "
                  f"recall {v['base_recall']:.4f} / {v['other_recall']:.4f}, work {100 * v['change']:+.2f}%", flush=True)
    if a.output:
        a.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
