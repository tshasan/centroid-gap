"""Check tools/ivf_tune on the cached million-vector indexes, through Faiss's real search path.

Fits a tuning from the labelled fit queries (the original fit split plus the Gate B extra labels),
then evaluates on the experiments' held-out test split (and, on Text-to-Image, the 1,993
image-shift queries), which ivf_tune never sees. Reports, for native index.search at the centroid
nprobe ivf_tune picked and for ivf_tune.search with the tuned biases and nprobe:

  * recall@10 and distance evaluations per query at those whole probe counts;
  * work interpolated to exactly 95% recall@10, with a paired query bootstrap 95% interval
    (experiments/routing_stats.py), so a one-probe step cannot make or hide a gain;
  * batch throughput: each arm searches the whole split in one call; the two arms alternate for
    REPEATS rounds and the median and range of the per-query time are reported. This is batch
    time divided by query count, not single-query latency.

--seed 2027 evaluates on the second index (same recipe, k-means seed 2027) staged by
experiments/power_router_data.py; the IVF-Flat index is rebuilt from its saved centroids.

    python experiments/validate_ivf_tune.py ~/.cache/centroid-gap/run ~/.cache/centroid-gap/gateb \
        --output results/ivf-tune-validation.json
    python experiments/validate_ivf_tune.py ~/.cache/centroid-gap/run ~/.cache/centroid-gap/gateb \
        --seed 2027 --data ~/.cache/centroid-gap/data --output results/ivf-tune-validation-seed2027.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ivf_tune"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ivf_tune  # noqa: E402
from ivf_tune.core import Layout, list_owners, owner_of, per_query_recall  # noqa: E402
from routing_stats import compare, per_query  # noqa: E402

REPEATS = 7
TARGET = 0.95


def timed_pair(fa, fb, repeats=REPEATS):
    """Alternate the two calls so drift on a shared machine hits both arms alike."""
    fa(), fb()  # warm up
    ta, tb = [], []
    for _ in range(repeats):
        for fn, ts in ((fa, ta), (fb, tb)):
            t = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t)
            if fn is fa:
                oa = out
            else:
                ob = out
    return oa, np.array(ta), ob, np.array(tb)


def rebuild(data, gateb, w, seed):
    """IVF-Flat index with the saved centroids of a staged second index seed."""
    centers = np.load(gateb / w / f"seed{seed}" / "centers.npy").astype(np.float32)
    base = np.load(data / f"{w}-base.npy", mmap_mode="r")
    metric = faiss.METRIC_INNER_PRODUCT if w == "t2i" else faiss.METRIC_L2
    coarse = (faiss.IndexFlatIP if w == "t2i" else faiss.IndexFlatL2)(centers.shape[1])
    coarse.add(centers)
    index = faiss.IndexIVFFlat(coarse, centers.shape[1], len(centers), metric)
    index.is_trained = True
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    sizes = np.array([index.invlists.list_size(c) for c in range(index.nlist)])
    if not np.array_equal(sizes, np.load(gateb / w / f"seed{seed}" / "sizes.npy")):
        raise SystemExit("rebuilt list sizes differ from the staged seed: different Faiss assignment")
    return index


def evaluate(index, tuning, centroid_nprobe, q, truth, k=10):
    layout = Layout(index)
    sizes = layout.sizes
    index.nprobe = centroid_nprobe
    (_, I0), t0, (_, I1), t1 = timed_pair(lambda: index.search(q, k), lambda: ivf_tune.search(index, tuning, q, k))
    _, assign = index.quantizer.search(q, centroid_nprobe)
    w0 = layout.nlist + sizes[assign].sum(axis=1).mean()
    lists = layout.route(tuning.bias, np.ascontiguousarray(q, dtype=np.float32), tuning.nprobe)[0]
    w1 = layout.nlist + sizes[lists].sum(axis=1).mean()
    ms = lambda t: {"median": float(1e3 * np.median(t) / len(q)), "min": float(1e3 * t.min() / len(q)),
                    "max": float(1e3 * t.max() / len(q))}
    out = {
        "queries": len(q),
        "centroids": {"nprobe": centroid_nprobe, "recall": float(per_query_recall(I0, truth).mean()),
                      "work": float(w0), "batch_ms_per_query": ms(t0)},
        "tuned": {"nprobe": tuning.nprobe, "recall": float(per_query_recall(I1, truth).mean()),
                  "work": float(w1), "batch_ms_per_query": ms(t1)},
    }
    out["work_change"] = w1 / w0 - 1
    out["time_change_median"] = float(np.median(t1) / np.median(t0) - 1)
    # paired ratio per round: both arms ran back to back, so this cancels slow drift
    ratio = t1 / t0 - 1
    out["time_change_rounds"] = {"min": float(ratio.min()), "max": float(ratio.max())}
    ids, owners = list_owners(index)
    owner = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    owner[ids] = owners
    S = layout.scores(q)
    out["at_target"] = compare(per_query(S, truth, owner, sizes), per_query(S + tuning.bias, truth, owner, sizes),
                               k, TARGET, layout.nlist)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("gateb", type=Path)
    p.add_argument("--workload", choices=("deep", "t2i", "both"), default="both")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026, choices=(2026, 2027), help="index k-means seed")
    p.add_argument("--data", type=Path, help="cached base vectors (needed for --seed 2027)")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    results = {"threads": a.threads, "faiss": faiss.__version__, "index_seed": a.seed, "repeats": REPEATS,
               "timing": "batch throughput: whole split per call, arms alternated, median and range"}
    for w in ("deep", "t2i") if a.workload == "both" else (a.workload,):
        if a.seed == 2026:
            index = faiss.read_index(str(a.run / w / "flat.faiss"))
        else:
            if a.data is None:
                raise SystemExit("--seed 2027 needs --data")
            index = rebuild(a.data, a.gateb, w, a.seed)
        fit, extra = np.load(a.run / w / "fit.npz"), np.load(a.gateb / w / "extra.npz")
        q = np.concatenate([fit["queries"], extra["queries"]])
        t = np.concatenate([fit["truth"], extra["truth"]])
        shift = shift_truth = None
        if w == "t2i":
            img = np.load(a.gateb / w / "image-extra.npz")
            shift, shift_truth = img["queries"][:3000], img["truth"][:3000]
        print(f"== {w}", flush=True)
        start = time.perf_counter()
        tuning = ivf_tune.fit(index, q, truth=t, shift_queries=shift, shift_truth=shift_truth)
        fit_seconds = time.perf_counter() - start
        centroid_nprobe = tuning.report["centroids"]["nprobe"]
        test = np.load(a.run / w / "test.npz")
        res = {"fit_seconds": fit_seconds, "report": tuning.report,
               "test": evaluate(index, tuning, centroid_nprobe, test["queries"], test["truth"])}
        if w == "t2i":
            image = np.load(a.run / w / "image.npz")
            res["image_shift"] = evaluate(index, tuning, centroid_nprobe, image["queries"], image["truth"])
        for split in ("test", "image_shift"):
            if split in res:
                r = res[split]
                it = r["at_target"]["interpolated"]
                print(f"{w} {split}: centroids nprobe {r['centroids']['nprobe']} recall {r['centroids']['recall']:.4f} "
                      f"work {r['centroids']['work']:,.0f} {r['centroids']['batch_ms_per_query']['median']:.3f} ms | tuned nprobe "
                      f"{r['tuned']['nprobe']} recall {r['tuned']['recall']:.4f} work {r['tuned']['work']:,.0f} "
                      f"{r['tuned']['batch_ms_per_query']['median']:.3f} ms | work {100 * r['work_change']:+.2f}% "
                      f"time {100 * r['time_change_median']:+.2f}% (rounds {100 * r['time_change_rounds']['min']:+.1f} to "
                      f"{100 * r['time_change_rounds']['max']:+.1f}%) | at 95%: {100 * it['change']:+.2f}% "
                      f"[{100 * it['ci95'][0]:+.2f}, {100 * it['ci95'][1]:+.2f}]", flush=True)
        results[w] = res
    if a.output:
        a.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
