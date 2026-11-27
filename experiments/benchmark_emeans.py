"""Compare paired Lloyd and e-means IVF-Flat builds on the prepared DEEP/T2I workloads."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import faiss
import faiss.contrib
import numpy as np
import scipy

if __package__:
    from .emeans_ivf_eval import evaluate_centers
    from .ivf_workload_index import N_LISTS, SEED, metric_for, router
else:
    from emeans_ivf_eval import evaluate_centers
    from ivf_workload_index import N_LISTS, SEED, metric_for, router


ARMS = (("lloyd", 5), ("e-means", 5), ("lloyd", 25))
SEEDS = (SEED, SEED + 1, SEED + 2)


def fit_centers(training, initial, metric, name, epochs, seed, emeans):
    k, d = initial.shape
    spherical = metric == "ip"
    start = perf_counter()
    if name == "lloyd":
        model = faiss.Clustering(d, k)
        model.niter, model.nredo, model.seed = epochs, 1, seed
        model.spherical = spherical
        model.max_points_per_centroid = (len(training) + k - 1) // k
        faiss.copy_array_to_vector(initial.ravel(), model.centroids)
        assigner = faiss.IndexFlatIP(d) if spherical else faiss.IndexFlatL2(d)
        model.train(training, assigner)
        centers = faiss.vector_to_array(model.centroids).reshape(k, d)
    else:
        model = emeans(d, k, nepochs=epochs, seed=seed, spherical=spherical,
                       capture_assignments=False)
        centers, _ = model.train(training, init_centroids=initial)
    seconds = perf_counter() - start
    stats = {"trainer": name, "epochs": epochs, "seed": seed, "fit_seconds": seconds}
    if name == "e-means":
        stats["recipe"] = {key: model.recipe[key] for key in
                           ("batch_law", "batch_size", "alpha0", "unused_threshold")}
    # Score the final frozen centers on every training row, outside the fit timer.
    distances, _ = router(centers, metric).search(training, 1)
    mean_score = float(distances.mean(dtype=np.float64))
    stats["mean_squared_training_distance"] = 2 - 2 * mean_score if spherical else mean_score
    return centers, stats


def benchmark(data, snapshots, workload, emeans):
    base = np.load(data / f"{workload}-base.npy", mmap_mode="r")
    metric = metric_for(workload)
    training = np.array(base, dtype=np.float32, order="C", copy=True)
    if metric == "ip":
        faiss.normalize_L2(training)
    rows = []
    for offset, seed in enumerate(SEEDS):
        ids = np.random.default_rng(seed).choice(len(training), N_LISTS, replace=False)
        initial = training[ids]
        fingerprint = hashlib.sha256(initial.tobytes()).hexdigest()
        # Each trainer occupies each position once over the three paired seeds.
        for name, epochs in ARMS[offset:] + ARMS[:offset]:
            print(f"{workload} seed={seed} {name} epochs={epochs}", flush=True)
            centers, stats = fit_centers(training, initial, metric, name, epochs, seed, emeans)
            stats["initial_centers_sha256"] = fingerprint
            stats["final_centers_sha256"] = hashlib.sha256(centers.tobytes()).hexdigest()
            stats["index"] = evaluate_centers(base, centers, metric, snapshots / workload,
                                              retune=(name == "e-means" or epochs == 25))
            rows.append(stats)
            p = stats["index"]["centroid"]["test"]["recall@10"]["at_95"]
            print(f"  fit {stats['fit_seconds']:.2f}s, {p['probes']} probes, "
                  f"{p['total']['mean']:.1f} scores/query", flush=True)
    return {"metric": metric, "base_vectors": len(base), "dimension": base.shape[1],
            "lists": N_LISTS, "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory prepared by ivf_workload_data.py")
    parser.add_argument("snapshots", type=Path, help="root with the original val/test exact labels")
    parser.add_argument("output", type=Path, help="directory for one JSON report per workload")
    parser.add_argument("--faiss-checkout", type=Path, required=True,
                        help="Faiss source checkout with contrib/e_means (see the report for the pin)")
    parser.add_argument("--workload", choices=("both", "deep", "t2i"), default="both")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    module_dir = (args.faiss_checkout / "contrib/e_means").resolve()
    if not (module_dir / "e_means.py").is_file():
        parser.error("--faiss-checkout must contain contrib/e_means/e_means.py")
    faiss.contrib.__path__.insert(0, str(module_dir.parent))
    from faiss.contrib.e_means import NumpyEmeans

    faiss.omp_set_num_threads(args.threads)
    environment = {
        "python": platform.python_version(), "faiss": faiss.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "architecture": platform.machine(),
        "faiss_threads": args.threads, "veclib_maximum_threads": os.getenv("VECLIB_MAXIMUM_THREADS"),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "emeans_source_sha256": {name: hashlib.sha256((module_dir / name).read_bytes()).hexdigest()
                                 for name in ("base.py", "e_means.py", "e_means_spherical.py")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for workload in ("deep", "t2i") if args.workload == "both" else (args.workload,):
        report = benchmark(args.data, args.snapshots, workload, NumpyEmeans)
        path = args.output / f"emeans-{workload}.json"
        path.write_text(json.dumps({"environment": environment, **report}, indent=2,
                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
