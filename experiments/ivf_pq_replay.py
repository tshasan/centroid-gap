"""Train residual IVF-PQ on fixed lists and replay original and learned routers.

--mode replay (default): train pq.faiss if missing, then replay every router and write pq.json.
--mode train: (re)train pq.faiss only.
--mode baseline: replay the original router only (no fitted routers needed); writes pq-baseline.json.
"""

import argparse
import json
from enum import Enum
from pathlib import Path

import faiss
import numpy as np

if __package__:
    from .ivf_pipeline import Stage, require_stage
    from .ivf_workload_index import SEED, TARGET, metric_for, router
else:
    from ivf_pipeline import Stage, require_stage
    from ivf_workload_index import SEED, TARGET, metric_for, router


SUBQUANTIZERS = {"deep": 16, "t2i": 25}
TRAIN_ROWS = 65536
NBITS = 8
ADD_BATCH = 32768
SEARCH_BATCH = 64
SHORTLIST = 100
PROBE_GRID = (1, 4, 8, 16, 32, 64, 128, 256, 512)
METRICS = ("recall@1", "recall@10")
METHODS = ("pq", "rerank100")


class Mode(str, Enum):
    REPLAY = "replay"
    TRAIN = "train"
    BASELINE = "baseline"


def load_layout(directory, workload):
    require_stage(directory, workload, Stage.INDEX)
    metric = metric_for(workload)
    centers = np.load(directory / "centers.npy")
    sizes = np.load(directory / "sizes.npy")
    flat = faiss.read_index(str(directory / "flat.faiss"), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    if type(flat) is not faiss.IndexIVFFlat or flat.metric_type != router(centers, metric).metric_type:
        raise ValueError("expected the original CPU IVF-Flat artifact in the workload metric")
    if not np.array_equal(flat.quantizer.reconstruct_n(0, flat.nlist), centers):
        raise ValueError("saved representatives differ from the IVF-Flat quantizer")
    return flat, centers, sizes, metric


def verify_layout(flat, pq, centers, sizes):
    if (not pq.is_trained or not pq.by_residual or pq.nlist != flat.nlist
            or pq.ntotal != flat.ntotal or pq.metric_type != flat.metric_type
            or pq.quantizer.ntotal != flat.nlist
            or not np.array_equal(pq.quantizer.reconstruct_n(0, pq.nlist), centers)):
        raise ValueError("IVF-PQ layout or residual quantizer differs from IVF-Flat")
    for bucket, size in enumerate(sizes):
        if (flat.invlists.list_size(bucket) != size or pq.invlists.list_size(bucket) != size):
            raise ValueError(f"list {bucket} changed size during PQ encoding")
        flat_ptr = flat.invlists.get_ids(bucket)
        pq_ptr = pq.invlists.get_ids(bucket)
        try:
            flat_ids = faiss.rev_swig_ptr(flat_ptr, int(size))
            pq_ids = faiss.rev_swig_ptr(pq_ptr, int(size))
            if not np.array_equal(flat_ids, pq_ids) and not np.array_equal(
                np.sort(flat_ids), np.sort(pq_ids)
            ):
                raise ValueError(f"stored IDs differ in list {bucket}")
        finally:
            flat.invlists.release_ids(bucket, flat_ptr)
            pq.invlists.release_ids(bucket, pq_ptr)


def train_pq(directory, workload, base):
    flat, centers, sizes, metric = load_layout(directory, workload)
    owner = np.load(directory / "owner.npy", mmap_mode="r")
    if (base.shape != (flat.ntotal, flat.d) or base.dtype != np.float32
            or owner.shape != (flat.ntotal,) or len(base) < TRAIN_ROWS):
        raise ValueError("base vectors or owner map differ from the stored IVF layout")

    quantizer = router(centers, metric)
    pq = faiss.IndexIVFPQ(quantizer, flat.d, flat.nlist, SUBQUANTIZERS[workload], NBITS, flat.metric_type)
    pq.pq.cp.seed = SEED
    train_ids = np.random.default_rng(SEED).choice(len(base), TRAIN_ROWS, replace=False)
    pq.train(np.ascontiguousarray(base[train_ids]))
    print(f"{workload}: trained residual PQ on {TRAIN_ROWS:,} vectors", flush=True)
    for start in range(0, len(base), ADD_BATCH):
        stop = min(start + ADD_BATCH, len(base))
        batch = np.ascontiguousarray(base[start:stop])
        ids = np.arange(start, stop, dtype=np.int64)
        lists = np.ascontiguousarray(owner[start:stop], dtype=np.int64)
        pq.add_core(len(ids), faiss.swig_ptr(batch), faiss.swig_ptr(ids), faiss.swig_ptr(lists))
    verify_layout(flat, pq, centers, sizes)
    transient = directory / "pq.faiss.tmp"
    faiss.write_index(pq, str(transient))
    transient.replace(directory / "pq.faiss")
    print(f"{workload}: encoded {pq.ntotal:,} vectors and verified every stored list ID", flush=True)


def sample_split(directory, split, amount, seed):
    with np.load(directory / f"{split}.npz") as artifact:
        rows = np.sort(np.random.default_rng(seed).choice(len(artifact["truth"]), amount, replace=False))
        return (np.ascontiguousarray(artifact["queries"][rows], dtype=np.float32),
                artifact["truth"][rows], artifact["source_rows"][rows])


def exact_rerank(base, queries, candidates, metric):
    valid = candidates >= 0
    if valid.all():
        vectors = base[candidates]
    else:
        vectors = np.zeros((*candidates.shape, base.shape[1]), dtype=base.dtype)
        vectors[valid] = base[candidates[valid]]
    if metric == "l2":
        vectors -= queries[:, None, :]
        scores = np.einsum("qnd,qnd->qn", vectors, vectors)
    else:
        scores = -np.einsum("qnd,qd->qn", vectors, queries)
    scores[~valid] = np.inf
    ranked = np.lexsort((candidates, scores), axis=1)
    return np.take_along_axis(candidates, ranked[:, :10], axis=1), int(valid.sum())


def recalls(found, truth):
    overlap = np.any(found[:, :10, None] == truth[:, None, :], axis=2)
    return {"recall@1": float(np.mean(found[:, 0] == truth[:, 0])),
            "recall@10": float(np.mean(overlap))}


def routing_orders(index, routers, queries):
    original_scores, original_order = index.quantizer.search(queries, index.nlist)
    by_list = np.empty_like(original_scores)
    np.put_along_axis(by_list, original_order, original_scores, axis=1)
    orders = {"original": original_order}
    for name, fitted in routers.items():
        orders[name] = original_order if fitted is None else fitted.search(queries, index.nlist)[1]
    return by_list, orders


def verify_native_replay(pq, queries):
    queries = queries[:32]
    # Compare the measured all-score gather and top-100 prefix to the native search.
    pq.nprobe = min(64, pq.nlist)
    scores, orders = routing_orders(pq, {}, queries)
    selected = np.ascontiguousarray(orders["original"][:, :pq.nprobe])
    coarse = np.take_along_axis(scores, selected, axis=1)
    direct_scores, direct_ids = pq.search(queries, 10)
    replay_scores, replay_ids = pq.search_preassigned(queries, SHORTLIST, selected, coarse)
    if (not np.array_equal(direct_ids, replay_ids[:, :10])
            or not np.allclose(direct_scores, replay_scores[:, :10], rtol=1e-5, atol=1e-6)):
        raise ValueError("original PQ replay disagrees with native top-10 search")


def measure(pq, base, sizes, metric, queries, truth, coarse, ordered, p, rescoring=False):
    pq.nprobe = p
    stats = faiss.cvar.indexIVF_stats
    native = np.empty((len(queries), 10), dtype=np.int64)
    reranked = np.empty_like(native)
    adc = reads = 0
    for start in range(0, len(queries), SEARCH_BATCH):
        rows = slice(start, start + SEARCH_BATCH)
        selected = np.ascontiguousarray(ordered[rows, :p])
        scores = np.take_along_axis(coarse[rows], selected, axis=1)
        stats.reset()
        _, shortlist = pq.search_preassigned(queries[rows], SHORTLIST, selected, scores)
        scanned = int(stats.ndis)
        if scanned != sizes[selected].sum():
            raise ValueError(f"native PQ candidate count disagrees with stored prefix at p={p}")
        native[rows] = shortlist[:, :10]
        reranked[rows], rerank_reads = exact_rerank(base, queries[rows], shortlist, metric)
        adc += scanned
        reads += rerank_reads

    count = len(queries)
    extra_coarse = p if rescoring else 0
    routing = pq.nlist + extra_coarse
    return {
        "probes": p, "queries": count,
        "pq": recalls(native, truth), "rerank100": recalls(reranked, truth),
        "work": {
            "routing_distance_evaluations_per_query": routing,
            "flat_routing_distance_evaluations_per_query": pq.nlist,
            "extra_original_centroid_scores_per_query": extra_coarse,
            "adc_candidates_total": adc,
            "mean_adc_candidates": adc / count,
            "exact_rerank_vectors_total": reads,
            "mean_exact_rerank_vectors": reads / count,
            "mean_pq_routing_plus_adc_count": routing + adc / count,
            "mean_rerank_routing_plus_adc_plus_exact_count": routing + (adc + reads) / count,
        },
    }


def choose(validation):
    points = {}
    for method in METHODS:
        points[method] = {}
        for metric in METRICS:
            feasible = [p for p, row in validation.items() if row[method][metric] >= TARGET]
            best = min(validation, key=lambda p: (-validation[p][method][metric], p))
            points[method][metric] = {
                "selected_probe": min(feasible) if feasible else None,
                "validation_best_probe": best,
                "validation_best_recall": validation[best][method][metric],
            }
    return points


def crossing_midpoints(curves):
    midpoints = set()
    for curve in curves.values():
        for metrics in choose(curve).values():
            for point in metrics.values():
                p = point["selected_probe"]
                if p is None or p == 1:
                    continue
                below = max(value for value in curve if value < p)
                if p - below > 1:
                    midpoints.add((below + p) // 2)
    return midpoints


def replay(directory, workload, base, args):
    baseline_only = args.mode is Mode.BASELINE
    if not baseline_only:
        require_stage(directory, workload, Stage.FITTED)
    flat, centers, sizes, metric = load_layout(directory, workload)
    pq = faiss.read_index(str(directory / "pq.faiss"))
    if type(pq) is not faiss.IndexIVFPQ or pq.pq.M != SUBQUANTIZERS[workload] or pq.pq.nbits != NBITS:
        raise ValueError("unexpected IVF-PQ encoding configuration")
    verify_layout(flat, pq, centers, sizes)
    del flat
    pq.max_codes = 0
    pq.parallel_mode = 0
    reps = {} if baseline_only else {
        "learned_at1": np.load(directory / "router-at1.npy"),
        "learned_at10": np.load(directory / "router-at10.npy"),
    }
    if metric == "l2" and not baseline_only:
        reps["half_scale_control"] = 0.5 * centers
    if args.challengers and not baseline_only:
        for k in (1, 10):
            if np.array_equal(reps[f"learned_at{k}"], centers):
                reps[f"challenger_at{k}"] = np.load(directory / f"challenger-at{k}.npy")
    if any(values.shape != centers.shape or not np.isfinite(values).all() for values in reps.values()):
        raise ValueError("replacement representatives must match the original list count and dimension")
    changed = {name for name, values in reps.items() if not np.array_equal(values, centers)}
    routers = {name: router(values, metric) if name in changed else None for name, values in reps.items()}
    val_x, val_y, val_rows = sample_split(directory, "val", args.val_queries, SEED + 100)
    verify_native_replay(pq, val_x)
    val_coarse, val_orders = routing_orders(pq, routers, val_x)
    if any(p > pq.nlist for p in args.extra_probes):
        raise ValueError("extra probes must not exceed the number of lists")
    grid = {p for p in PROBE_GRID if p < pq.nlist} | set(args.extra_probes) | {pq.nlist}
    validation = {name: {} for name in val_orders}

    def run_validation(p):
        measured = {}
        for name, order in val_orders.items():
            # The all-list diagnostic uses the same lists and scan order for every arm.
            if p == pq.nlist:
                order = val_orders["original"]
            rescoring = name in changed and p < pq.nlist
            key = (rescoring, order.tobytes())
            if key not in measured:
                measured[key] = measure(pq, base, sizes, metric, val_x, val_y, val_coarse, order, p, rescoring)
            validation[name][p] = measured[key]
        row = validation["original"][p]
        print(f"{workload}: validation p={p}, original PQ {row['pq']}, rerank {row['rerank100']}", flush=True)

    for p in sorted(grid):
        run_validation(p)
    for p in sorted(crossing_midpoints(validation) - grid):
        run_validation(p)
        grid.add(p)

    operating = {name: choose(curve) for name, curve in validation.items()}
    test_x, test_y, test_rows = sample_split(directory, "test", args.test_queries, SEED + 101)
    test_coarse, test_orders = routing_orders(pq, routers, test_x)
    exhaustive = measure(pq, base, sizes, metric, test_x, test_y, test_coarse,
                         test_orders["original"], pq.nlist)
    measured = {}
    results = {}
    for name, points in operating.items():
        test = {pq.nlist: exhaustive}
        for metrics in points.values():
            for point in metrics.values():
                p = point["selected_probe"] or point["validation_best_probe"]
                if p not in test:
                    order = test_orders[name]
                    rescoring = name in changed
                    key = (p, rescoring, order.tobytes())
                    if key not in measured:
                        measured[key] = measure(pq, base, sizes, metric, test_x, test_y, test_coarse, order, p, rescoring)
                    test[p] = measured[key]
                point["test_probe"] = p
        results[name] = {"changed_router": name in changed, "validation": validation[name],
                         "operating_points": points, "test": test}

    return {
        "workload": workload, "metric": metric, "faiss": faiss.__version__, "numpy": np.__version__,
        "nlist": pq.nlist, "vectors": pq.ntotal, "residual": True,
        "pq_subquantizers": pq.pq.M, "bits_per_subquantizer": pq.pq.nbits,
        "bytes_per_code": pq.code_size, "training_rows": TRAIN_ROWS, "training_seed": SEED,
        "layout_ids_verified": True, "native_top10_replay_verified_queries": min(32, len(val_x)),
        "validation_selection": "lowest tested probe at target, or highest validation recall for diagnosis",
        "target_recall": TARGET, "probe_grid": sorted(grid),
        "validation_source_rows": val_rows.tolist(), "test_source_rows": test_rows.tolist(),
        "common_all_list_order": "original quantizer",
        "routers": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("workload", choices=["deep", "t2i"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), default=Mode.REPLAY,
                        help="replay (default), train (PQ codes only) or baseline (original router only)")
    parser.add_argument("--challengers", action="store_true",
                        help="include the best validation non-original router when selection kept the original")
    parser.add_argument("--extra-probes", type=int, nargs="*", default=[])
    parser.add_argument("--val-queries", type=int, default=384)
    parser.add_argument("--test-queries", type=int, default=640)
    args = parser.parse_args()
    if min(args.threads, args.val_queries, args.test_queries, *args.extra_probes) < 1:
        parser.error("threads, query counts, and probes must be positive")
    if args.mode is Mode.TRAIN and (args.challengers or args.extra_probes or any(
            getattr(args, name) != parser.get_default(name) for name in ("val_queries", "test_queries"))):
        parser.error("--mode train skips the replay, so it takes no replay options")
    if args.mode is Mode.BASELINE and args.challengers:
        parser.error("--challengers needs the fitted routers, which --mode baseline skips")
    faiss.omp_set_num_threads(args.threads)
    base = np.load(args.data / f"{args.workload}-base.npy", mmap_mode="r")
    directory = args.run / args.workload
    if args.mode is Mode.TRAIN or not (directory / "pq.faiss").exists():
        train_pq(directory, args.workload, base)
    if args.mode is not Mode.TRAIN:
        report = replay(directory, args.workload, base, args)
        name = "pq-baseline.json" if args.mode is Mode.BASELINE else "pq.json"
        (directory / name).write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.workload}: wrote {directory / name}", flush=True)


if __name__ == "__main__":
    main()
