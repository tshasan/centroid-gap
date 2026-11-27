"""Evaluate frozen clustering centroids with native Faiss IVF-Flat."""

from time import perf_counter, perf_counter_ns

import faiss
import numpy as np

if __package__:
    from .benchmark_ivf_workloads import (
        check_native_result, probes, public_summary, split, validation_choices,
    )
    from .fit_ivf_router import fit_candidates, install_flat_router
    from .ivf_oracle import evaluate
    from .ivf_workload_index import SEED, router, stored_lists
else:
    from benchmark_ivf_workloads import (
        check_native_result, probes, public_summary, split, validation_choices,
    )
    from fit_ivf_router import fit_candidates, install_flat_router
    from ivf_oracle import evaluate
    from ivf_workload_index import SEED, router, stored_lists


def verify_native(index, queries, truth, report, cutoffs=(1, 10)):
    verification = {}
    for k in cutoffs:
        label = f"recall@{k}"
        checked = set()
        for name in ("at_95", "locked"):
            point = report[label][name]
            if point is None or point["probes"] in checked:
                continue
            index.nprobe = point["probes"]
            faiss.cvar.indexIVF_stats.reset()
            _, found = index.search(queries, k)
            check_native_result(found, truth, point, k)
            checked.add(index.nprobe)
        verification[label] = {"query_count": len(queries), "probes_checked": sorted(checked)}
    return verification


def matched_latency(index, queries, probe):
    sample = np.random.default_rng(SEED).permutation(len(queries))[:384]
    rows = [queries[i:i + 1] for i in sample]
    index.nprobe = probe
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        for row in rows:
            index.search(row, 10)
        elapsed_us = []
        for _ in range(3):
            for row in rows:
                start = perf_counter_ns()
                index.search(row, 10)
                elapsed_us.append((perf_counter_ns() - start) / 1000)
    finally:
        faiss.omp_set_num_threads(threads)
    p50, p95 = np.percentile(elapsed_us, [50, 95]).tolist()
    return {"probes": probe, "query_count": len(rows), "repeats": 3, "threads": 1,
            "local_wall_time_us_p50": p50, "local_wall_time_us_p95": p95}


def evaluate_retuned_at10(index, centers, metric, directory, owner, sizes, queries, truth, result):
    label = "recall@10"
    fit, fit_truth = split(directory, "fit")
    order = np.random.default_rng(SEED + 3).permutation(len(fit))
    candidates = fit_candidates(centers, fit[order], owner[fit_truth[order, 0]], metric)
    val, val_truth = split(directory, "val")
    choices, challengers, validation = validation_choices(candidates, val, val_truth, owner, sizes, metric)
    selected, changed = choices[10], challengers[10]
    models = {}
    for name in dict.fromkeys((selected, changed)):
        if name == "original":
            test = result["centroid"]["test"][label]
            verified = result["native_verified_queries"]["test"][label]
        else:
            install_flat_router(index, candidates[name])
            routing = index.quantizer.search(queries, index.nlist)[1]
            locked = {10: validation[name][label]["at_95"]["probes"]}
            report = evaluate(routing, truth, owner, sizes, probes=locked, return_details=True)
            test = public_summary(report)[label]
            verified = verify_native(index, queries, truth, report, cutoffs=(10,))[label]
        models[name] = {**test, "native_verified_queries": verified}
    return {
        "fit_queries": len(fit), "selected": selected, "changed_challenger": changed,
        "models": models,
        "validation_mean_total": {
            name: report[label]["at_95"]["total"]["mean"] for name, report in validation.items()
        },
    }


def evaluate_centers(base, centers, metric, directory, *, retune=False):
    """Build one IVF table from raw base vectors and evaluate its native serving path."""
    coarse = router(centers, metric)
    index = faiss.IndexIVFFlat(coarse, coarse.d, coarse.ntotal, coarse.metric_type)
    start = perf_counter()
    index.add(base)
    add_seconds = perf_counter() - start
    owner, sizes = stored_lists(index)
    p50, p95, p99 = np.quantile(sizes, [0.5, 0.95, 0.99], method="inverted_cdf")
    result = {
        "add_seconds": add_seconds,
        "list_sizes": {"min": int(sizes.min()), "p50": int(p50), "p95": int(p95),
                       "p99": int(p99), "max": int(sizes.max()), "mean": float(sizes.mean())},
        "centroid": {}, "oracle": {}, "native_verified_queries": {},
    }
    centroid_probes = oracle_probes = None
    for split_name, path_name in (("validation", "val"), ("test", "test")):
        queries, truth = split(directory, path_name)
        order = coarse.search(queries, index.nlist)[1]
        native = evaluate(order, truth, owner, sizes, probes=centroid_probes, return_details=True)
        del order
        exact = evaluate(None, truth, owner, sizes, oracle=True, probes=oracle_probes)
        result["centroid"][split_name] = public_summary(native)
        result["oracle"][split_name] = public_summary(exact)
        result["native_verified_queries"][split_name] = verify_native(index, queries, truth, native)
        centroid_probes, oracle_probes = probes(native), probes(exact)

    result["matched_test_at10_latency"] = matched_latency(index, queries, centroid_probes[10])
    if retune:
        result["retuned_at10"] = evaluate_retuned_at10(
            index, centers, metric, directory, owner, sizes, queries, truth, result,
        )
    return result
