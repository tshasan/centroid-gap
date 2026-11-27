"""Measure truth-aware headroom and fit one query-trained representative per stored list."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

if __package__:
    from .fit_ivf_router import fit_candidates, install_flat_router
    from .ivf_oracle import evaluate, oracle_order
    from .ivf_pipeline import Stage, UPSTREAM, check_stage_order, require_stage
    from .ivf_workload_index import SEED, metric_for, router
else:
    from fit_ivf_router import fit_candidates, install_flat_router
    from ivf_oracle import evaluate, oracle_order
    from ivf_pipeline import Stage, UPSTREAM, check_stage_order, require_stage
    from ivf_workload_index import SEED, metric_for, router


def split(directory, name):
    with np.load(directory / f"{name}.npz") as data:
        return data["queries"], data["truth"]


def probes(report):
    return {k: report[f"recall@{k}"]["at_95"]["probes"] for k in (1, 10)}


def public_summary(report):
    return {metric: {name: ({k: v for k, v in point.items() if k != "details"}
                            if isinstance(point, dict) else point)
                     for name, point in measurement.items()}
            for metric, measurement in report.items()}


def original_scores(reps, queries, metric):
    distances, ids = router(reps, metric).search(queries, len(reps))
    scores = np.empty_like(distances)
    np.put_along_axis(scores, ids, distances, axis=1)
    return scores, ids


def check_native_result(found, truth, point, k):
    actual = (found[:, :, None] == truth[:, None, :k]).any(axis=2).sum(axis=1)
    if not np.array_equal(actual, point["details"]["hits"]):
        raise AssertionError("exact Faiss retrieval and true-list ranks disagree")
    if faiss.cvar.indexIVF_stats.ndis != point["details"]["scans"].sum():
        raise AssertionError("native Faiss scans and list sizes disagree")


def native_check(index, queries, truth, order, scores, point, k):
    p = point["probes"]
    ids = np.ascontiguousarray(order[:, :p])
    original_distances = np.take_along_axis(scores, ids, axis=1)
    index.nprobe = p
    faiss.cvar.indexIVF_stats.reset()
    _, found = index.search_preassigned(queries, k, ids, original_distances)
    check_native_result(found, truth, point, k)


def baseline_state(directory, workload):
    require_stage(directory, workload, Stage.INDEX)
    metric = metric_for(workload)
    reps, owner, sizes = [np.load(directory / f"{key}.npy") for key in ("centers", "owner", "sizes")]
    index = faiss.read_index(str(directory / "flat.faiss"))
    vq, vtruth = split(directory, "val")
    tq, truth = split(directory, "test")
    val_order = router(reps, metric).search(vq, len(sizes))[1]
    test_scores, test_order = original_scores(reps, tq, metric)
    summaries = {}
    records = {"truth": truth.astype(np.int32)}
    routes = [("centroid", val_order, test_order), ("oracle", None, None)]
    if metric == "l2":
        # Carry over the fixed half-scale rule from the smaller DEEP pilot.
        half = router(0.5 * reps, metric)
        routes.append(("half_scale_control", half.search(vq, len(sizes))[1],
                       half.search(tq, len(sizes))[1]))
    for name, val_routing, test_routing in routes:
        oracle = name == "oracle"
        val = evaluate(val_routing, vtruth, owner, sizes, oracle=oracle)
        test = evaluate(test_routing, truth, owner, sizes, oracle=oracle, probes=probes(val),
                        return_details=True)
        for k in (1, 10):
            label = f"recall@{k}"
            order = oracle_order(truth[:, :k], owner, sizes) if oracle else test_routing
            seen = set()
            for which in ("at_95", "locked"):
                point = test[label][which]
                if point["probes"] not in seen:
                    native_check(index, tq, truth, order, test_scores, point, k)
                    seen.add(point["probes"])
            if name == "half_scale_control":
                install_flat_router(index, 0.5 * reps)
                point = test[label]["at_95"]
                index.nprobe = point["probes"]
                faiss.cvar.indexIVF_stats.reset()
                _, found = index.search(tq, k)
                check_native_result(found, truth, point, k)
            records[f"{name}_at{k}_scans"] = test[label]["at_95"]["details"]["scans"].astype(np.int32)
            if name == "centroid" and k == 10:
                records["centroid_ranks"] = test[label]["at_95"]["details"]["ranks"].astype(np.int16)
        summaries[name] = {"validation": public_summary(val), "test": public_summary(test)}
    np.savez_compressed(directory / "ranks.npz", **records)
    (directory / "flat-baseline.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def savings_report(first, second, nlist):
    rng = np.random.default_rng(SEED + 7)
    diffs = []
    a = first["details"]["scans"] + nlist
    b = second["details"]["scans"] + nlist
    for _ in range(1000):
        idx = rng.integers(len(a), size=len(a))
        diffs.append(1 - b[idx].mean() / a[idx].mean())
    return {"saving": float(1 - b.mean() / a.mean()),
            "locked_query_bootstrap_95": np.percentile(diffs, [2.5, 97.5]).tolist()}


def validation_choices(candidates, queries, truth, owner, sizes, metric):
    reports = {}
    for name, reps in candidates.items():
        order = router(reps, metric).search(queries, len(sizes))[1]
        reports[name] = evaluate(order, truth, owner, sizes)
    choices, challengers = {}, {}
    for k in (1, 10):
        key = f"recall@{k}"
        ordered = sorted(reports, key=lambda name: (
            reports[name][key]["at_95"]["total"]["mean"], -reports[name][key]["at_95"]["recall"]))
        choices[k] = ordered[0]
        challengers[k] = next(name for name in ordered if name != "original")
    return choices, challengers, reports


def fit_state(directory, workload):
    require_stage(directory, workload, Stage.INDEX)
    metric = metric_for(workload)
    initial, owner, sizes = [np.load(directory / f"{key}.npy") for key in ("centers", "owner", "sizes")]
    nlist = len(sizes)
    train, train_ids = split(directory, "fit")
    val, val_ids = split(directory, "val")
    test, test_ids = split(directory, "test")
    index = faiss.read_index(str(directory / "flat.faiss"))
    order = np.random.default_rng(SEED + 3).permutation(len(train))
    train, home = train[order], owner[train_ids[order, 0]]
    val_order = router(initial, metric).search(val, nlist)[1]
    base_val = evaluate(val_order, val_ids, owner, sizes)
    scores, base_order = original_scores(initial, test, metric)
    base_test = evaluate(base_order, test_ids, owner, sizes, probes=probes(base_val), return_details=True)
    results = {}
    for count in (256, 1024, 4096, len(train)):
        candidates = fit_candidates(initial, train[:count], home[:count], metric)
        choices, challengers, validation = validation_choices(candidates, val, val_ids, owner, sizes, metric)
        models = {}
        names = set(choices.values())
        if count == len(train):
            names.update(challengers.values())
        for name in names:
            routing = router(candidates[name], metric).search(test, nlist)[1]
            evaluated = evaluate(routing, test_ids, owner, sizes, probes=probes(validation[name]),
                                 return_details=True)
            models[name] = (routing, evaluated)
        row = {}
        for k in (1, 10):
            key = f"recall@{k}"
            pair = [("router", choices[k])]
            if count == len(train):
                pair.append(("challenger", challengers[k]))
            for role, name in pair:
                routing, evaluated = models[name]
                test_row, base_row = evaluated[key], base_test[key]
                result = {"name": name, "val_probe": probes(validation[name])[k],
                          "test": public_summary(evaluated)[key],
                          "saving_at_frozen_probes": savings_report(base_row["locked"], test_row["locked"], nlist),
                          "matched_curve_saving": 1 - test_row["at_95"]["total"]["mean"] / base_row["at_95"]["total"]["mean"]}
                if role == "router":
                    row[key] = result
                else:
                    row[key]["diagnostic_challenger"] = result
                    row[key]["validation_candidates"] = {
                        candidate: report[key]["at_95"]["total"]["mean"] for candidate, report in validation.items()}
                if count == len(train):
                    np.save(directory / f"{role}-at{k}.npy", candidates[name])
                    for which in ("at_95", "locked"):
                        if which == "locked" and test_row["at_95"]["probes"] == test_row["locked"]["probes"]:
                            continue
                        native_check(index, test, test_ids, routing, scores, test_row[which], k)
                    install_flat_router(index, candidates[name])
                    point = test_row["at_95"]
                    index.nprobe = point["probes"]
                    faiss.cvar.indexIVF_stats.reset()
                    _, found = index.search(test, k)
                    check_native_result(found, test_ids, point, k)
                    result["single_table_native_verified_queries"] = len(test)
        results[str(count)] = row
        print(f"{workload} {count} queries: {choices}", flush=True)
    (directory / "flat-fitted.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def export_results(directory, workload, data, destination):
    require_stage(data, workload, Stage.DATA)
    require_stage(directory, workload, Stage.INDEX)
    require_stage(directory, workload, Stage.BASELINE)
    require_stage(directory, workload, Stage.FITTED)
    require_stage(directory, workload, Stage.PQ_REPLAY)
    reports = ["index", "flat-baseline", "flat-fitted", "pq"]
    if workload == "t2i":
        require_stage(directory, workload, Stage.SHIFT_CONTROL)
        reports.append("shift")
    check_stage_order(directory, ["ranks.npz", *(f"{name}.json" for name in reports)])
    destination.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((data / f"{workload}-workload.json").read_text())
    with np.load(directory / "test.npz") as sample, np.load(directory / "ranks.npz") as ranks:
        public_ids = np.asarray(manifest["selected_source_ids"]["test"])[sample["source_rows"]]
        owner = np.load(directory / "owner.npy")
        np.savez_compressed(destination / f"{workload}-ranks.npz", public_query_row=public_ids,
                            true_lists=owner[ranks["truth"]], **dict(ranks))
    result = {name: json.loads((directory / f"{name}.json").read_text()) for name in reports}
    (destination / f"{workload}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="root prepared by ivf_workload_index.py")
    parser.add_argument("workload", choices=["deep", "t2i"])
    parser.add_argument("stage", choices=["oracle", "fit", "export"])
    parser.add_argument("--data", type=Path, help="prepared workload data for source query IDs on export")
    parser.add_argument("--destination", type=Path, help="directory for shareable counts and ranks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)
    directory = args.output / args.workload
    if args.stage == "oracle":
        baseline_state(directory, args.workload)
    elif args.stage == "fit":
        fit_state(directory, args.workload)
    else:
        if args.data is None or args.destination is None:
            parser.error("export requires --data and --destination")
        export_results(directory, args.workload, args.data, args.destination)
