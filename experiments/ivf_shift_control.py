"""Replay text-calibrated T2I routers on held-out text and image queries."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np

from benchmark_ivf_workloads import native_check, original_scores, probes, public_summary, split
from ivf_oracle import evaluate
from ivf_pipeline import Stage, require_stage
from ivf_workload_index import SEED, metric_for, router


IMAGE_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
RECALL_K = (1, 10)


def arms_for(k):
    return {"centroid": "centroid", "fitted": f"fitted@{k}", "challenger": f"challenger@{k}"}


def read_json(directory, name):
    return json.loads((directory / f"{name}.json").read_text())


def operating_points(reps, metric, choices, query_split, owner, sizes):
    queries, truth = query_split
    orders = {
        name: router(representatives, metric).search(queries, len(sizes))[1]
        for name, representatives in reps.items()
    }
    results = {
        name: evaluate(order, truth, owner, sizes, probes=choices[name], return_details=True)
        for name, order in orders.items()
    }
    return orders, results


def mix_means(text, image, image_share):
    return {
        name: (1 - image_share) * left + image_share * right
        for name, left, right in (
            ("recall", text["recall"], image["recall"]),
            ("mean_scans", text["scans"]["mean"], image["scans"]["mean"]),
            ("mean_total_work", text["total"]["mean"], image["total"]["mean"]),
        )
    }


def check_image_replay(directory, centers, metric, image, sample, orders, evaluations):
    queries, truth = image
    queries, truth = queries[sample], truth[sample]
    scores, _ = original_scores(centers, queries, metric)
    index = faiss.read_index(str(directory / "flat.faiss"))
    checks = {}
    for k in RECALL_K:
        key = f"recall@{k}"
        checks[key] = {}
        for arm, name in arms_for(k).items():
            results = evaluations[name][key]
            for label in ("locked", "at_95"):
                point = results[label]
                details = {field: values[sample] for field, values in point["details"].items()}
                subset = {**point, "details": details}
                native_check(index, queries, truth, orders[name][sample], scores, subset, k)
                checks[key].setdefault(arm, {})[label] = {
                    "probes": point["probes"],
                    "queries": len(sample),
                    "native_vectors_scanned": int(faiss.cvar.indexIVF_stats.ndis),
                    "expected_vectors_scanned": int(details["scans"].sum()),
                    "verified_truth_hits": int(details["hits"].sum()),
                    "truth_neighbors": len(sample) * k,
                }
    return checks


def run(directory, replay_queries):
    require_stage(directory, "t2i", Stage.INDEX)
    require_stage(directory, "t2i", Stage.BASELINE)
    require_stage(directory, "t2i", Stage.FITTED)
    metric = metric_for("t2i")
    index_report = read_json(directory, "index")
    baseline = read_json(directory, "flat-baseline")["centroid"]
    fit_report = read_json(directory, "flat-fitted")
    train_count = max(map(int, fit_report))
    fitted = fit_report[str(train_count)]
    centers, owner, sizes = [
        np.load(directory / f"{key}.npy") for key in ("centers", "owner", "sizes")
    ]
    reps = {"centroid": centers}
    choices = {"centroid": probes(baseline["validation"])}
    for k in RECALL_K:
        row = fitted[f"recall@{k}"]
        reps[f"fitted@{k}"] = np.load(directory / f"router-at{k}.npy")
        choices[f"fitted@{k}"] = {k: row["val_probe"]}
        reps[f"challenger@{k}"] = np.load(directory / f"challenger-at{k}.npy")
        choices[f"challenger@{k}"] = {k: row["diagnostic_challenger"]["val_probe"]}

    val, test, image = (split(directory, name) for name in ("val", "test", "image"))
    if metric != index_report["metric"] or any(
        len(q) != index_report["splits"][split_name]["retained"]
        for split_name, (q, _) in (("val", val), ("test", test), ("image", image))
    ) or train_count != index_report["splits"]["fit"]["retained"]:
        raise ValueError("T2I index and calibration report counts must match the query files")
    for name, representatives in reps.items():
        if representatives.shape != centers.shape:
            raise ValueError(f"{name}: expected one representative per stored list")
        order = router(representatives, metric).search(val[0], len(sizes))[1]
        val_probes = probes(evaluate(order, val[1], owner, sizes))
        if any(val_probes[k] != locked for k, locked in choices[name].items()):
            raise ValueError(f"{name}: stored probes disagree with text validation")

    text_eval = operating_points(reps, metric, choices, test, owner, sizes)[1]
    image_orders, image_eval = operating_points(reps, metric, choices, image, owner, sizes)
    sample = np.sort(np.random.default_rng(SEED).choice(
        len(image[0]), size=min(replay_queries, len(image[0])), replace=False
    ))
    replays = check_image_replay(directory, centers, metric, image, sample, image_orders, image_eval)
    result = {
        "workload": "t2i",
        "metric": metric,
        "base_vectors": len(owner),
        "lists": len(sizes),
        "shift": "Change the image share in a mixture of held-out text and image queries",
        "work": "Exact candidate scans plus one comparison per list representative",
        "splits": index_report["splits"],
        "fitted_training_queries": train_count,
        "challenger": "Best non-original router on text validation, reported as a diagnostic arm",
        "locked_probe_source": "95% recall on text validation, one probe count per router and recall cutoff",
        "endpoint_recalibration": "Descriptive first 95% recall prefix on each pure held-out endpoint",
        "image_native_replay_sample_seed": SEED,
        "image_native_replay_sample_rows": sample.tolist(),
        "metrics": {},
    }
    for k in RECALL_K:
        key = f"recall@{k}"
        arms = arms_for(k)
        endpoints = {
            modality: {arm: public_summary(evaluation[name])[key] for arm, name in arms.items()}
            for modality, evaluation in (("text", text_eval), ("image", image_eval))
        }
        mixtures = []
        for image_share in IMAGE_SHARES:
            point = {
                arm: mix_means(endpoints["text"][arm]["locked"],
                               endpoints["image"][arm]["locked"], image_share)
                for arm in arms
            }
            mixtures.append({"image_share": image_share, **point})
        result["metrics"][key] = {
            "fitted_router": fitted[key]["name"],
            "challenger_router": fitted[key]["diagnostic_challenger"]["name"],
            "validation_probes": {arm: choices[name][k] for arm, name in arms.items()},
            "endpoints": endpoints,
            "text_calibrated_mixtures": mixtures,
            "image_native_replay": replays[key],
        }
    path = directory / "shift.json"
    with TemporaryDirectory(dir=directory) as temporary:
        partial = Path(temporary) / path.name
        partial.write_text(json.dumps(result, indent=2) + "\n")
        partial.replace(path)
    print(path, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="root containing the prepared t2i index and routers")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replay-queries", type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1 or args.replay_queries < 1:
        parser.error("threads and replay query count must be positive")
    faiss.omp_set_num_threads(args.threads)
    run(args.output / "t2i", args.replay_queries)


if __name__ == "__main__":
    main()
