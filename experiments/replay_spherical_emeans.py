"""Replay the spherical counterexample through the pinned CPU NumpyEmeans trainer."""

import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import faiss
import faiss.contrib
import numpy as np
import scipy

if __package__:
    from .fit_ivf_router import install_flat_router, make_router
    from .ivf_workload_index import stored_lists
else:
    from fit_ivf_router import install_flat_router, make_router
    from ivf_workload_index import stored_lists


COMMIT = "2b648f0605f2ef3d14c4705350f8a0b214463878"
SOURCE_HASHES = {
    "__init__.py": "ad196786de8eb8ab7a229908b86ad8548a72459ade95e9e9c3275943dd50bb78",
    "base.py": "bcffd7829ebfc25e1354308770b49e6f99d8566ba5e474a41a22e9ec414b17cd",
    "e_means.py": "b1710504fdbef21efdd33d8489a56f8e7fc6056182afbed3f355985302c2e77c",
    "e_means_spherical.py": "d4e4177593f70e284bd2f6ad8812170e2a830ea62c697c514edc6298a00ea7a6",
}
SEED = 2026
START_OFFSET = 1 / 200


def load_trainer(checkout):
    module = (checkout / "contrib/e_means").resolve()
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((module / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{module / name} differs from the pinned source at {COMMIT}")
    faiss.contrib.__path__.insert(0, str(module.parent))
    from faiss.contrib.e_means import NumpyEmeans

    if Path(inspect.getfile(NumpyEmeans)).resolve() != module / "e_means.py":
        raise ValueError("NumpyEmeans did not load from the requested checkout")
    return NumpyEmeans


def example(k):
    """Construct Vectors.site at t = 1/(1000 k²), then round to the trainer's FP32."""
    n, d = k * k, 1 + k + k * k
    row, address = np.arange(k), np.arange(n)
    homes = address // k
    anchors = np.zeros((k, d))
    anchors[:, 0] = 2 / 5
    anchors[row, row + 1] = np.sqrt(21) / 5
    anchors[0, :2] = [0, 1]
    t = 1 / (1000 * n)
    docs = (1 - t * t) / (1 + t * t) * anchors[homes]
    docs[address, 1 + k + address] = 2 * t / (1 + t * t)
    docs[0] = 0
    docs[0, :2] = [1 / 2, np.sqrt(3) / 2]
    docs = docs.astype("float32")
    query = np.zeros((1, d), dtype="float32")
    query[0, 0] = 1
    # A private direction changes each normalized start as well as its raw norm.
    initial = anchors.copy()
    initial[row, 1 + k + row * k] = START_OFFSET
    initial = initial.astype("float32")
    displacement = np.linalg.norm(initial - anchors, axis=1)
    if not np.all((0 < displacement) & (displacement < 0.01)):
        raise AssertionError("raw FP32 starts must lie strictly inside the anchor neighborhood")
    unit_error = float(np.max(np.abs(np.linalg.norm(docs.astype("float64"), axis=1) - 1)))
    distinct = len(np.unique(docs, axis=0))
    if distinct != n or unit_error >= 2e-7:
        raise AssertionError("the FP32 documents lost their distinctness or unit accuracy")
    scores, ids = make_router(docs, "ip").search(query, 2)
    gap = float(scores[0, 0] - scores[0, 1])
    if ids[0, 0] != 0 or gap < 0.09:
        raise AssertionError("the tip must be the unique top document by at least 0.09")
    metadata = {
        "k": k, "documents": n, "dimension": d, "t": t, "distinct_fp32_documents": distinct,
        "max_document_norm_error": unit_error, "top_two_cosines": scores[0].tolist(),
        "tip_lead": gap, "max_raw_start_distance_from_anchor": float(displacement.max()),
    }
    return docs, query, initial, homes, metadata


class TrainingTrace:
    """Observe the trainer's initial centers, assignments and revival checks."""

    def __init__(self, model, homes):
        self.model = model
        self.homes = homes
        self.seen = np.zeros((model.nepochs, len(homes)), dtype=int)
        self.labels = np.full_like(self.seen, -1)
        self.initial = None
        self.batch_counts = np.zeros(model.k, dtype=int)
        self.assignment_errors = self.unbalanced = self.missing_home = self.steps = 0
        self.revival_ratios = []
        self.unused_targets = 0

    def observe(self, frame, event, result):
        model = self.model
        if event == "call" and frame.f_code is model.begin.__code__:
            self.initial = frame.f_locals["init_centroids"].copy()
        if event != "return":
            return
        if frame.f_code is model._assign_rows.__code__:
            # _accumulate_core passes the exact slice of its permuted document IDs.
            batch = frame.f_back.f_locals
            ids = batch["index"][batch["begin"]:batch["end"]]
            homes = self.homes[ids]
            labels = result[1]
            self.labels[model._epoch, ids] = labels
            np.add.at(self.seen[model._epoch], ids, 1)
            self.assignment_errors += int(np.count_nonzero(labels != homes))
            self.batch_counts += np.bincount(homes, minlength=model.k)
        elif frame.f_code is model._finish_step.__code__:
            self.unbalanced += int(self.batch_counts.min() != self.batch_counts.max())
            self.missing_home += int(self.batch_counts.min() == 0)
            self.steps += 1
            self.batch_counts.fill(0)
        elif frame.f_code is model._choose_splits.__code__:
            self.unused_targets += int(np.count_nonzero(frame.f_locals["unused"]))
            self.revival_ratios.append(float(model._usage.min() / model._usage.mean()))

    def report(self):
        if self.steps != self.model.recipe["total_steps"] or not np.all(self.seen == 1):
            raise AssertionError("incomplete trace of logical batches or epoch permutations")
        return {
            "all_epochs_visit_each_document_once": True,
            "assignment_errors_all_epochs": self.assignment_errors,
            "unbalanced_batches": self.unbalanced, "batches_missing_a_home": self.missing_home,
            "revival_checks": len(self.revival_ratios), "unused_targets_at_checks": self.unused_targets,
            "min_mass_over_mean_at_checks": min(self.revival_ratios), "splits": self.model.n_splits,
        }


def native_top1(index, query, probes):
    index.nprobe = probes
    faiss.cvar.indexIVF_stats.reset()
    _, ids = index.search(query, 1)
    return int(ids[0, 0]), int(faiss.cvar.indexIVF_stats.ndis)


def owner_scores(index, query, owner):
    scores, labels = index.quantizer.search(query, index.nlist)
    return scores[0, labels[0] == owner].item(), scores[0, labels[0] != owner]


def owner_rank(index, query, owner):
    owner_score, others = owner_scores(index, query, owner)
    if np.any(others == owner_score):
        raise AssertionError("the owner tied another list")
    return 1 + int(np.count_nonzero(others > owner_score)), owner_score, float(others.min())


def check_serving(docs, query, centers, homes):
    k, d = centers.shape
    quantizer = make_router(centers, "ip")
    index = faiss.IndexIVFFlat(quantizer, d, k, faiss.METRIC_INNER_PRODUCT)
    index.add(docs)
    stored, sizes = stored_lists(index)
    np.testing.assert_array_equal(stored, homes)
    np.testing.assert_array_equal(sizes, np.full(k, k))
    owner = stored[0].item()
    rank, owner_score, lowest_other = owner_rank(index, query, owner)
    miss, partial_scans = native_top1(index, query, k - 1)
    best, full_scans = native_top1(index, query, k)
    if rank != k or miss == 0 or best != 0 or (partial_scans, full_scans) != ((k - 1) * k, k * k):
        raise AssertionError("the normalized-mean index must first return the tip on the last probe")
    retuned = centers.copy()
    retuned[owner] = docs[0]
    install_flat_router(index, retuned)
    np.testing.assert_array_equal(stored_lists(index)[0], stored)
    new_rank, _, _ = owner_rank(index, query, owner)
    best, one_scan = native_top1(index, query, 1)
    if new_rank != 1 or best != 0 or one_scan != k:
        raise AssertionError("replacing just the owner representative must return the tip in one probe")
    return {
        "min_max_stored_list_size": [int(sizes.min()), int(sizes.max())],
        "owner_cosine": owner_score, "lowest_other_cosine": lowest_other,
        "owner_rank_before_after": [rank, new_rank], "native_scans_before_after": [full_scans, one_scan],
        "work_ratio": (k + full_scans) / (k + one_scan),
    }


def train_with_trace(trainer, docs, k, initial, homes, options, seed):
    model = trainer(docs.shape[1], k, seed=seed, spherical=True, **options)
    centers, labels = model.train(docs, init_centroids=initial)
    observed = trainer(docs.shape[1], k, seed=seed, spherical=True, **options)
    trace = TrainingTrace(observed, homes)
    previous = sys.getprofile()
    sys.setprofile(trace.observe)
    try:
        observed_centers, observed_labels = observed.train(docs, init_centroids=initial)
    finally:
        sys.setprofile(previous)
    np.testing.assert_array_equal(observed_centers.view("uint32"), centers.view("uint32"))
    np.testing.assert_array_equal(observed_labels, labels)
    np.testing.assert_array_equal(trace.labels[-1], labels)
    if model.n_splits != observed.n_splits:
        raise AssertionError("the observed replay changed the number of splits")
    return model, centers, labels, trace


def run(trainer, docs, query, initial, homes, options):
    k, d = initial.shape
    model, centers, labels, trace = train_with_trace(trainer, docs, k, initial, homes, options, SEED)
    _, frozen = model.assign(docs)
    np.testing.assert_array_equal(labels, homes)
    np.testing.assert_array_equal(frozen, homes)
    row_sums = docs.astype("float64").reshape(k, k, d).sum(axis=1)
    means = row_sums / np.linalg.norm(row_sums, axis=1)[:, None]
    mean_error = float(np.linalg.norm(centers - means, axis=1).max())
    training = trace.report()
    if training["assignment_errors_all_epochs"] or mean_error >= 2e-6:
        raise AssertionError("trajectories must keep the original labels and close at their normalized means")
    if training["unused_targets_at_checks"] or training["splits"]:
        raise AssertionError("a scheduled revival check left the proved no-split regime")
    recipe = {key: model.recipe[key] for key in (
        "name", "batch_law", "batch_size", "alpha0", "total_steps", "unused_threshold",
    )}
    return {
        "recipe": recipe, "training": training, "observed_replay_matches_uninstrumented": True,
        "moving_last_epoch_labels_match_rows": True, "frozen_labels_match_rows": True,
        "max_closing_mean_l2_error": mean_error, **check_serving(docs, query, centers, homes),
    }


def warm_start_cases(trainer):
    results = []
    for k in (2, 8, 32):
        docs, query, initial, homes, case = example(k)
        case["runs"] = {
            name: run(trainer, docs, query, initial, homes, options)
            for name, options in (
                ("mini_no_revival", {"batch_size": k - 1, "unused_threshold": 0}),
                ("full_default_threshold", {"batch_size": k * k}),
                ("default_recipe", {}),
            )
        }
        results.append(case)
    return results


def random_start_case(trainer, k):
    docs, query, _, homes, case = example(k)
    _, best = make_router(docs, "ip").search(query, 1)
    winner = best.item()
    target = homes == homes[winner]
    # The FP32 vectors are distinct. Match the exact inputs observed at begin to IDs.
    id_by_vector = {row.tobytes(): i for i, row in enumerate(docs)}
    epochs = 5
    options = {"batch_size": len(docs), "nepochs": epochs, "unused_threshold": 0}
    event_count = event_last = total_last = owner_ties = 0
    event_nonzero_owner = empty_closings = event_empty_closings = 0
    for seed in range(100):
        model, centers, closing_labels, trace = train_with_trace(
            trainer, docs, k, None, homes, options, seed
        )
        observed = trace.report()
        if (model.recipe["epoch_steps"] != [1] * epochs
                or model.recipe["epoch_rows"] != [len(docs)] * epochs
                or not 0 < model.recipe["alpha0"] <= 1 or observed["splits"]):
            raise AssertionError(f"unexpected full-batch recipe at k={k}, seed={seed}")
        seeds = np.array([id_by_vector[row.tobytes()] for row in trace.initial])
        if len(np.unique(seeds)) != k:
            raise AssertionError("the initializer must select k distinct document IDs")
        target_labels = np.flatnonzero(target[seeds])
        event = len(target_labels) == 1 and int(seeds[target_labels[0]]) != winner

        quantizer = make_router(centers, "ip")
        index = faiss.IndexIVFFlat(quantizer, docs.shape[1], k, faiss.METRIC_INNER_PRODUCT)
        index.add(docs)
        index.make_direct_map()
        stored = np.array([faiss.lo_listno(index.direct_map.get(i)) for i in range(len(docs))])
        _, frozen = model.assign(docs)
        np.testing.assert_array_equal(stored, frozen)
        owner = stored[winner].item()
        owner_score, others = owner_scores(index, query, owner)
        last = bool(np.all(owner_score < others))
        empty = len(np.unique(closing_labels)) < k
        if event:
            label = target_labels.item()
            target_by_epoch = np.broadcast_to(target, trace.labels.shape)
            np.testing.assert_array_equal(trace.labels == label, target_by_epoch)
            np.testing.assert_array_equal(stored == label, target)
            if not last:
                raise AssertionError(f"winner's list did not rank strictly last at k={k}, seed={seed}")
        event_count += event
        event_last += event and last
        total_last += last
        owner_ties += bool(np.any(owner_score == others))
        empty_closings += empty
        event_empty_closings += event and empty
        event_nonzero_owner += event and owner != 0
    return {
        "k": k, "documents": len(docs), "dimension": docs.shape[1], "t": case["t"],
        "winner_id": winner, "batch_size": len(docs), "alpha0": model.recipe["alpha0"],
        "seed_event": event_count, "seed_event_and_strict_last": event_last,
        "strict_last_outside_event": total_last - event_last, "strict_last_total": total_last,
        "runs_with_winner_list_score_ties": owner_ties,
        "runs_with_empty_closing_lists": empty_closings,
        "seed_event_with_empty_closing_lists": event_empty_closings,
        "seed_event_with_nonzero_winner_list_label": event_nonzero_owner,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faiss-checkout", type=Path, required=True)
    parser.add_argument("--random-starts", action="store_true", help="replay native starts for seeds 0–99")
    args = parser.parse_args()
    trainer = load_trainer(args.faiss_checkout)
    faiss.omp_set_num_threads(1)
    if args.random_starts:
        sweep = {
            "seeds_inclusive": [0, 99], "epochs": 5, "spherical": True, "unused_threshold": 0,
            "init_centroids": None, "seed_ids_observed_at_begin": True,
            "all_epochs_visit_each_document_once": True, "observed_replay_matches_uninstrumented": True,
            "fresh_assignments_match_native_ivf_lists": True,
            "seed_event_target_intact_all_epochs_and_final": True,
            "cases": [random_start_case(trainer, k) for k in (4, 8, 16)],
        }
    else:
        sweep = {
            "warm_start": "anchor[i] + (1/200) * address_axis[i,0], normalized inside begin",
            "seed": SEED, "epochs": 5, "spherical": True, "capture_assignments": True,
            "cases": warm_start_cases(trainer),
        }
    print(json.dumps({
        "environment": {
            "python": platform.python_version(), "faiss": faiss.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "architecture": platform.machine(), "faiss_threads": 1,
            "emeans_commit": COMMIT, "emeans_source_sha256": SOURCE_HASHES,
        },
        **sweep,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
