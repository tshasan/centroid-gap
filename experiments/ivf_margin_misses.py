"""Measure held-out cosine winner misses at a fixed lead on spherical IVF lists.

Stage both workloads with ivf_workload_data.py, then run
python experiments/ivf_margin_misses.py DATA OUTPUT [--workload deep|t2i|both].
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from statistics import NormalDist
from tempfile import TemporaryDirectory

import faiss
import numpy as np

if __package__:
    from .ivf_workload_data import BASE_ROWS, PUBLIC_SEED, SOURCES, TEST_ROWS, _sha256
    from .ivf_workload_index import N_LISTS, SEED, stored_lists
else:
    from ivf_workload_data import BASE_ROWS, PUBLIC_SEED, SOURCES, TEST_ROWS, _sha256
    from ivf_workload_index import N_LISTS, SEED, stored_lists


LEAD = 0.09
PROBES = (1, 2, 4, 8, 16, 32, 64)


def normalize(rows):
    """Keep source order and report every row for which cosine is undefined."""
    unit = np.array(rows, dtype=np.float64, order="C", copy=True)
    norms = np.sqrt(np.einsum("ij,ij->i", unit, unit))
    kept = np.flatnonzero(np.isfinite(norms) & (norms > 0))
    excluded = {"zero_norm": np.flatnonzero(norms == 0).tolist(),
                "nonfinite": np.flatnonzero(~np.isfinite(norms)).tolist()}
    if len(kept) < len(rows):
        unit = unit[kept]
    unit /= norms[kept, None]
    return unit, kept, excluded


def build_lists(documents, seed=SEED):
    """Train and add once. Recover membership from the populated inverted lists."""
    dim = documents.shape[1]
    coarse = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(coarse, dim, N_LISTS, faiss.METRIC_INNER_PRODUCT)
    index.cp.spherical = True
    index.cp.niter = 20
    index.cp.nredo = 2
    index.cp.seed = seed
    training = np.ascontiguousarray(documents, dtype=np.float32)
    index.train(training)
    index.add(training)
    owner, sizes = stored_lists(index)
    postings = []
    for label in range(index.nlist):
        size = index.invlists.list_size(label)
        ptr = index.invlists.get_ids(label)
        try:
            postings.append(np.sort(faiss.rev_swig_ptr(ptr, size)))
        finally:
            index.invlists.release_ids(label, ptr)
    parameters = {
        "metric": "inner_product", "spherical": bool(index.cp.spherical),
        "lists": index.nlist, "niter": index.cp.niter, "nredo": index.cp.nredo,
        "seed": index.cp.seed, "max_points_per_centroid": index.cp.max_points_per_centroid,
        "training_rows": min(len(training), index.nlist * index.cp.max_points_per_centroid),
    }
    return owner, sizes, postings, parameters


def full_data_means(documents, postings):
    means = np.stack([documents[ids].mean(axis=0) for ids in postings])
    means, valid, _ = normalize(means)
    if len(valid) != len(postings):
        raise ValueError("A posting list has no defined mean direction")
    return means


def top_in_block(queries, documents, ids, k):
    """Score the entire block, then resolve close candidates with one dot-product order."""
    k = min(k, len(ids))
    scores = queries @ documents.T
    rows = np.arange(len(queries))
    first = scores.argmax(axis=1)
    cutoff = scores[rows, first]
    if k == 2:
        scores[rows, first] = -np.inf
        second = scores.argmax(axis=1)
        scores[rows, first] = cutoff
        cutoff = scores[rows, second]

    # The absolute dot-product roundoff on unit rows is O(d * eps). Retain every
    # contender within this guard of the kth score, including duplicate vectors.
    # The guard chooses what to rescore; it never defines a tie or a margin.
    guard = 8 * documents.shape[1] * np.finfo(np.float64).eps
    q, d = np.nonzero(scores >= cutoff[:, None] - guard)
    refined = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), 8192):
        batch = slice(start, start + 8192)
        refined[batch] = np.einsum("ij,ij->i", queries[q[batch]], documents[d[batch]])
    by_score = np.lexsort((ids[d], -refined, q))
    first_in_group = np.r_[0, np.cumsum(np.bincount(q, minlength=len(queries)))[:-1]]
    best = by_score[first_in_group[:, None] + np.arange(k)]
    return refined[best], ids[d[best]]


def exhaustive_top_two(queries, documents):
    """Compare every query with every normalized document in bounded-memory blocks."""
    if len(documents) < 2:
        raise ValueError("Two valid documents are required")
    best_scores = np.full((len(queries), 2), -np.inf)
    best_ids = np.full((len(queries), 2), len(documents), dtype=np.int64)
    document_ids = np.arange(len(documents))
    for start in range(0, len(documents), 8192):
        chunk = slice(start, start + 8192)
        for row in range(0, len(queries), 512):
            batch = slice(row, row + 512)
            scores, ids = top_in_block(queries[batch], documents[chunk], document_ids[chunk], 2)
            scores = np.column_stack((best_scores[batch], scores))
            ids = np.column_stack((best_ids[batch], ids))
            take = np.lexsort((ids, -scores), axis=1)[:, :2]
            best_scores[batch] = np.take_along_axis(scores, take, axis=1)
            best_ids[batch] = np.take_along_axis(ids, take, axis=1)
    return best_scores, best_ids


def route(queries, means, owners, probes):
    scores = np.einsum("ij,kj->ik", queries, means)
    order = np.argsort(-scores, axis=1, kind="stable")  # list ID breaks score ties
    rows = np.arange(len(queries))
    rank = (order == owners[:, None]).argmax(axis=1) + 1
    tie_size = (scores == scores[rows, owners, None]).sum(axis=1)
    boundary_ties = {
        p: int((scores[rows, order[:, p-1]] == scores[rows, order[:, p]]).sum())
        for p in probes if p < len(means)
    }
    return order[:, :max(probes)], rank, tie_size, boundary_ties


def scan_selected(queries, documents, postings, order, probes):
    """Independently visit and score the actual documents in every selected posting."""
    nqueries, width = order.shape
    # Group (query, probe position) pairs by list to score a posting in one batch.
    flat = order.ravel()
    by_list = np.argsort(flat, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(flat, minlength=len(postings)))]
    scores = np.empty(order.shape)
    ids = np.empty(order.shape, dtype=np.int64)
    scanned = np.empty_like(ids)
    for label, members in enumerate(postings):
        positions = by_list[bounds[label]:bounds[label + 1]]
        if not len(positions):
            continue
        rows, columns = np.divmod(positions, width)
        best_score, best_id = top_in_block(queries[rows], documents[members], members, 1)
        scores[rows, columns] = best_score[:, 0]
        ids[rows, columns] = best_id[:, 0]
        scanned[rows, columns] = len(members)

    returned = np.empty((nqueries, len(probes)), dtype=np.int64)
    high = np.full(nqueries, -np.inf)
    winner = np.full(nqueries, len(documents), dtype=np.int64)
    for column in range(width):
        better = (scores[:, column] > high) | ((scores[:, column] == high) & (ids[:, column] < winner))
        high[better] = scores[better, column]
        winner[better] = ids[better, column]
        if column + 1 in probes:
            returned[:, probes.index(column + 1)] = winner
    return returned, scanned.cumsum(axis=1)[:, np.array(probes) - 1]


def wilson95(count, n):
    if not n:
        return None
    z = NormalDist().inv_cdf(0.975)
    rate = count / n
    center = count + z*z/2
    half = z * math.sqrt(n * rate * (1-rate) + z*z/4)
    return [0.0 if count == 0 else (center-half) / (n + z*z),
            1.0 if count == n else (center+half) / (n + z*z)]


def work(scans, mask, nlist):
    n = int(mask.sum())
    documents = int(scans[mask].sum())
    total = documents + nlist * n
    return {"queries": n, "documents_scanned": documents, "total_scores": total,
            "mean_documents_scanned": documents / n if n else None,
            "mean_total_scores": total / n if n else None}


def load_workload(data, name):
    manifest = json.loads((data / f"{name}-workload.json").read_text())
    dim = SOURCES[name][0].dimensions
    if manifest["public_permutation_seed"] != PUBLIC_SEED:
        raise ValueError("Unexpected public query split")
    arrays, hashes = {}, {}
    test_rows = manifest.get("test_rows", TEST_ROWS)
    for key, shape in (("base", (BASE_ROWS, dim)), ("test", (test_rows, dim))):
        path = data / f"{name}-{key}.npy"
        hashes[key] = _sha256(path)
        arrays[key] = np.load(path, mmap_mode="r")
        if hashes[key] != manifest["outputs"][key]["sha256"] or arrays[key].shape != shape:
            raise ValueError(f"{path}: expected the staged million-document experiment data")
    source_ids = manifest["selected_source_ids"]["test"]
    if len(set(source_ids)) != test_rows or len(source_ids) != test_rows:
        raise ValueError("Unexpected held-out public source IDs")
    public_exclusions = {key: len(ids) for key, ids in manifest["excluded_source_ids"]["public"].items()}
    return arrays["base"], arrays["test"], source_ids, hashes, public_exclusions


def experiment(data, output, name, snapshot_dir, seed=SEED):
    raw_docs, raw_queries, public_ids, hashes, public_exclusions = load_workload(data, name)
    docs, doc_rows, doc_exclusions = normalize(raw_docs)
    queries, query_rows, query_exclusions = normalize(raw_queries)
    print(f"{name}: training spherical lists on {len(docs):,} normalized database rows", flush=True)
    owner, sizes, postings, training = build_lists(docs, seed)
    means = full_data_means(docs, postings)
    source_owner = np.full(len(raw_docs), -1, dtype=np.int32)
    source_owner[doc_rows] = owner
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=snapshot_dir) as temporary:
            path = Path(temporary) / f"{name}-lists.npz"
            np.savez_compressed(path, owner=source_owner, means=means, sizes=sizes)
            path.replace(snapshot_dir / path.name)

    print(f"{name}: exhaustive top two for {len(queries):,} held-out queries", flush=True)
    scores, top_ids = exhaustive_top_two(queries, docs)
    winner, runner_up = top_ids.T
    leads = scores[:, 0] - scores[:, 1]
    eligible = leads > 0
    qualifies = eligible & (leads >= LEAD)
    owners = owner[winner]
    order, ranks, tie_size, boundary_ties = route(queries, means, owners, PROBES)
    print(f"{name}: {qualifies.sum():,}/{eligible.sum():,} qualify; checking selected-list scans", flush=True)
    returned, scans = scan_selected(queries, docs, postings, order, PROBES)
    counts_from_sizes = np.cumsum(sizes[order], axis=1)[:, np.array(PROBES) - 1]
    if not np.array_equal(scans, counts_from_sizes):
        raise AssertionError("Actual docs scored and the sum of the selected list sizes disagree")
    if not np.array_equal(returned == winner[:, None], ranks[:, None] <= np.array(PROBES)):
        raise AssertionError("Exact scan winner IDs and owner-list ranks disagree")

    records = [dict(split_row=i, public_query_row=public, eligible=0, exclusion="",
                    qualifies=0) for i, public in enumerate(public_ids)]
    for reason, rows in query_exclusions.items():
        for row in rows:
            records[row]["exclusion"] = reason
    for i, row in enumerate(query_rows):
        record = records[row]
        record.update(eligible=int(eligible[i]), exclusion="" if eligible[i] else "top_score_tie",
                      qualifies=int(qualifies[i]), winner_id=doc_rows[winner[i]], winner_cosine=scores[i, 0],
                      runner_up_id=doc_rows[runner_up[i]], runner_up_cosine=scores[i, 1],
                      cosine_lead=leads[i], owner_list=owners[i], owner_rank=ranks[i],
                      owner_routing_tie_size=tie_size[i])
        for j, p in enumerate(PROBES):
            record[f"scan_winner_p{p}"] = doc_rows[returned[i, j]]
            record[f"documents_scanned_p{p}"] = scans[i, j]
            record[f"total_scores_p{p}"] = scans[i, j] + len(means)

    n, nq = int(eligible.sum()), int(qualifies.sum())
    probe_results = []
    for j, p in enumerate(PROBES):
        miss = int(np.count_nonzero(qualifies & (ranks > p)))
        probe_results.append({
            "p": p, "eligible_queries": n, "qualifying_queries": nq, "misses": miss,
            "conditional_miss_rate": miss / nq if nq else None, "conditional_wilson_95": wilson95(miss, nq),
            "joint_fraction_of_eligible": miss / n if n else None,
            "work": {"eligible": work(scans[:, j], eligible, len(means)),
                     "qualifying": work(scans[:, j], qualifies, len(means))},
            "routing_boundary_ties": boundary_ties[p], "scan_winner_disagreements": 0,
            "scan_count_disagreements": 0,
        })
    summary = {
        "workload": name, "split": "test", "lead_threshold": LEAD,
        "tested_queries": len(records), "eligible_queries": n, "qualifying_queries": nq,
        "excluded_queries": {**{reason: len(rows) for reason, rows in query_exclusions.items()},
                             "top_score_tie": int((~eligible).sum())},
        "public_exclusions_before_split": public_exclusions, "public_permutation_seed": PUBLIC_SEED,
        "input_npy_sha256": hashes, "base_prefix_rows": len(raw_docs), "indexed_documents": len(docs),
        "excluded_document_rows": doc_exclusions, "dimension": docs.shape[1], "training": training,
        "list_size_min_max": [int(sizes.min()), int(sizes.max())],
        "source_owner_sha256_le_i4": hashlib.sha256(source_owner.astype("<i4", copy=False)).hexdigest(),
        "normalized_mean_sha256_le_f8": hashlib.sha256(means.astype("<f8", copy=False)).hexdigest(),
        "document_ties": "Equal float64 top scores use ascending source document ID. Tied winners have zero lead and are excluded from eligible counts.",
        "routing_ties": "Equal float64 scores use ascending list ID. Routing ties remain eligible.",
        "queries_with_owner_routing_tie": int((tie_size > 1).sum()),
        "scan_verified_queries_per_budget": len(queries),
        "interval": "Pointwise two-sided 95% Wilson score interval for misses among qualifying queries.",
        "python": platform.python_version(), "numpy": np.__version__, "faiss": faiss.__version__,
        "faiss_threads": faiss.omp_get_max_threads(), "probes": probe_results,
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        query_path = Path(temporary) / f"{name}-queries.csv"
        columns = ["split_row", "public_query_row", "eligible", "exclusion", "qualifies",
                   "winner_id", "winner_cosine", "runner_up_id", "runner_up_cosine", "cosine_lead",
                   "owner_list", "owner_rank", "owner_routing_tie_size"]
        for p in PROBES:
            columns.extend([f"scan_winner_p{p}", f"documents_scanned_p{p}", f"total_scores_p{p}"])
        with query_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        summary_path = Path(temporary) / f"{name}-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        query_path.replace(output / query_path.name)
        summary_path.replace(output / summary_path.name)
    print(f"{name}: wrote {output / query_path.name} and {output / summary_path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="Directory staged by ivf_workload_data.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workload", choices=("deep", "t2i", "both"), default="both")
    parser.add_argument("--threads", type=int, default=4, help="Faiss CPU threads")
    parser.add_argument("--snapshot-dir", type=Path, help="Also save frozen owners, means and sizes")
    parser.add_argument("--seed", type=int, default=SEED, help="Faiss k-means seed for the frozen lists")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    faiss.omp_set_num_threads(args.threads)
    for workload in SOURCES if args.workload == "both" else (args.workload,):
        experiment(args.data, args.output, workload, args.snapshot_dir, args.seed)
