"""Fit coarse representatives for a fixed Faiss IVF index using held-out list coverage."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def squared_norms(rows):
    return np.einsum("ij,ij->i", rows, rows)


def fit_space(reps, x, metric):
    if metric == "ip":
        norms = np.sqrt(squared_norms(x.astype(np.float64)))[:, None]
        x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
        scale = np.sqrt(squared_norms(reps.astype(np.float64)).mean()) or 1.0
        return reps / scale, x, 0.0, scale

    # Center and scale both operands together, preserving Euclidean rankings.
    origin = reps.mean(axis=0, dtype=np.float64)
    reps = reps - origin
    x = x - origin
    scale = np.sqrt(max(squared_norms(reps).mean(), squared_norms(x).mean())) or 1.0
    return reps / scale, x / scale, origin, scale


@dataclass
class AdamRouterState:
    """Explicit state machine for Adam optimization of IVF router representatives."""

    reps: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    steps: int = 0
    epoch: int = 0

    @classmethod
    def init(cls, initial: np.ndarray) -> "AdamRouterState":
        reps = initial.copy()
        return cls(
            reps=reps,
            first_moment=np.zeros_like(reps),
            second_moment=np.zeros_like(reps),
        )

    def step(self, batch: np.ndarray, batch_home: np.ndarray, initial: np.ndarray,
             metric: str, regularization: float, rate: float, temperature: float = 12.0) -> None:
        error = batch @ self.reps.T
        if metric == "l2":
            error -= 0.5 * squared_norms(self.reps)
        error *= temperature
        error -= error.max(axis=1, keepdims=True)
        np.exp(error, out=error)
        error /= error.sum(axis=1, keepdims=True)
        error[np.arange(len(batch)), batch_home] -= 1.0

        gradient = error.T @ batch
        if metric == "l2":
            gradient -= error.sum(axis=0)[:, None] * self.reps
        gradient *= temperature / len(batch)
        gradient += (regularization / len(self.reps)) * (self.reps - initial)
        self.steps += 1
        self.first_moment = 0.9 * self.first_moment + 0.1 * gradient
        self.second_moment = 0.999 * self.second_moment + 0.001 * gradient**2
        corrected_first = self.first_moment / (1 - 0.9**self.steps)
        corrected_second = self.second_moment / (1 - 0.999**self.steps)
        self.reps -= rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)


def softmax_reps(initial, x, home, metric, regularization):
    """Optimize the serving score itself, with a penalty toward the original reps."""
    state = AdamRouterState.init(initial)
    rng = np.random.default_rng(2026)
    batch_size = max(1, min(512, (1 << 20) // len(state.reps)))
    for epoch in range(6):
        state.epoch = epoch
        order = rng.permutation(len(x))
        rate = (0.06 - 0.008 * epoch) / np.sqrt(state.reps.shape[1])
        for start in range(0, len(x), batch_size):
            rows = order[start:start + batch_size]
            state.step(x[rows], home[rows], initial, metric, regularization, rate)
    return state.reps


def fit_candidates(reps, x, home, metric):
    """Return float32 representatives in their original list-ID order."""
    reps = np.asarray(reps, dtype=np.float32, order="C")
    x = np.asarray(x, dtype=np.float32, order="C")
    home = np.asarray(home)
    if metric not in ("l2", "ip"):
        raise ValueError("metric must be 'l2' or 'ip'")
    if (reps.ndim != 2 or not all(reps.shape) or x.ndim != 2
            or len(x) == 0 or x.shape[1] != reps.shape[1]
            or not np.isfinite(reps).all() or not np.isfinite(x).all()):
        raise ValueError("representatives and queries need compatible, nonempty finite matrices")
    if (home.shape != (len(x),) or home.dtype.kind not in "iu"
            or np.any(home < 0) or np.any(home >= len(reps))):
        raise ValueError("home must provide one valid list ID per training query")
    home = home.astype(np.int64, copy=False)
    initial, queries, origin, scale = fit_space(reps, x, metric)
    initial = np.ascontiguousarray(initial, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    counts = np.bincount(home, minlength=len(reps))[:, None]
    sums = np.zeros(reps.shape, dtype=np.float64)
    np.add.at(sums, home, queries)

    candidates = {"original": reps.copy()}
    for prior in (8, 32, 128):
        updated = (sums + prior * initial) / (counts + prior)
        means = np.ascontiguousarray(updated * scale + origin, dtype=np.float32)
        means[counts[:, 0] == 0] = reps[counts[:, 0] == 0]
        candidates[f"query_mean_prior_{prior}"] = means
    for regularization in (0.1, 1.0):
        updated = softmax_reps(initial, queries, home, metric, regularization)
        candidates[f"supervised_softmax_reg_{regularization:g}"] = np.ascontiguousarray(
            updated * scale + origin, dtype=np.float32
        )
    if not all(np.isfinite(value).all() for value in candidates.values()):
        raise ValueError("fitting overflowed float32 representatives")
    return candidates


def make_router(reps, metric):
    """Build one flat router with row numbers equal to original list IDs."""
    import faiss

    router_type = {"l2": faiss.IndexFlatL2, "ip": faiss.IndexFlatIP}.get(metric)
    if router_type is None:
        raise ValueError("metric must be 'l2' or 'ip'")
    reps = np.ascontiguousarray(reps, dtype=np.float32)
    if reps.ndim != 2 or not all(reps.shape) or not np.isfinite(reps).all():
        raise ValueError("representatives must form a nonempty finite matrix")
    router = router_type(reps.shape[1])
    router.add(reps)
    return router


def install_flat_router(index, reps):
    """Replace the centroid table of a dedicated, mutable IVF-Flat index in RAM.

    Install before serving concurrent searches. Stored vectors stay in their
    original lists. Future adds use the new representatives for assignment.
    """
    import faiss

    metrics = (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT)
    if (type(index) is not faiss.IndexIVFFlat or index.by_residual
            or not index.is_trained or index.metric_type not in metrics):
        raise ValueError("install requires a trained, non-residual CPU IndexIVFFlat with L2 or IP")
    quantizer = faiss.downcast_index(index.quantizer)
    flat_types = (faiss.IndexFlat, faiss.IndexFlatL2, faiss.IndexFlatIP)
    if (type(quantizer) not in flat_types or quantizer.metric_type != index.metric_type
            or quantizer.d != index.d or quantizer.ntotal != index.nlist):
        raise ValueError("install requires a bare flat coarse quantizer with one row per list")
    shape = (index.nlist, index.d)
    reps = np.asarray(reps)
    if reps.shape != shape or reps.dtype.kind != "f":
        raise ValueError(f"representatives must be a floating-point array of shape {shape}")
    reps = np.ascontiguousarray(reps, dtype=np.float32)
    if not np.isfinite(reps).all():
        raise ValueError("representatives must be finite in float32")

    table = faiss.rev_swig_ptr(quantizer.get_xb(), index.nlist * index.d).reshape(shape)
    np.copyto(table, reps)
    if isinstance(quantizer, faiss.IndexFlatL2):
        quantizer.clear_l2norms()


def original_coarse_scores(index, queries, lists):
    """Compute the scores the scanner expects for these original list IDs."""
    import faiss

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    lists = np.ascontiguousarray(lists, dtype=np.int64)
    if (queries.ndim != 2 or queries.shape[1] != index.d or lists.ndim != 2
            or len(lists) != len(queries) or np.any((lists < 0) | (lists >= index.nlist))):
        raise ValueError("invalid queries or original list IDs")
    quantizer = faiss.downcast_index(index.quantizer)
    if isinstance(quantizer, faiss.IndexFlat):
        scores = np.empty(lists.shape, dtype=np.float32)
        quantizer.compute_distance_subset(
            len(queries), faiss.swig_ptr(queries), lists.shape[1],
            faiss.swig_ptr(scores), faiss.swig_ptr(lists)
        )
        return scores

    centers = quantizer.reconstruct_batch(lists.ravel()).reshape(*lists.shape, index.d)
    if index.metric_type == faiss.METRIC_INNER_PRODUCT:
        return np.einsum("qd,qpd->qp", queries, centers)
    centers -= queries[:, None, :]
    return np.einsum("qpd,qpd->qp", centers, centers)


def search_with_router(index, fitted_router, queries, topk, nprobe, *, stats=None):
    """Scan the original CPU IVF lists using a separate fitted flat router."""
    import faiss

    if type(index) not in (faiss.IndexIVFFlat, faiss.IndexIVFPQ):
        raise ValueError("search requires a CPU IndexIVFFlat or IndexIVFPQ")
    if (fitted_router.ntotal != index.nlist or fitted_router.metric_type != index.metric_type
            or fitted_router.d != index.d or not 1 <= nprobe <= index.nlist or topk < 1):
        raise ValueError("invalid router shape, metric, topk or nprobe")
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    coarse_scores, lists = fitted_router.search(queries, nprobe)
    if np.any((lists < 0) | (lists >= index.nlist)) or not np.isfinite(coarse_scores).all():
        raise ValueError("routing produced invalid scores or list IDs")
    if isinstance(index, faiss.IndexIVFPQ) and index.by_residual:
        # PQ codes were encoded against the original centroids.
        coarse_scores = original_coarse_scores(index, queries, lists)
    scores = np.empty((len(queries), topk), dtype=np.float32)
    ids = np.empty((len(queries), topk), dtype=np.int64)
    params = faiss.SearchParametersIVF()
    params.nprobe = nprobe
    params.max_codes = index.max_codes
    # Faiss 1.12's NumPy wrapper cannot pass per-call parameters.
    index.search_preassigned_c(
        len(queries), faiss.swig_ptr(queries), topk, faiss.swig_ptr(lists),
        faiss.swig_ptr(coarse_scores), faiss.swig_ptr(scores), faiss.swig_ptr(ids),
        False, params, stats
    )
    return scores, ids


@dataclass
class ProbeCalibrationState:
    """Explicit state machine for monotonic probe-budget calibration."""

    probe: int
    max_probe: int
    target_recall: float
    hits: int = 0
    covered_top10: int = 0
    scanned: int = 0
    coverage: float = 0.0
    done: bool = False

    def evaluate(self, router, queries, home, sizes, batch_size, home_top10, select_at) -> bool:
        hits = covered_top10 = scanned = 0
        for start in range(0, len(queries), batch_size):
            rows = slice(start, start + batch_size)
            _, order = router.search(queries[rows], self.probe)
            hits += int(np.any(order == home[rows, None], axis=1).sum())
            scanned += int(sizes[order].sum())
            if home_top10 is not None:
                covered_top10 += sum(
                    int(np.any(order == one_neighbor_list[:, None], axis=1).sum())
                    for one_neighbor_list in home_top10[rows].T
                )
        self.hits = hits
        self.covered_top10 = covered_top10
        self.scanned = scanned
        self.coverage = hits / len(queries) if select_at == 1 else covered_top10 / home_top10.size
        if self.coverage >= self.target_recall:
            self.done = True
            return True
        if self.probe == self.max_probe:
            raise ValueError("validation coverage missed the target even when probing all lists")
        self.probe = min(self.max_probe, max(self.probe + 1, int(self.probe * 1.1)))
        return False


def validate_router(reps, queries, home, sizes, metric, target_recall, home_top10=None, *, select_at=1):
    if select_at not in (1, 10) or (select_at == 10 and home_top10 is None):
        raise ValueError("top-10 selection requires top-10 list labels")
    router = make_router(reps, metric)
    batch_size = max(1, min(128, (1 << 20) // len(reps)))
    ranks = np.empty(len(queries), dtype=np.int64)
    top10_ranks = np.empty_like(home_top10) if select_at == 10 else None
    for start in range(0, len(queries), batch_size):
        rows = slice(start, start + batch_size)
        distances, order = router.search(queries[rows], len(reps))
        if np.any(order < 0) or not np.isfinite(distances).all():
            raise ValueError("validation routing produced invalid scores or list IDs")
        ranks[rows] = 1 + (order == home[rows, None]).argmax(axis=1)
        if top10_ranks is not None:
            row_ids = np.arange(len(order))[:, None]
            positions = np.empty_like(order)
            positions[row_ids, order] = np.arange(1, len(reps) + 1)
            top10_ranks[rows] = positions[row_ids, home_top10[rows]]

    selection_ranks = ranks if top10_ranks is None else top10_ranks
    initial_probe = int(np.quantile(selection_ranks, target_recall, method="inverted_cdf"))
    # Measure the actual top-probe search. Equal centroid scores can change the
    # order returned at different k values, so increase the cutoff if needed.
    cal = ProbeCalibrationState(probe=initial_probe, max_probe=len(reps), target_recall=target_recall)
    while not cal.evaluate(router, queries, home, sizes, batch_size, home_top10, select_at):
        pass

    result = {
        "probe": cal.probe,
        "routing_recall_at_1": cal.hits / len(queries),
        "mean_scanned": cal.scanned / len(queries),
        "mean_routing": len(reps),
        "mean_total": len(reps) + cal.scanned / len(queries),
        "true_list_ranks_p50_p95_p99": np.quantile(
            ranks, [0.5, 0.95, 0.99], method="inverted_cdf"
        ).tolist(),
    }
    if home_top10 is not None:
        result["routing_recall_at_10"] = cal.covered_top10 / home_top10.size
    return result


def load_queries(path, dimension):
    queries = np.load(path, allow_pickle=False)
    if (not isinstance(queries, np.ndarray) or queries.dtype.kind != "f"
            or queries.ndim != 2 or queries.shape[1] != dimension or len(queries) == 0):
        raise ValueError(f"{path} must contain a nonempty floating-point (n, d) array")
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if not np.isfinite(queries).all():
        raise ValueError(f"{path} contains queries outside finite float32 range")
    return queries


def load_ids(path, shape):
    ids = np.load(path, allow_pickle=False)
    if (not isinstance(ids, np.ndarray) or ids.dtype.kind != "i"
            or ids.shape != shape or np.any(ids == -1)):
        raise ValueError(f"{path} must contain signed integer IDs with shape {shape} and no -1")
    return ids


def find_lists(index, truth):
    # Track only labeled IDs. Read stored membership instead of reassigning vectors.
    wanted, inverse = np.unique(truth, return_inverse=True)
    home = np.full(len(wanted), -1, dtype=np.int64)
    sizes = np.empty(index.nlist, dtype=np.int64)
    for bucket in range(index.nlist):
        size = sizes[bucket] = index.invlists.list_size(bucket)
        if size == 0:
            continue
        pointer = index.invlists.get_ids(bucket)
        try:
            ids = __import__("faiss").rev_swig_ptr(pointer, int(size))
            for start in range(0, size, 1 << 18):
                chunk = ids[start:start + (1 << 18)]
                positions = np.searchsorted(wanted, chunk)
                positions = np.minimum(positions, len(wanted) - 1)
                hits = positions[wanted[positions] == chunk]
                unique, counts = np.unique(hits, return_counts=True)
                if np.any(counts > 1) or np.any(home[unique] >= 0):
                    raise ValueError("a supplied truth ID is stored more than once")
                home[unique] = bucket
        finally:
            index.invlists.release_ids(bucket, pointer)
    if np.any(home < 0) or sizes.sum() != index.ntotal:
        raise ValueError("supplied truth IDs or stored list sizes do not match the index")
    return home[inverse], sizes


def fit_snapshot(args):
    import faiss

    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(str(args.snapshot), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    metrics = {faiss.METRIC_L2: "l2", faiss.METRIC_INNER_PRODUCT: "ip"}
    if (type(index) not in (faiss.IndexIVFFlat, faiss.IndexIVFPQ)
            or index.metric_type not in metrics or not index.is_trained or index.ntotal == 0):
        raise ValueError("snapshot must be a trained, nonempty CPU IVF-Flat or IVF-PQ with L2 or IP")
    lists = faiss.downcast_InvertedLists(index.invlists)
    if isinstance(lists, faiss.OnDiskInvertedLists) and lists.filename:
        raise ValueError("snapshot must contain its inverted lists in the same file")
    router = faiss.downcast_index(index.quantizer)
    if router.ntotal != index.nlist or router.metric_type != index.metric_type:
        raise ValueError("coarse router must provide one representative per list in the index metric")
    reps = router.reconstruct_n(0, index.nlist)
    metric = metrics[index.metric_type]
    if args.train_queries.samefile(args.validation_queries):
        raise ValueError("training and validation queries must come from separate files")
    train = load_queries(args.train_queries, index.d)
    valid = load_queries(args.validation_queries, index.d)
    train_ids = load_ids(args.train_top1_ids, (len(train),))
    valid_ids = load_ids(args.validation_top1_ids, (len(valid),))
    id_sets = [train_ids, valid_ids]
    top10 = None
    if args.validation_top10_ids is not None:
        top10 = load_ids(args.validation_top10_ids, (len(valid), 10))
        if not np.all(np.any(top10 == valid_ids[:, None], axis=1)) or np.any(
            np.diff(np.sort(top10, axis=1), axis=1) == 0
        ):
            raise ValueError("each validation top-10 row must contain its top-1 ID and ten distinct IDs")
        id_sets.append(top10.ravel())
    homes, sizes = find_lists(index, np.concatenate(id_sets))
    train_home = homes[:len(train)]
    valid_home = homes[len(train):len(train) + len(valid)]
    valid_home_top10 = homes[len(train) + len(valid):].reshape(-1, 10) if top10 is not None else None
    candidates = fit_candidates(reps, train, train_home, metric)
    results = {
        name: validate_router(
            value, valid, valid_home, sizes, metric, args.target_recall, valid_home_top10,
            select_at=args.select_at
        )
        for name, value in candidates.items()
    }
    coverage_key = f"routing_recall_at_{args.select_at}"
    selected = min(results, key=lambda name: (
        results[name]["mean_total"], -results[name][coverage_key]
    ))
    # Publish only a completed array. Linking fails if the output already exists.
    with NamedTemporaryFile(dir=args.output_reps.parent) as output:
        np.save(output, candidates[selected], allow_pickle=False)
        output.flush()
        os.link(output.name, args.output_reps)
    return {
        "metric": metric,
        "index_type": type(index).__name__,
        "lists": index.nlist,
        "training_queries": len(train),
        "validation_queries": len(valid),
        "truth": "caller_supplied",
        "validation_use": "router_selection",
        "routing": "exhaustive_flat_scoring",
        "work_model": "representatives_scored_plus_candidate_vectors",
        f"target_{coverage_key}": args.target_recall,
        "selected": selected,
        "output_reps": str(args.output_reps),
        "validation": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("train_queries", type=Path)
    parser.add_argument("train_top1_ids", type=Path)
    parser.add_argument("validation_queries", type=Path)
    parser.add_argument("validation_top1_ids", type=Path)
    parser.add_argument("output_reps", type=Path, help="new .npy file for selected representatives")
    parser.add_argument("--validation-top10-ids", type=Path, help="optional (n, 10) exact ID array")
    parser.add_argument("--target-recall", type=float, default=0.95, help="validation recall target")
    parser.add_argument("--select-at", type=int, choices=(1, 10), default=1, help="top-1 or top-10")
    parser.add_argument("--threads", type=int, default=4, help="Faiss validation threads")
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1 or args.threads < 1:
        parser.error("--target-recall must be in (0, 1] and --threads must be positive")
    if args.select_at == 10 and args.validation_top10_ids is None:
        parser.error("--select-at 10 requires --validation-top10-ids")
    try:
        report = fit_snapshot(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
