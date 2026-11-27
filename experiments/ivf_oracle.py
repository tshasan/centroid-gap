"""Measure fixed-probe IVF routing using exact neighbors and stored list sizes."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np


_BATCH_SIZE = 512
_PERCENTILES = (0.5, 0.95, 0.99)


def _truth_lists(truth, owner, sizes):
    truth, owner, sizes = map(np.asarray, (truth, owner, sizes))
    if sizes.ndim != 1 or not sizes.size or sizes.dtype.kind not in "iu":
        raise ValueError("sizes must be a nonempty integer vector")
    if (owner.ndim != 1 or not owner.size or owner.dtype.kind not in "iu"
            or owner.min() < 0 or owner.max() >= len(sizes)):
        raise ValueError("owner must be a vector of valid stored list IDs")
    if not np.array_equal(np.bincount(owner.astype(np.intp, copy=False), minlength=len(sizes)), sizes):
        raise ValueError("sizes must count the actual owners")
    if (truth.ndim != 2 or not truth.size or truth.dtype.kind not in "iu"
            or truth.min() < 0 or truth.max() >= len(owner)):
        raise ValueError("truth must be a nonempty matrix of valid vector IDs")
    if np.any(np.diff(np.sort(truth, axis=1), axis=1) == 0):
        raise ValueError("truth IDs must be distinct within each query")
    return owner[truth], sizes


def _oracle_order(homes, sizes):
    # The stable hit-count sort inherits the ascending size and ID tie order.
    by_size = np.argsort(sizes, kind="stable")
    order = np.empty((len(homes), len(sizes)), dtype=np.int64)
    for start in range(0, len(homes), _BATCH_SIZE):
        batch = homes[start:start + _BATCH_SIZE]
        counts = np.zeros((len(batch), len(sizes)), dtype=np.int32)
        np.add.at(counts, (np.arange(len(batch))[:, None], batch), 1)
        order[start:start + len(batch)] = by_size[
            np.argsort(-counts[:, by_size], axis=1, kind="stable")
        ]
    return order


def oracle_order(truth_subset, owner, sizes):
    """Return list permutations for the supplied truth IDs, in signed int64."""
    homes, sizes = _truth_lists(truth_subset, owner, sizes)
    return _oracle_order(homes, sizes)


def _ranks(order, homes):
    ranks = np.empty(homes.shape, dtype=np.int64)
    count = order.shape[1]
    positions = np.arange(1, count + 1)
    for start in range(0, len(order), _BATCH_SIZE):
        batch = order[start:start + _BATCH_SIZE]
        if batch.min() < 0 or batch.max() >= count:
            raise ValueError("routing order contains an invalid list ID")
        inverse = np.zeros(batch.shape, dtype=np.int64)
        np.put_along_axis(inverse, batch, positions[None, :], axis=1)
        if np.any(inverse == 0):
            raise ValueError("each routing order row must be a permutation of list IDs")
        stop = start + len(batch)
        ranks[start:stop] = np.take_along_axis(inverse, homes[start:stop], axis=1)
    return ranks


def _summary(values):
    p50, p95, p99 = np.quantile(values, _PERCENTILES, method="inverted_cdf")
    return {"p50": int(p50), "p95": int(p95), "p99": int(p99)}


def _measure(order, sizes, ranks, probes, return_details):
    scans = np.empty(len(order), dtype=np.int64)
    for start in range(0, len(order), _BATCH_SIZE):
        prefix = order[start:start + _BATCH_SIZE, :probes]
        scans[start:start + len(prefix)] = sizes[prefix].sum(axis=1, dtype=np.int64)
    hits = np.count_nonzero(ranks <= probes, axis=1)
    scan_summary = {"mean": float(scans.mean()), **_summary(scans)}
    result = {
        "probes": probes,
        "recall": int(hits.sum()) / ranks.size,
        "any_hit": float(np.mean(hits > 0)),
        "scans": scan_summary,
        "total": {name: value + len(sizes) for name, value in scan_summary.items()},
    }
    if return_details:
        result["details"] = {"ranks": ranks, "scans": scans, "hits": hits}
    return result


def evaluate(order, truth, owner, sizes, *, oracle=False, probes=None, return_details=False):
    """Measure recall@1 and recall@10 at the first 95% prefix and at optional locked probes.

    Columns of truth give exact neighbors in increasing distance order. At
    each recall cutoff, the oracle orders lists by the number of those
    neighbors stored in each list, then by ascending list size and ID. This
    maximizes recall at each fixed probe count. Flat routing work is charged
    as one comparison per list for both arms. With return_details enabled,
    each operating point includes NumPy arrays in query order.
    """
    homes, sizes = _truth_lists(truth, owner, sizes)
    if homes.shape[1] != 10:
        raise ValueError("evaluation requires exact top-10 IDs")
    if oracle:
        if order is not None:
            raise ValueError("pass order=None for oracle evaluation")
    else:
        order = np.asarray(order)
        if order.shape != (len(homes), len(sizes)) or order.dtype.kind not in "iu":
            raise ValueError("order must be an integer matrix with one list permutation per query")
        list_ranks = _ranks(order, homes)

    if probes is None:
        probes = {}
    elif isinstance(probes, Integral):
        probes = {1: probes, 10: probes}
    if not isinstance(probes, Mapping) or any(
        k not in (1, 10) or isinstance(p, bool) or not isinstance(p, Integral)
        or not 1 <= p <= len(sizes) for k, p in probes.items()
    ):
        raise ValueError("locked probes must be integers from 1 through the number of lists")

    result = {}
    for k in (1, 10):
        if oracle:
            order = _oracle_order(homes[:, :k], sizes)
            ranks = _ranks(order, homes[:, :k])
        else:
            ranks = list_ranks[:, :k]
        # Use the exact ceiling of 95/100 * count to avoid a rounded target.
        target = (95 * ranks.size + 99) // 100
        at_95 = int(np.partition(ranks.ravel(), target - 1)[target - 1])
        measurement = _measure(order, sizes, ranks, at_95, return_details)
        locked = None
        if k in probes:
            locked = measurement if probes[k] == at_95 else _measure(
                order, sizes, ranks, int(probes[k]), return_details
            )
        result[f"recall@{k}"] = {
            "at_95": measurement,
            "recall_below_95": np.count_nonzero(ranks < at_95) / ranks.size,
            "locked": locked,
            "rank": _summary(ranks),
        }
    return result
