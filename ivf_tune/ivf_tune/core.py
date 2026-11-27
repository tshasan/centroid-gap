"""Fit one scalar bias per IVF list so an existing Faiss index reaches a recall target with less work.

Routing score for list c (higher is better), with b_c the learned bias:
    inner product:  <q, c> + b_c
    L2:             2<q, c> - |c|^2 + b_c      (= |q|^2 - |q - c|^2 + b_c, a power diagram)
The biases add no vector work at query time. They are fitted on sample queries with the
cost-aware loss from the Gate B v2 experiments: for a query with label weights T_c
(share of its exact top-k held by list c) and logits z_c = score_c / tau,
    L = -sum_c T_c z_c + log sum_c w_c exp(z_c + b_c / tau),   w_c = (size_c / mean size)^gamma,
plus an L2 penalty on b. gamma = 1 charges each list its size, which matches how work is counted.

Work per query = nlist centroid scores + the sizes of the probed lists (distance evaluations).
"""

import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

import faiss
import numpy as np

CAND = 96             # softmax runs over the top CAND lists by centroid score plus the label lists
CHUNK = 8192           # queries per search batch
DENSE_CELLS = 1 << 24  # max query x list cells held at once while fitting (bounds memory for large nlist)
DEFAULT_ARMS = tuple(dict(tau_frac=t, gamma=g, lam=1e-4) for t in (0.1, 0.2, 0.3, 0.5) for g in (0.5, 1.0))
Z_ONE_SIDED_95 = 1.645


# ---------------------------------------------------------------- index layout

def _ivf(index):
    if not isinstance(index, faiss.IndexIVF):
        raise ValueError("ivf_tune needs a faiss.IndexIVF (IndexIVFFlat, IndexIVFPQ, ...); "
                         "unwrap IndexPreTransform / IndexIDMap and transform queries yourself")
    if index.metric_type not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
        raise ValueError("only inner-product and L2 IVF indexes are supported")
    if not index.is_trained:
        raise ValueError("index is not trained")
    return index


def _queries(index, x, name="queries"):
    """float32 C-contiguous (n, d) copy of `x`, or ValueError if it cannot be queries for `index`."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != index.d:
        raise ValueError(f"{name} must have shape (n, {index.d}); got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contain NaN or infinite values")
    return x


def _rows(nlist):
    """Queries per chunk so that a dense (chunk, nlist) block stays under DENSE_CELLS cells."""
    return max(1, DENSE_CELLS // max(1, nlist))


class Layout:
    """Centroids, list sizes and metric of an IVF index, read once.

    For L2 the routing score is computed on vectors centred at the centroid mean mu:
        2<x - mu, c - mu> - |c - mu|^2  =  |x - mu|^2 - |x - c|^2,
    which ranks lists exactly like 2<x, c> - |c|^2 (they differ by a per-query constant) but does not
    lose float32 precision when the data sit far from the origin."""

    def __init__(self, index):
        ivf = _ivf(index)
        self.nlist = ivf.nlist
        self.ntotal = int(ivf.ntotal)
        self.metric = "ip" if ivf.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
        raw = ivf.quantizer.reconstruct_n(0, ivf.nlist).astype(np.float64)
        self.fingerprint = [int(self.nlist), self.metric, float(raw.sum())]
        self.probe = _probe_rows(ivf)
        self.mu = raw.mean(axis=0) if self.metric == "l2" else np.zeros(raw.shape[1])
        centred = raw - self.mu
        self.centers = centred.astype(np.float32)
        self.c2 = (centred ** 2).sum(axis=1).astype(np.float32)
        self.mu32 = self.mu.astype(np.float32)
        self.sizes = np.array([ivf.invlists.list_size(c) for c in range(ivf.nlist)], dtype=np.int64)
        self.flat_quantizer = isinstance(faiss.downcast_index(ivf.quantizer), faiss.IndexFlat)
        self._router_cache = None

    def matches(self, index):
        """Cheap check that `index` still has the centroids and contents this layout was read from."""
        return (index.nlist == self.nlist and int(index.ntotal) == self.ntotal
                and np.array_equal(_probe_rows(index), self.probe))

    def _centre(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x if self.metric == "ip" else x - self.mu32

    def scores(self, x):
        """Routing score of every list (higher is better), up to a per-query constant for L2."""
        s = self._centre(x) @ self.centers.T
        return s if self.metric == "ip" else 2 * s - self.c2[None, :]

    def router(self, bias):
        """Faiss flat index whose inner product with [x - mu, 1] is the biased routing score.

        inner product: [c, b]·[x, 1] = <x, c> + b;  L2: [2c', b - |c'|^2]·[x', 1] = 2<x', c'> - |c'|^2 + b
        with ' meaning centred. So list selection runs in Faiss's own C++ top-k search."""
        key = bias.tobytes()
        cached = self._router_cache  # read once: another thread may replace it
        if cached is None or cached[0] != key:
            c = self.centers if self.metric == "ip" else 2 * self.centers
            extra = bias if self.metric == "ip" else bias - self.c2
            r = faiss.IndexFlatIP(c.shape[1] + 1)
            r.add(np.ascontiguousarray(np.hstack([c, extra[:, None]]), dtype=np.float32))
            cached = self._router_cache = (key, r)
        return cached[1]

    def route(self, bias, x, nprobe):
        """Top-nprobe lists by biased score, best first, with the distances Faiss's quantizer would report."""
        xc = self._centre(x)
        xa = np.hstack([xc, np.ones((len(x), 1), dtype=np.float32)])
        score, lists = self.router(bias).search(xa, int(nprobe))
        plain = score - bias[lists]
        if self.metric == "ip":
            return lists, plain
        return lists, (np.einsum("nd,nd->n", xc, xc)[:, None] - plain).astype(np.float32)


def _probe_rows(index):
    """A few centroids spread over the list range: changes whenever the index is retrained."""
    rows = np.unique(np.linspace(0, index.nlist - 1, num=min(8, index.nlist)).astype(np.int64))
    return np.stack([index.quantizer.reconstruct(int(r)) for r in rows]).astype(np.float32)


def list_owners(index):
    """Sorted stored ids and the list holding each."""
    ivf = _ivf(index)
    ids, owners = [], []
    for c in range(ivf.nlist):
        n = ivf.invlists.list_size(c)
        if n == 0:
            continue
        pointer = ivf.invlists.get_ids(c)
        try:
            ids.append(faiss.rev_swig_ptr(pointer, n).copy())
        finally:
            ivf.invlists.release_ids(c, pointer)
        owners.append(np.full(n, c, dtype=np.int32))
    ids, owners = np.concatenate(ids), np.concatenate(owners)
    order = np.argsort(ids, kind="stable")
    ids, owners = ids[order], owners[order]
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError("index stores some id more than once")
    return ids, owners


def owner_of(truth, ids, owners):
    pos = np.searchsorted(ids, truth)
    pos = np.minimum(pos, len(ids) - 1)
    if np.any(ids[pos] != truth):
        raise ValueError("ground-truth ids missing from the index (use the index's own ids, no -1)")
    return owners[pos].astype(np.int32)


def exact_neighbors(index, x, k, base=None, base_ids=None):
    """Exact top-k ids: brute force on `base` if given, else every list of an IVF-Flat index.

    Brute force returns row numbers of `base`. They are mapped to stored ids through `base_ids`
    (the ids passed to add_with_ids, row by row). Without base_ids the index must store ids
    0..ntotal-1, i.e. row i of `base` is id i."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if base is not None:
        if len(base) != index.ntotal:
            raise ValueError(f"base has {len(base):,} rows but the index stores {index.ntotal:,} vectors")
        if base_ids is None:
            ids, _ = list_owners(index)
            if not np.array_equal(ids, np.arange(index.ntotal)):
                raise ValueError("the index uses custom ids (add_with_ids): pass base_ids, the id of each "
                                 "row of base, or pass exact truth ids instead")
        flat = (faiss.IndexFlatIP if index.metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2)(index.d)
        flat.add(np.ascontiguousarray(base, dtype=np.float32))
        rows = flat.search(x, k)[1]
        truth = rows if base_ids is None else np.asarray(base_ids, dtype=np.int64)[rows]
        _check_alignment(index, x, truth, k)
        return truth
    if not isinstance(index, faiss.IndexIVFFlat):
        raise ValueError("compressed index: pass the base vectors (base=) or exact truth ids (truth=)")
    params = faiss.SearchParametersIVF(nprobe=index.nlist)
    return index.search(x, k, params=params)[1]


def _check_alignment(index, x, truth, k, sample=64, floor=0.1):
    """Catch base rows that do not line up with stored ids (e.g. add_with_ids with a permutation).

    A full-probe search of the index finds the true neighbors (IVF-Flat) or a good share of them
    (compressed codes). If it shares almost none of them with the brute-force ids, the ids are wrong."""
    m = min(sample, len(x))
    if m == 0:
        return
    got = index.search(x[:m], k, params=faiss.SearchParametersIVF(nprobe=index.nlist))[1]
    overlap = np.mean([len(set(a) & set(t)) / k for a, t in zip(got.tolist(), truth[:m].tolist())])
    if overlap < floor:
        raise ValueError(f"exact neighbors from base= share only {100 * overlap:.1f}% of ids with a full search "
                         "of the index: row i of base is probably not stored id i. Pass base_ids (the id of "
                         "each row), or exact truth ids")


# ---------------------------------------------------------------- fitting

def _candidates(layout, x, labels):
    """Per query: candidate lists (top CAND by score plus label lists), raw scores, label weights."""
    n, k = labels.shape
    cand = min(CAND, layout.nlist)
    width = cand + k
    C = np.empty((n, width), dtype=np.int32)
    raw = np.empty((n, width), dtype=np.float32)
    T = np.empty((n, width), dtype=np.float32)
    spread = None
    for s in range(0, n, _rows(layout.nlist)):
        e = min(n, s + _rows(layout.nlist))
        S = layout.scores(x[s:e])
        if spread is None:
            top = -np.sort(-S, axis=1)[:, : min(32, layout.nlist)]
            spread = float(np.median(top[:, 0] - top[:, -1]))
        if cand < layout.nlist:
            top = np.argpartition(-S, cand - 1, axis=1)[:, :cand]
        else:
            top = np.broadcast_to(np.arange(layout.nlist), (e - s, layout.nlist))
        cc = np.sort(np.concatenate([top, labels[s:e]], axis=1), axis=1)
        first = np.ones(cc.shape, dtype=bool)
        first[:, 1:] = cc[:, 1:] != cc[:, :-1]
        C[s:e] = cc
        raw[s:e] = np.where(first, np.take_along_axis(S, cc, axis=1), -np.inf)
        T[s:e] = (cc[:, :, None] == labels[s:e, None, :]).sum(-1) / k * first
    if not spread or spread <= 0:
        spread = 1.0
    return C, raw, T, spread


def fit_bias(cands, sizes, spread, tau_frac, gamma, lam, steps=400):
    """Full-batch Adam on the weighted-softmax loss; returns biases with mean zero."""
    C, raw, T = cands
    nlist = len(sizes)
    tau = tau_frac * spread
    logw = (gamma * np.log(np.maximum(sizes, 1) / max(sizes.mean(), 1))).astype(np.float32)
    qw = 1.0 / len(C)
    b = np.zeros(nlist)
    m = np.zeros(nlist)
    v = np.zeros(nlist)
    lr, l2 = 0.05 * tau, lam / tau
    flat = C.ravel()
    for t in range(1, steps + 1):
        z = (raw + b[C].astype(np.float32)) / tau + logw[C]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = np.bincount(flat, weights=((p - T) * qw).ravel(), minlength=nlist) / tau + l2 * b
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        b = b - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
    return (b - b.mean()).astype(np.float32)


# ---------------------------------------------------------------- evaluation

def coverage_curve(S, labels, sizes):
    """For each probe count p (index p-1): share of true neighbors whose list is probed, and mean work.

    For IVF-Flat this share is exactly recall@k; for compressed indexes it is an upper bound."""
    n, nlist = S.shape
    hits = np.zeros(nlist, dtype=np.int64)
    scanned = np.zeros(nlist, dtype=np.float64)
    step = _rows(nlist)
    for s in range(0, n, step):
        e = min(n, s + step)
        ranked = np.argsort(-S[s:e], axis=1, kind="stable")
        pos = np.empty_like(ranked)
        pos[np.arange(e - s)[:, None], ranked] = np.arange(nlist)[None, :]
        rank = np.take_along_axis(pos, labels[s:e].astype(np.int64), axis=1)
        hits += np.bincount(rank.ravel(), minlength=nlist)
        scanned += sizes[ranked].cumsum(axis=1).sum(axis=0)
    recall = hits.cumsum() / labels.size
    work = nlist + scanned / n
    return recall, work


def interp_work(recall, work, target, nlist):
    """Work linearly interpolated to exactly `target` coverage (no one-probe steps).

    Coverage reaches 1 when every list is probed, so any target in (0, 1] is reachable."""
    r = np.concatenate([[0.0], recall])
    w = np.concatenate([[float(nlist)], work])
    p = int(np.searchsorted(r, target))
    if p >= len(r):
        raise ValueError(f"coverage never reaches {target:g} (best {r[-1]:.4f}); are the labels right?")
    f = (target - r[p - 1]) / max(r[p] - r[p - 1], 1e-12)
    return float(w[p - 1] + f * (w[p] - w[p - 1]))


def _search(index, layout, bias, x, k, nprobe):
    nprobe = max(1, min(int(nprobe), layout.nlist))  # like Faiss, never probe more lists than exist
    x = np.ascontiguousarray(x, dtype=np.float32)
    D = np.empty((len(x), k), dtype=np.float32)
    I = np.empty((len(x), k), dtype=np.int64)
    work = np.empty(len(x))
    # The Python wrapper of search_preassigned insists on index.nprobe; call the C++ entry point
    # with per-call parameters instead, so the caller's index is never modified.
    params = faiss.SearchParametersIVF(nprobe=int(nprobe))
    # Called directly with parallel_mode 0, search_preassigned schedules its threads worse than
    # index.search does (about 40% slower on 1M IVF-Flat with 6 threads); mode 3 matches index.search.
    # Switch only for this call and restore the caller's setting (and its high flag bits).
    saved = index.parallel_mode
    if saved & 0xFF == 0:
        index.parallel_mode = (saved & ~0xFF) | 3
    try:
        for s in range(0, len(x), CHUNK):
            e = min(len(x), s + CHUNK)
            xs = np.ascontiguousarray(x[s:e])
            lists, dq = layout.route(bias, xs, nprobe)
            lists, dq = np.ascontiguousarray(lists), np.ascontiguousarray(dq, dtype=np.float32)
            Ds, Is = D[s:e], I[s:e]
            index.search_preassigned_c(e - s, faiss.swig_ptr(xs), k, faiss.swig_ptr(lists), faiss.swig_ptr(dq),
                                       faiss.swig_ptr(Ds), faiss.swig_ptr(Is), False, params)
            work[s:e] = layout.nlist + layout.sizes[lists].sum(axis=1)
    finally:
        index.parallel_mode = saved
    return D, I, work


def per_query_recall(I, truth):
    k = truth.shape[1]
    return np.array([len(set(a[:k]) & set(t)) / k for a, t in zip(I.tolist(), truth.tolist())])


def choose_nprobe(index, layout, bias, x, truth, target, start=1):
    """Smallest nprobe >= start whose one-sided 95% lower bound on real search recall@k reaches target.

    Recall grows with nprobe (up to ties), so: exponential search for a probe count that clears the
    target, then binary search between the last miss and that count."""
    k = truth.shape[1]
    nlist = layout.nlist

    def recall_at(p):
        return per_query_recall(_search(index, layout, bias, x, k, p)[1], truth)

    def clears(p):
        r = recall_at(p)
        return r.mean() - Z_ONE_SIDED_95 * r.std(ddof=1) / np.sqrt(len(r)) >= target

    start = max(1, min(int(start), nlist))
    if clears(start):
        return start
    miss, hit = start, None          # miss: known to fall short; hit: known to clear
    p = start
    while p < nlist:
        p = min(nlist, 2 * p)
        if clears(p):
            hit = p
            break
        miss = p
    if hit is None:
        raise ValueError(f"recall@{k} target {target:g} is out of reach: probing all {nlist} lists gives "
                         f"{recall_at(nlist).mean():.4f} (lower bound below target). Compressed codes cap "
                         f"recall; lower the target, re-rank with exact vectors, or check the truth ids")
    while miss + 1 < hit:
        mid = (miss + hit) // 2
        if clears(mid):
            hit = mid
        else:
            miss = mid
    return hit


# ---------------------------------------------------------------- public API

class Routing(str, Enum):
    """What a tuning routes by."""
    BIASES = "biases"         # centroid score plus the learned per-list bias
    CENTROIDS = "centroids"   # plain centroid score (all-zero bias)


@dataclass(frozen=True)
class Decision:
    routing: Routing
    reason: str


@dataclass
class Tuning:
    bias: np.ndarray
    nprobe: int
    metric: str
    fingerprint: list
    report: dict = field(default_factory=dict)

    def save(self, path):
        """Write the tuning as .npz and return the path written (np.savez adds .npz if missing)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(path, bias=self.bias, nprobe=self.nprobe, metric=self.metric,
                 fingerprint=json.dumps(self.fingerprint), report=json.dumps(self.report))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["bias"].astype(np.float32), int(z["nprobe"]), str(z["metric"]),
                       json.loads(str(z["fingerprint"])), json.loads(str(z["report"])))

    @property
    def uses_bias(self):
        return bool(np.any(self.bias != 0))


class _LayoutCache:
    """Small LRU of Layouts keyed by id(index), so repeated search calls skip re-reading centroids.

    id() can be reused by a new index after the old one is freed, and an index can be retrained or
    grown in place, so a cached layout is returned only if its nlist, ntotal and sampled centroids
    still match the index."""

    def __init__(self, size=4):
        self._size = size
        self._layouts = OrderedDict()

    def get(self, index):
        key = id(index)
        layout = self._layouts.get(key)
        if layout is None or not layout.matches(_ivf(index)):
            layout = Layout(index)
        self._layouts[key] = layout
        self._layouts.move_to_end(key)
        while len(self._layouts) > self._size:
            self._layouts.popitem(last=False)
        return layout


_LAYOUTS = _LayoutCache()


def _layout_for(index, tuning=None):
    """Layout of `index`; with `tuning`, also check that the tuning was fitted on this index."""
    layout = _LAYOUTS.get(index)
    if tuning is not None:
        n, metric, total = tuning.fingerprint
        same = (n == layout.nlist and metric == layout.metric and len(tuning.bias) == layout.nlist
                and np.isclose(total, layout.fingerprint[2], rtol=1e-5, atol=1e-3))
        if not same:
            raise ValueError("this tuning was fitted on a different index (centroids or nlist differ)")
    return layout


def search(index, tuning, x, k, nprobe=None):
    """Search `index` with the tuned routing. Returns (D, I) like index.search.

    nprobe defaults to the tuned probe count; values above nlist are clamped, as in Faiss."""
    layout = _layout_for(index, tuning)
    x = _queries(index, x)
    if int(k) < 1:
        raise ValueError(f"k must be at least 1; got {k}")
    nprobe = tuning.nprobe if nprobe is None else int(nprobe)
    if nprobe < 1:
        raise ValueError(f"nprobe must be at least 1; got {nprobe}")
    if not layout.flat_quantizer:
        warnings.warn("ivf_tune routes by scanning every centroid; with a non-flat quantizer (e.g. HNSW) "
                      "this can be slower than index.search", RuntimeWarning, stacklevel=2)
    D, I, _ = _search(index, layout, tuning.bias, x, int(k), nprobe)
    return D, I


def _split_rows(n, holdout, max_fit, seed):
    """Disjoint random (select, check, fit) row indices: select picks the arm, check reports."""
    n_hold = min(holdout, n // 3)
    if n_hold < 500:
        raise ValueError(f"need at least 1,500 queries; got {n:,}")
    order = np.random.default_rng(seed).permutation(n)
    return order[: n_hold // 2], order[n_hold // 2 : n_hold], order[n_hold:][:max_fit]


def fit(index, queries, truth=None, base=None, k=10, target=0.95, shift_queries=None, shift_truth=None,
        arms=DEFAULT_ARMS, max_fit=50_000, holdout=6000, min_gain=0.01, steps=400, seed=0, log=print,
        base_ids=None):
    """Learn per-list biases and a probe count for `target` recall@k on `index`.

    queries: sample of real queries (a few thousand or more). truth: their exact top-k ids in the
    index; computed if omitted (IVF-Flat, or pass base=, plus base_ids= if the index uses custom ids).
    Queries are split into fit / select / check.

    Returns a Tuning. It keeps the plain centroids (zero bias), and still sets nprobe, if any of
    three checks fails: the best bias must cut interpolated work by at least min_gain on the select
    queries; it must not raise interpolated work on shift_queries; and at the chosen whole probe
    counts it must cut real search work on the select queries. Raises ValueError on bad inputs or
    if the recall target cannot be reached at all.
    """
    _ivf(index)
    if not 0 < target <= 1:
        raise ValueError(f"target is a recall between 0 and 1; got {target}")
    if not 1 <= k <= index.ntotal:
        raise ValueError(f"k must be between 1 and the number of stored vectors; got {k}")
    if not arms:
        raise ValueError("arms must list at least one (tau_frac, gamma, lam) setting")
    layout = _layout_for(index)
    if not layout.flat_quantizer:
        log("note: the index has a non-flat quantizer; ivf_tune.search scans every centroid, "
            "which can cost more than the quantizer's own search")
    x = _queries(index, queries)
    if truth is None:
        log(f"computing exact top-{k} for {len(x):,} queries")
        truth = exact_neighbors(index, x, k, base, base_ids)
    truth = np.asarray(truth, dtype=np.int64)[:, :k]
    if truth.shape != (len(x), k):
        raise ValueError(f"truth must have shape ({len(x)}, {k}); got {truth.shape}")
    ids, owners = list_owners(index)
    labels = owner_of(truth, ids, owners)

    sel, chk, fit_rows = _split_rows(len(x), holdout, max_fit, seed)
    log(f"{len(x):,} queries: {len(fit_rows):,} fit, {len(sel):,} select, {len(chk):,} check; "
        f"nlist {layout.nlist}, metric {layout.metric}")

    S_sel = layout.scores(x[sel])

    def select_work(b):
        return interp_work(*coverage_curve(S_sel + b, labels[sel], layout.sizes), target, layout.nlist)

    zero = np.zeros(layout.nlist, dtype=np.float32)
    base_cost = select_work(zero)

    *cands, spread = _candidates(layout, x[fit_rows], labels[fit_rows])
    rows = []
    for arm in arms:
        b = fit_bias(cands, layout.sizes, spread, steps=steps, **arm)
        cost = select_work(b)
        rows.append({**arm, "select_work_interp": cost, "bias": b})
        log(f"  tau {arm['tau_frac']:<4} gamma {arm['gamma']:<3} lam {arm['lam']:g}: "
            f"select work {cost:,.0f} ({100 * (cost / base_cost - 1):+.2f}% vs centroids {base_cost:,.0f})")
    best = min(rows, key=lambda r: r["select_work_interp"])

    # Check 1: enough interpolated gain on the select queries.
    if best["select_work_interp"] <= base_cost * (1 - min_gain):
        decision = Decision(Routing.BIASES, "lower work on held-out queries")
    else:
        decision = Decision(Routing.CENTROIDS, f"best bias saved less than {100 * min_gain:g}% on held-out queries")

    # Check 2: no regression on queries from a different distribution.
    shift_report = None
    if shift_queries is not None and decision.routing is Routing.BIASES:
        xs = _queries(index, shift_queries, "shift_queries")
        ts = exact_neighbors(index, xs, k, base, base_ids) if shift_truth is None else np.asarray(shift_truth)[:, :k]
        ls = owner_of(ts, ids, owners)
        Ss = layout.scores(xs)
        sb = interp_work(*coverage_curve(Ss, ls, layout.sizes), target, layout.nlist)
        st = interp_work(*coverage_curve(Ss + best["bias"], ls, layout.sizes), target, layout.nlist)
        shift_report = {"queries": len(xs), "centroids_work_interp": sb, "biases_work_interp": st}
        log(f"  shift queries: {st:,.0f} vs centroids {sb:,.0f}")
        if st > sb:
            decision = Decision(Routing.CENTROIDS, "biases raised work on the shift queries")

    def summary(b):
        start = int(np.searchsorted(coverage_curve(S_sel + b, labels[sel], layout.sizes)[0], target)) + 1
        p = choose_nprobe(index, layout, b, x[sel], truth[sel], target, start=start)
        sel_work = _search(index, layout, b, x[sel], k, p)[2]
        _, I, work = _search(index, layout, b, x[chk], k, p)
        return {"nprobe": int(p), "select_work": float(sel_work.mean()),
                "check_recall": float(per_query_recall(I, truth[chk]).mean()),
                "check_work": float(work.mean())}

    # Check 3: interpolated coverage can promise a saving that a whole probe count does not deliver.
    centroids = summary(zero)
    tuned = centroids
    if decision.routing is Routing.BIASES:
        tuned = summary(best["bias"])
        if tuned["select_work"] >= centroids["select_work"]:
            decision = Decision(Routing.CENTROIDS,
                                "at whole probe counts the biases did not cut real search work on held-out queries")
            tuned = centroids

    uses_bias = decision.routing is Routing.BIASES
    bias = best["bias"] if uses_bias else zero
    report = {
        "metric": layout.metric, "nlist": layout.nlist, "k": k, "target": target,
        "queries": {"fit": len(fit_rows), "select": len(sel), "check": len(chk)},
        "centroids_select_work_interp": base_cost,
        "arms": [{kk: vv for kk, vv in r.items() if kk != "bias"} for r in rows],
        "chosen_arm": {kk: best[kk] for kk in ("tau_frac", "gamma", "lam")} if uses_bias else None,
        "decision": decision.routing.value, "reason": decision.reason, "shift": shift_report,
        "centroids": centroids, "tuned": tuned,
        "work_change": tuned["check_work"] / centroids["check_work"] - 1,
    }
    log(f"decision: {decision.routing.value} ({decision.reason})")
    log(f"centroids: nprobe {centroids['nprobe']}, check recall@{k} {centroids['check_recall']:.4f}, "
        f"work {centroids['check_work']:,.0f}")
    log(f"tuned:     nprobe {tuned['nprobe']}, check recall@{k} {tuned['check_recall']:.4f}, "
        f"work {tuned['check_work']:,.0f} ({100 * report['work_change']:+.2f}%)")
    return Tuning(bias.astype(np.float32), tuned["nprobe"], layout.metric, layout.fingerprint, report)
