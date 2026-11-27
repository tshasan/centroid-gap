"""Unit tests on small synthetic IVF indexes. Run: python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ivf_tune  # noqa: E402
from ivf_tune.core import Layout, coverage_curve, list_owners, owner_of  # noqa: E402


def clustered(n, d, clusters, rng, spread=0.35):
    """Clusters of very different sizes, so list sizes vary and the size-aware loss has work to do."""
    weights = rng.pareto(1.2, clusters) + 0.2
    centers = rng.normal(size=(clusters, d))
    which = rng.choice(clusters, size=n, p=weights / weights.sum())
    return (centers[which] + spread * rng.normal(size=(n, d))).astype(np.float32)


def build(metric, n=20_000, d=24, nlist=64, seed=0):
    rng = np.random.default_rng(seed)
    base = clustered(n, d, 40, rng)
    queries = clustered(6000, d, 40, np.random.default_rng(seed))  # same cluster centers
    queries += 0.15 * rng.normal(size=queries.shape).astype(np.float32)
    if metric == "ip":
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        coarse, mt = faiss.IndexFlatIP(d), faiss.METRIC_INNER_PRODUCT
    else:
        coarse, mt = faiss.IndexFlatL2(d), faiss.METRIC_L2
    index = faiss.IndexIVFFlat(coarse, d, nlist, mt)
    index.cp.seed = seed
    index.train(base)
    index.add(base)
    return index, base, np.ascontiguousarray(queries)


def skewed(seed=0, n=20_000, d=16, nlist=48):
    """Index whose list sizes vary a lot and queries land evenly over clusters, so biases do help."""
    rng = np.random.default_rng(seed)
    centers = 4 * rng.normal(size=(nlist, d))
    weights = rng.pareto(0.8, nlist) + 0.05
    which = rng.choice(nlist, size=n, p=weights / weights.sum())
    base = (centers[which] + rng.normal(size=(n, d))).astype(np.float32)
    queries = (centers[rng.integers(0, nlist, 8000)] + rng.normal(size=(8000, d))).astype(np.float32)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
    index.cp.seed = seed
    index.train(base)
    index.add(base)
    return index, base, queries


class IvfTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(2)
        cls.cases = {m: build(m) for m in ("l2", "ip")}

    def test_zero_bias_matches_faiss_search(self):
        for metric, (index, _, q) in self.cases.items():
            layout = Layout(index)
            t = ivf_tune.Tuning(np.zeros(index.nlist, np.float32), 7, metric, layout.fingerprint)
            D, I = ivf_tune.search(index, t, q[:500], 10)
            index.nprobe = 7
            D0, I0 = index.search(q[:500], 10)
            self.assertEqual((I == I0).mean(), 1.0, metric)
            np.testing.assert_allclose(D, D0, rtol=1e-4, atol=1e-4)

    def test_exact_neighbors_match_brute_force(self):
        for metric, (index, base, q) in self.cases.items():
            a = ivf_tune.exact_neighbors(index, q[:300], 10)
            b = ivf_tune.exact_neighbors(index, q[:300], 10, base=base)
            self.assertGreater((a == b).mean(), 0.999, metric)

    def test_coverage_equals_real_recall_for_flat(self):
        index, _, q = self.cases["l2"]
        q = q[:800]
        truth = ivf_tune.exact_neighbors(index, q, 10)
        layout = Layout(index)
        labels = owner_of(truth, *list_owners(index))
        recall, work = coverage_curve(layout.scores(q), labels, layout.sizes)
        index.nprobe = 5
        _, I = index.search(q, 10)
        real = np.mean([len(set(a) & set(t)) / 10 for a, t in zip(I.tolist(), truth.tolist())])
        self.assertAlmostEqual(recall[4], real, places=3)
        self.assertTrue(np.all(np.diff(work) >= 0))

    def test_fit_reaches_target_and_does_not_lose(self):
        for metric, (index, _, q) in self.cases.items():
            t = ivf_tune.fit(index, q, target=0.9, steps=150, log=lambda *_: None)
            r = t.report
            print(f"\n  {metric}: {r['decision']}, work change {100 * r['work_change']:+.1f}%, "
                  f"nprobe {r['centroids']['nprobe']} -> {r['tuned']['nprobe']}", file=sys.stderr)
            self.assertIn(r["decision"], ("biases", "centroids"))
            self.assertGreaterEqual(r["tuned"]["check_recall"], 0.88, metric)
            if r["decision"] == "biases":
                best = min(a["select_work_interp"] for a in r["arms"])
                self.assertTrue(t.uses_bias)
                self.assertLess(best, 0.99 * r["centroids_select_work_interp"])
            else:
                self.assertFalse(t.uses_bias)

    def test_fallback_when_bias_cannot_help(self):
        index, _, q = self.cases["ip"]
        t = ivf_tune.fit(index, q, target=0.9, arms=[dict(tau_frac=0.2, gamma=0.0, lam=1e9)],
                         steps=20, log=lambda *_: None)
        self.assertEqual(t.report["decision"], "centroids")
        self.assertFalse(t.uses_bias)
        self.assertEqual(t.nprobe, t.report["centroids"]["nprobe"])

    def test_save_load_and_index_mismatch(self):
        index, _, q = self.cases["l2"]
        t = ivf_tune.fit(index, q[:3000], target=0.9, arms=[dict(tau_frac=0.2, gamma=1.0, lam=1e-4)],
                         steps=50, log=lambda *_: None)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.npz"
            t.save(path)
            u = ivf_tune.Tuning.load(path)
        np.testing.assert_array_equal(t.bias, u.bias)
        self.assertEqual(t.nprobe, u.nprobe)
        self.assertEqual(t.report, u.report)
        other, _, _ = build("l2", seed=1)
        with self.assertRaises(ValueError):
            ivf_tune.search(other, u, q[:10], 10)

    def test_nonzero_bias_search_is_exact_over_routed_lists(self):
        index, base, q = self.cases["l2"]
        layout = Layout(index)
        bias = np.random.default_rng(3).normal(scale=5.0, size=index.nlist).astype(np.float32)
        t = ivf_tune.Tuning(bias, 6, "l2", layout.fingerprint)
        q = q[:200]
        D, I = ivf_tune.search(index, t, q, 10)
        S = layout.scores(q) + bias
        lists = np.argsort(-S, axis=1, kind="stable")[:, :6]
        ids, owners = list_owners(index)
        _, I0 = index.search(q, 10)
        self.assertLess((I == I0).mean(), 1.0)  # the bias really changed routing
        for qi in range(len(q)):
            member = np.isin(owners, lists[qi])
            cand = ids[member]
            d = ((base[cand] - q[qi]) ** 2).sum(axis=1)
            want = cand[np.argsort(d, kind="stable")[:10]]
            self.assertEqual(set(I[qi].tolist()), set(want.tolist()))

    def test_fit_with_bias_cuts_real_work(self):
        index, _, q = skewed()
        t = ivf_tune.fit(index, q, target=0.9, steps=200, log=lambda *_: None)
        r = t.report
        if r["decision"] == "biases":
            self.assertTrue(t.uses_bias)
            self.assertLess(r["tuned"]["select_work"], r["centroids"]["select_work"])
        else:
            self.assertFalse(t.uses_bias)
            self.assertEqual(r["tuned"], r["centroids"])

    def test_nprobe_above_nlist_is_clamped(self):
        index, _, q = self.cases["l2"]
        t = ivf_tune.Tuning(np.zeros(index.nlist, np.float32), 1, "l2", Layout(index).fingerprint)
        _, I = ivf_tune.search(index, t, q[:200], 10, nprobe=index.nlist + 9)
        truth = ivf_tune.exact_neighbors(index, q[:200], 10)
        self.assertEqual((np.sort(I, 1) == np.sort(truth, 1)).mean(), 1.0)

    def test_unreachable_or_invalid_target_raises(self):
        index, base, q = self.cases["l2"]
        for target in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                ivf_tune.fit(index, q[:3000], target=target, log=lambda *_: None)
        pq = faiss.IndexIVFPQ(faiss.IndexFlatL2(base.shape[1]), base.shape[1], 32, 2, 4)
        pq.train(base)
        pq.add(base)
        with self.assertRaises(ValueError):
            ivf_tune.fit(pq, q[:3000], base=base, target=0.99, arms=[dict(tau_frac=0.2, gamma=1.0, lam=1e-4)],
                         steps=20, log=lambda *_: None)

    def test_custom_ids_need_base_ids(self):
        index, base, q = self.cases["l2"]
        perm = np.random.default_rng(5).permutation(len(base)).astype(np.int64) * 3 + 7
        custom = faiss.IndexIVFFlat(faiss.IndexFlatL2(base.shape[1]), base.shape[1], 32)
        custom.train(base)
        custom.add_with_ids(base, perm)
        with self.assertRaises(ValueError):
            ivf_tune.exact_neighbors(custom, q[:50], 10, base=base)
        shuffled = faiss.IndexIVFFlat(faiss.IndexFlatL2(base.shape[1]), base.shape[1], 32)
        shuffled.train(base)
        shuffled.add_with_ids(base, np.random.default_rng(6).permutation(len(base)).astype(np.int64))
        with self.assertRaises(ValueError):  # same id set as 0..n-1, wrong order
            ivf_tune.exact_neighbors(shuffled, q[:50], 10, base=base)
        got = ivf_tune.exact_neighbors(custom, q[:50], 10, base=base, base_ids=perm)
        custom.nprobe = custom.nlist
        _, want = custom.search(q[:50], 10)
        self.assertGreater((np.sort(got, 1) == np.sort(want, 1)).mean(), 0.999)

    def test_layout_cache_follows_retrain_and_add(self):
        index, base, q = build("l2", n=8000, nlist=32, seed=4)
        old = ivf_tune.Tuning(np.zeros(32, np.float32), 4, "l2", Layout(index).fingerprint)
        ivf_tune.search(index, old, q[:10], 10)          # warm the cache
        index.reset()
        index.is_trained = False
        index.quantizer.reset()
        index.cp.seed = 99
        index.train(base[::-1].copy())
        index.add(base)
        with self.assertRaises(ValueError):               # stale tuning is refused
            ivf_tune.search(index, old, q[:10], 10)
        new = ivf_tune.Tuning(np.zeros(32, np.float32), 4, "l2", Layout(index).fingerprint)
        _, I = ivf_tune.search(index, new, q[:300], 10)
        index.nprobe = 4
        _, I0 = index.search(q[:300], 10)
        self.assertEqual((I == I0).mean(), 1.0)
        index.add(base[:1000])                            # grown in place: sizes must refresh
        from ivf_tune.core import _layout_for
        self.assertEqual(_layout_for(index).sizes.sum(), index.ntotal)

    def test_l2_far_from_origin_matches_faiss(self):
        index, base, q = build("l2", n=8000, nlist=32, seed=6)
        shifted_base, shifted_q = base + 1000.0, q[:500] + 1000.0
        far = faiss.IndexIVFFlat(faiss.IndexFlatL2(base.shape[1]), base.shape[1], 32)
        far.train(shifted_base)
        far.add(shifted_base)
        t = ivf_tune.Tuning(np.zeros(32, np.float32), 4, "l2", Layout(far).fingerprint)
        _, I = ivf_tune.search(far, t, shifted_q, 10)
        lists = Layout(far).route(np.zeros(32, np.float32), shifted_q, 4)[0]
        ref = (((shifted_q.astype(np.float64)[:, None, :] - far.quantizer.reconstruct_n(0, 32)[None]) ** 2)
               .sum(-1).argsort(axis=1)[:, :4])
        self.assertGreater((np.sort(lists, 1) == np.sort(ref, 1)).mean(), 0.99)

    def test_cli_writes_npz_and_reports_path(self):
        import contextlib, io
        from ivf_tune.__main__ import main
        index, _, q = self.cases["l2"]
        with tempfile.TemporaryDirectory() as d:
            faiss.write_index(index, f"{d}/i.faiss")
            np.save(f"{d}/q.npy", q[:3000])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["fit", f"{d}/i.faiss", f"{d}/q.npy", "-o", f"{d}/tuning", "--target", "0.9", "--max-fit", "1500"])
            self.assertEqual(code, 0)
            self.assertIn(f"{d}/tuning.npz", out.getvalue().splitlines()[-1])
            self.assertTrue(Path(f"{d}/tuning.npz").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["fit", f"{d}/i.faiss", f"{d}/q.npy", "-o", f"{d}/x", "--target", "1.5"]), 1)

    def test_bad_query_shapes_and_values_raise(self):
        index, _, q = self.cases["l2"]
        t = ivf_tune.Tuning(np.zeros(index.nlist, np.float32), 4, "l2", Layout(index).fingerprint)
        for bad in (q[:10, :-1], q[0], np.full((10, q.shape[1]), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                ivf_tune.search(index, t, bad, 10)
        with self.assertRaises(ValueError):
            ivf_tune.search(index, t, q[:10], 10, nprobe=0)
        with self.assertRaises(ValueError):
            ivf_tune.fit(index, q[:3000, :-1], target=0.9, log=lambda *_: None)
        with self.assertRaises(ValueError):
            ivf_tune.fit(index, q[:3000], target=0.9, arms=[], log=lambda *_: None)

    def test_cli_reports_missing_file(self):
        import contextlib, io
        from ivf_tune.__main__ import main
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["fit", f"{d}/missing.faiss", f"{d}/q.npy", "-o", f"{d}/t"]), 1)
        self.assertIn("ivf_tune:", err.getvalue())

    def test_rejects_non_ivf(self):
        flat = faiss.IndexFlatL2(8)
        with self.assertRaises(ValueError):
            ivf_tune.fit(flat, np.zeros((2000, 8), np.float32))


if __name__ == "__main__":
    unittest.main()
