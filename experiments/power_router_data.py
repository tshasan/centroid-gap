"""Stage extra labeled queries and a second index seed for Gate B v2.

Writes only to OUT (a new directory); the original DATA and RUN files are read only.

  1. Extra fit queries. Range-downloads learn rows [40,000, 140,000) of each
     public learn set and appends them to the cached rows [fit end, 40,000), so
     the pool is every learn row after the original 16,384 fit queries.
  2. Image fit/val queries (t2i only). Range-downloads Text-to-Image base rows
     [1,004,000, 1,044,000), past the 1,004,000-row prefix that holds the
     million-row base and the 1,993 image-shift test queries.
  3. Rows equal to a base, public, earlier fit or image-test row, or repeated in
     the pool, are dropped. Exact top-11 by flat search; queries whose top-1 or
     top-10 boundary is within 1e-5 are dropped, as in ivf_workload_index.py.
  4. Second index (seed 2027): the same Faiss IVF-Flat recipe (1,000 lists,
     niter 20, nredo 2) on the same million vectors; owner/sizes/centers only.
     Answer keys do not depend on the index and are reused.

    python experiments/power_router_data.py ~/.cache/centroid-gap/data \
        ~/.cache/centroid-gap/run ~/.cache/centroid-gap/gateb t2i
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import faiss
import numpy as np

if __package__:
    from .ivf_workload_data import SOURCES, BASE_ROWS, IMAGE_PREFIX
    from .ivf_workload_index import stored_lists
else:
    from ivf_workload_data import SOURCES, BASE_ROWS, IMAGE_PREFIX
    from ivf_workload_index import stored_lists

LEARN_STOP = 140_000
IMAGE_STOP = BASE_ROWS + IMAGE_PREFIX + 40_000
SEED2 = 2027


def fetch_rows(url, start, stop, dim, path):
    """Download rows [start, stop) of a .fbin by HTTP range into a raw float32 file."""
    lo, hi = 8 + start * dim * 4, 8 + stop * dim * 4 - 1
    if not (path.is_file() and path.stat().st_size == hi - lo + 1):
        req = Request(url, headers={"Accept-Encoding": "identity", "Range": f"bytes={lo}-{hi}"})
        part = path.with_suffix(".part")
        with urlopen(req, timeout=120) as r, part.open("wb") as f:
            if r.status != 206:
                raise ValueError(f"{url}: expected 206, got {r.status}")
            while chunk := r.read(1 << 22):
                f.write(chunk)
        if part.stat().st_size != hi - lo + 1:
            raise ValueError(f"{url}: short read")
        part.replace(path)
    sha = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    return np.fromfile(path, dtype="<f4").reshape(stop - start, dim), sha


def keys(x):
    x = np.array(x, dtype=np.float32)
    x[x == 0] = 0
    return [hash(bytes(r)) for r in x]


def fresh(x, blocked):
    kept = []
    for i, k in enumerate(keys(x)):
        if k not in blocked:
            blocked.add(k)
            kept.append(i)
    return np.asarray(kept)


def label(flat, q):
    scores, ids = flat.search(np.ascontiguousarray(q, dtype=np.float32), 11)
    keep = (np.abs(scores[:, 0] - scores[:, 1]) > 1e-5) & (np.abs(scores[:, 9] - scores[:, 10]) > 1e-5)
    return keep, ids[:, :10]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("workload", choices=["deep", "t2i"])
    p.add_argument("--threads", type=int, default=10)
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    w = a.workload
    out = a.out / w
    out.mkdir(parents=True, exist_ok=True)
    base_spec, learn_spec, public_spec = SOURCES[w]
    dim = base_spec.dimensions
    manifest = json.loads((a.data / f"{w}-workload.json").read_text())
    fit_end = max(manifest["selected_source_ids"]["fit"]) + 1
    src = lambda name, rows: np.memmap(a.data / f"{name}.source.fbin", dtype="<f4", mode="r",
                                       offset=8, shape=(rows, dim))
    learn_local = src(learn_spec.name, learn_spec.rows)
    public = src(public_spec.name, public_spec.rows)
    base = np.load(a.data / f"{w}-base.npy")
    record = {"workload": w, "learn_url": learn_spec.url, "fit_end_row": fit_end}

    learn_new, sha = fetch_rows(learn_spec.url, learn_spec.rows, LEARN_STOP, dim, out / "learn-rows.f32")
    record["learn_download"] = {"rows": [learn_spec.rows, LEARN_STOP], "sha256": sha}
    pool = np.concatenate([np.asarray(learn_local[fit_end:]), learn_new])
    pool_rows = np.arange(fit_end, LEARN_STOP)
    blocked = set(keys(base)) | set(keys(public)) | set(keys(learn_local[:fit_end]))
    if w == "t2i":
        blocked |= set(keys(src(base_spec.name, base_spec.rows)[BASE_ROWS:]))
    keep = fresh(pool, blocked)
    pool, pool_rows = pool[keep], pool_rows[keep]
    print(f"{w}: {len(pool):,} distinct new learn rows", flush=True)

    flat = faiss.IndexFlatIP(dim) if w == "t2i" else faiss.IndexFlatL2(dim)
    flat.add(base)
    ok, truth = label(flat, pool)
    np.savez(out / "extra.npz", queries=pool[ok], truth=truth[ok], source_rows=pool_rows[ok])
    record["extra"] = {"candidates": len(pool), "retained": int(ok.sum())}
    print(f"{w} extra: {ok.sum():,}/{len(pool):,} exact labels", flush=True)

    if w == "t2i":
        img, sha = fetch_rows(base_spec.url, BASE_ROWS + IMAGE_PREFIX, IMAGE_STOP, dim, out / "image-rows.f32")
        record["image_download"] = {"rows": [BASE_ROWS + IMAGE_PREFIX, IMAGE_STOP], "sha256": sha}
        rows = np.arange(BASE_ROWS + IMAGE_PREFIX, IMAGE_STOP)
        keep = fresh(img, blocked)
        img, rows = img[keep], rows[keep]
        ok, truth = label(flat, img)
        np.savez(out / "image-extra.npz", queries=img[ok], truth=truth[ok], source_rows=rows[ok])
        record["image_extra"] = {"candidates": len(img), "retained": int(ok.sum())}
        print(f"{w} image extra: {ok.sum():,}/{len(img):,} exact labels", flush=True)
    del flat

    metric = faiss.METRIC_INNER_PRODUCT if w == "t2i" else faiss.METRIC_L2
    coarse = faiss.IndexFlatIP(dim) if w == "t2i" else faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(coarse, dim, 1000, metric)
    index.cp.niter, index.cp.nredo, index.cp.seed = 20, 2, SEED2
    index.train(base)
    index.add(base)
    owner, sizes = stored_lists(index)
    d2 = out / f"seed{SEED2}"
    d2.mkdir(exist_ok=True)
    np.save(d2 / "owner.npy", owner)
    np.save(d2 / "sizes.npy", sizes)
    np.save(d2 / "centers.npy", coarse.reconstruct_n(0, 1000))
    record["seed2"] = {"seed": SEED2, "list_size_min_max": [int(sizes.min()), int(sizes.max())]}
    print(f"{w} seed {SEED2}: lists {sizes.min()}..{sizes.max()}", flush=True)
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
