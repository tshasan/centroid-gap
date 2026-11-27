#!/usr/bin/env python3
"""Stage the Yandex Deep1B and Text-to-Image IVF workloads as mmap-ready NPYs.

Run python3 experiments/ivf_workload_data.py /path/to/cache with NumPy installed.
Use --offline when the six pinned *.source.fbin files are already in the cache.
Dataset attribution: https://big-ann-benchmarks.com/neurips21.html (CC BY 4.0).
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

if __package__:
    from .ivf_pipeline import StreamDownloadState
else:
    from ivf_pipeline import StreamDownloadState

SOURCE = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/"
HEADER = struct.Struct("<II")
DEEP_PUBLIC_NAME = "query.public.10K.fbin"
DEEP_PUBLIC_SHA256 = "8438fc763f14e0f9741fda15b3e11215aef089ce17d1ad47d53b52a7c9fda5bb"


def _row_keys(vectors):
    for start in range(0, len(vectors), 4096):
        batch = vectors[start : start + 4096].copy()
        batch[batch == 0] = 0  # Treat positive and negative zero as equal.
        yield from map(bytes, batch)


BASE_ROWS = 1_000_000
IMAGE_PREFIX = 4_000
IMAGE_ROWS = 2_000
FIT_ROWS = 16_384
VAL_ROWS = 3_000
TEST_ROWS = 5_000
PUBLIC_SEED = 2026
DATASET_PAGE = "https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search"
T2I_SOURCE = urljoin(SOURCE, "../T2I/")


class Slice(NamedTuple):
    name: str
    url: str
    source_rows: int
    rows: int
    dimensions: int
    sha256: str

    @property
    def size(self):
        return HEADER.size + self.rows * self.dimensions * 4

    @property
    def content_range(self):
        if self.rows == self.source_rows:
            return None
        total = HEADER.size + self.source_rows * self.dimensions * 4
        return f"bytes 0-{self.size - 1}/{total}"


SOURCES = {
    "deep": (
        Slice("deep-base", SOURCE + "base.1B.fbin", 10**9, BASE_ROWS, 96,
              "1c5169a4eb8dd860727f47ca641e48f06a2253b7a235ecf4cbde7dda3ec893c3"),
        Slice("deep-learn", SOURCE + "learn.350M.fbin", 350_000_000, 40_000, 96,
              "1676ab38a7ac63dd30aaea426a1649e4a56d0dd77fab51b7e9d207099d72e22d"),
        Slice("deep-public", SOURCE + DEEP_PUBLIC_NAME, 10_000, 10_000, 96,
              DEEP_PUBLIC_SHA256),
    ),
    "t2i": (
        Slice("t2i-base", T2I_SOURCE + "base.1B.fbin", 10**9,
              BASE_ROWS + IMAGE_PREFIX, 200,
              "efa1401b6d626430044963f95c5caf1c54954d808a6ce079490b01463110af44"),
        Slice("t2i-learn", T2I_SOURCE + "query.learn.50M.fbin", 50_000_000, 40_000, 200,
              "e7650ed6d4d77671bd0cc69ab9dc8e0c1f868b406f154e8ce862125ea0ca8834"),
        Slice("t2i-public", T2I_SOURCE + "query.public.100K.fbin", 100_000, 100_000, 200,
              "4b9aabc321ca42a674d5854d1f95347deb902c489c7b49b34e38960f60dad784"),
    ),
}


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _valid_source(path, spec):
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    with path.open("rb") as source:
        if source.read(HEADER.size) != HEADER.pack(spec.source_rows, spec.dimensions):
            return False
    return _sha256(path) == spec.sha256


def _source_array(directory, spec, offline):
    path = directory / f"{spec.name}.source.fbin"
    if not _valid_source(path, spec):
        if offline:
            raise ValueError(f"{path}: missing or invalid pinned source")
        request = Request(spec.url, headers={"Accept-Encoding": "identity"})
        if spec.content_range is not None:
            request.add_header("Range", f"bytes=0-{spec.size - 1}")
        with TemporaryDirectory(dir=directory) as temporary:
            partial = Path(temporary) / path.name
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                if (
                    response.status != (206 if spec.content_range is not None else 200)
                    or response.headers.get("Content-Range") != spec.content_range
                    or response.headers.get("Content-Length") != str(spec.size)
                ):
                    raise ValueError(f"{spec.url}: unexpected HTTP status, length or range")
                dl = StreamDownloadState(
                    label=spec.url,
                    expected_bytes=spec.size,
                    expected_sha256=spec.sha256,
                )
                dl.stream_all(response, output)
            if not _valid_source(partial, spec):
                raise ValueError(f"{spec.url}: source header or SHA-256 mismatch")
            partial.replace(path)
    return np.memmap(
        path, dtype="<f4", mode="r", offset=HEADER.size,
        shape=(spec.rows, spec.dimensions),
    )


def _select(keys, order, blocked):
    """Keep the first occurrence in selection order and report excluded source rows."""
    seen = set()
    kept, overlap, duplicate = [], [], []
    for row in order:
        row = int(row)
        key = keys[row]
        if key in blocked:
            overlap.append(row)
        elif key in seen:
            duplicate.append(row)
        else:
            seen.add(key)
            kept.append(row)
    return kept, {"overlap": overlap, "duplicate": duplicate}


def _save(directory, name, vectors):
    path = directory / f"{name}.npy"
    with TemporaryDirectory(dir=directory) as temporary:
        partial = Path(temporary) / path.name
        np.save(partial, vectors, allow_pickle=False)
        partial.replace(path)
    print(path, flush=True)
    return {"shape": list(vectors.shape), "sha256": _sha256(path)}


def prepare_workload(directory, name, offline=False, test_rows=TEST_ROWS):
    """test_rows=None keeps every public query left after validation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base_spec, learn_spec, public_spec = SOURCES[name]
    all_images = _source_array(directory, base_spec, offline)
    base = all_images[:BASE_ROWS]
    files = {"base": _save(directory, f"{name}-base", base)}
    learn = _source_array(directory, learn_spec, offline)
    public = _source_array(directory, public_spec, offline)
    extra_images = all_images[BASE_ROWS:]

    # Keep keys for candidate queries and only matching keys from the million-row base.
    learn_keys = list(_row_keys(learn))
    public_keys = list(_row_keys(public))
    image_keys = list(_row_keys(extra_images))
    candidates = set(learn_keys).union(public_keys, image_keys)
    base_matches = {key for key in _row_keys(base) if key in candidates}
    del candidates

    rng = np.random.Generator(np.random.PCG64(PUBLIC_SEED))
    public_order, public_exclusions = _select(
        public_keys, rng.permutation(len(public)), base_matches
    )
    fit_order, fit_exclusions = _select(
        learn_keys, range(len(learn)), base_matches.union(public_keys)
    )
    image_order, image_exclusions = _select(
        image_keys, range(len(extra_images)), base_matches.union(public_keys, learn_keys)
    )
    if test_rows is None:
        test_rows = len(public_order) - VAL_ROWS
    if test_rows < 1 or len(public_order) < VAL_ROWS + test_rows or len(fit_order) < FIT_ROWS:
        raise ValueError(f"{name}: too few distinct query vectors for the requested splits")
    if name == "t2i" and len(image_order) < IMAGE_ROWS:
        raise ValueError("t2i: too few distinct image control vectors")

    selections = {
        "fit": fit_order[:FIT_ROWS],
        "val": public_order[:VAL_ROWS],
        "test": public_order[VAL_ROWS : VAL_ROWS + test_rows],
    }
    query_sources = {"fit": learn, "val": public, "test": public}
    for split, rows in selections.items():
        files[split] = _save(directory, f"{name}-{split}", query_sources[split][rows])
    if name == "t2i":
        rows = image_order[:IMAGE_ROWS]
        files["image"] = _save(directory, "t2i-image", extra_images[rows])
        selections["image"] = [BASE_ROWS + row for row in rows]
        image_exclusions = {
            reason: [BASE_ROWS + row for row in rows]
            for reason, rows in image_exclusions.items()
        }

    manifest = {
        "dataset_source": DATASET_PAGE,
        "metric": "l2" if name == "deep" else "inner_product",
        "public_permutation_seed": PUBLIC_SEED,
        "public_bit_generator": "PCG64",
        "sources": [
            {**spec._asdict(), "bytes": spec.size, "content_range": spec.content_range}
            for spec in SOURCES[name]
        ],
        "base_source_ids": {"start": 0, "stop_exclusive": BASE_ROWS},
        "selected_source_ids": selections,
        "excluded_source_ids": {
            "fit": fit_exclusions, "public": public_exclusions, "image": image_exclusions
        },
        "outputs": files,
    }
    if test_rows != TEST_ROWS:  # the default manifest stays byte-identical to published runs
        manifest["test_rows"] = test_rows
    path = directory / f"{name}-workload.json"
    with TemporaryDirectory(dir=directory) as temporary:
        partial = Path(temporary) / path.name
        partial.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        partial.replace(path)
    print(path, flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workload", choices=("deep", "t2i", "both"), default="both")
    parser.add_argument("--offline", action="store_true", help="Use only verified cached sources")
    parser.add_argument("--test-rows", default=str(TEST_ROWS), metavar="N|all",
                        help=f"Held-out test queries (default {TEST_ROWS}); 'all' uses every public query not in validation")
    args = parser.parse_args()
    if args.test_rows == "all":
        test_rows = None
    elif args.test_rows.isdigit() and int(args.test_rows) > 0:
        test_rows = int(args.test_rows)
    else:
        parser.error("--test-rows must be a positive integer or 'all'")
    try:
        for name in SOURCES if args.workload == "both" else (args.workload,):
            prepare_workload(args.directory, name, args.offline, test_rows)
    except (OSError, ValueError) as error:
        parser.exit(1, f"IVF dataset preparation failed: {error}\n")


if __name__ == "__main__":
    main()
