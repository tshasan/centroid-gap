"""Command line: fit a tuning for a saved Faiss IVF index, or show one.

    python -m ivf_tune fit index.faiss queries.npy -o tuning.npz [--truth truth.npy] [--base base.npy [--base-ids ids.npy]]
    python -m ivf_tune show tuning.npz
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from .core import Tuning, fit


def _load(path):
    return None if path is None else np.load(path, mmap_mode="r")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ivf_tune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="learn per-list biases and a probe count")
    f.add_argument("index", type=Path, help="Faiss IndexIVF file")
    f.add_argument("queries", type=Path, help=".npy float32 (n, d): a sample of real queries")
    f.add_argument("-o", "--output", type=Path, required=True)
    f.add_argument("--truth", type=Path, help=".npy int64 (n, k): exact top-k ids (else computed)")
    f.add_argument("--base", type=Path, help=".npy base vectors, for exact truth on compressed indexes; "
                   "row i must be stored id i unless --base-ids is given")
    f.add_argument("--base-ids", type=Path, help=".npy int64: the stored id of each row of --base (for add_with_ids)")
    f.add_argument("--shift", type=Path, help=".npy queries from a different distribution to guard against")
    f.add_argument("--k", type=int, default=10)
    f.add_argument("--target", type=float, default=0.95, help="recall@k target in (0, 1] (default 0.95)")
    f.add_argument("--max-fit", type=int, default=50_000)
    f.add_argument("--threads", type=int, help="Faiss threads")
    s = sub.add_parser("show", help="print a tuning's report")
    s.add_argument("tuning", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "show":
        t = Tuning.load(a.tuning)
        print(json.dumps({"nprobe": t.nprobe, "uses_bias": t.uses_bias, **t.report}, indent=2))
        return 0
    if a.threads:
        faiss.omp_set_num_threads(a.threads)
    try:
        index = faiss.read_index(str(a.index))
        tuning = fit(index, np.load(a.queries), truth=_load(a.truth), base=_load(a.base), base_ids=_load(a.base_ids),
                     k=a.k, target=a.target, shift_queries=_load(a.shift), max_fit=a.max_fit)
    except (ValueError, OSError, RuntimeError) as error:  # bad inputs, unreadable files, Faiss errors
        print(f"ivf_tune: {error}", file=sys.stderr)
        return 1
    print(f"wrote {tuning.save(a.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
