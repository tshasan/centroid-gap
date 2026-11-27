"""Prepare a standard Faiss IVF-Flat snapshot and exact query labels for the roadmap benchmark."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

if __package__:
    from .fit_ivf_router import make_router as router
    from .frozen_index import metric_for
    from .ivf_pipeline import Stage, require_stage
else:
    from fit_ivf_router import make_router as router
    from frozen_index import metric_for
    from ivf_pipeline import Stage, require_stage


N_LISTS = 1000
TARGET = 0.95
SEED = 2026


def stored_lists(index):
    owner = np.full(index.ntotal, -1, dtype=np.int32)
    sizes = np.empty(index.nlist, dtype=np.int64)
    for label in range(index.nlist):
        size = sizes[label] = index.invlists.list_size(label)
        pointer = index.invlists.get_ids(label)
        try:
            ids = faiss.rev_swig_ptr(pointer, int(size))
            owner[ids] = label
        finally:
            index.invlists.release_ids(label, pointer)
    if np.any(owner < 0) or np.any(sizes == 0) or sizes.sum() != index.ntotal:
        raise ValueError("expected one listing per base ID and 1000 nonempty lists")
    return owner, sizes


def prepare(data, out, workload):
    require_stage(data, workload, Stage.DATA)
    directory = out / workload
    directory.mkdir(exist_ok=True, parents=True)
    base = np.load(data / f"{workload}-base.npy", mmap_mode="r")
    metric = metric_for(workload)
    dim = base.shape[1]
    coarse = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(coarse, dim, N_LISTS, coarse.metric_type)
    index.cp.niter = 20
    index.cp.nredo = 2
    index.cp.seed = SEED
    index.train(base)
    index.add(base)
    owner, sizes = stored_lists(index)
    np.save(directory / "owner.npy", owner)
    np.save(directory / "sizes.npy", sizes)
    np.save(directory / "centers.npy", coarse.reconstruct_n(0, N_LISTS))
    faiss.write_index(index, str(directory / "flat.faiss"))
    print(f"{workload}: built {index.ntotal:,} × {index.nlist} IVF", flush=True)
    del index, coarse
    exact = router(base, metric)
    retained = {}
    for split in (["val", "test", "fit", "image"] if workload == "t2i" else ["val", "test", "fit"]):
        queries = np.load(data / f"{workload}-{split}.npy", mmap_mode="r")
        scores, ids = exact.search(queries, 11)
        # Top-1 and the top-ten set must be separated at their two boundaries.
        top_gap = np.abs(scores[:, 0] - scores[:, 1])
        ten_gap = np.abs(scores[:, 9] - scores[:, 10])
        keep = (top_gap > 1e-5) & (ten_gap > 1e-5)
        np.savez(directory / f"{split}.npz", queries=queries[keep], truth=ids[keep, :10],
                 source_rows=np.flatnonzero(keep))
        retained[split] = {"retained": int(keep.sum()), "input": len(queries),
                           "top1_margin_exclusions": int((top_gap <= 1e-5).sum()),
                           "top10_margin_exclusions": int((ten_gap <= 1e-5).sum())}
        print(f"{workload} {split}: {keep.sum():,}/{len(queries):,} exact labels", flush=True)
    result = {"metric": metric, "N": len(base), "nlist": N_LISTS, "faiss": faiss.__version__,
              "list_size_min_max": [int(sizes.min()), int(sizes.max())], "splits": retained}
    (directory / "index.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("workload", choices=["deep", "t2i"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)
    prepare(args.data, args.output, args.workload)
