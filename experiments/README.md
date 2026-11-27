# Experiments

Scripts behind Section 7 of the [paper](../paper/main.pdf). Outputs the paper cites are in `../results/`. Run from the repository root with Python 3.12 and [`../requirements-experiments.txt`](../requirements-experiments.txt). Staging downloads public benchmark files (DEEP and Yandex Text-to-Image from big-ann-benchmarks, CC BY 4.0). Everything runs on CPU.

```sh
data=~/.cache/centroid-gap/data    # staged vectors and query splits
run=~/.cache/centroid-gap/run      # indexes and answer keys
gateb=~/.cache/centroid-gap/gateb  # extra queries and the second index seed
export VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
```

## Million-vector routing

Oracle headroom and retrained routers (Table 1).

```sh
python experiments/ivf_workload_data.py "$data"
for w in deep t2i; do
  python experiments/ivf_workload_index.py "$data" "$run" "$w"
  python experiments/benchmark_ivf_workloads.py "$run" "$w" oracle
  python experiments/benchmark_ivf_workloads.py "$run" "$w" fit
done
python experiments/ivf_pq_replay.py "$data" "$run" deep --challengers --extra-probes 13 14 16 17 18
python experiments/ivf_pq_replay.py "$data" "$run" t2i --challengers --extra-probes 25 32
python experiments/ivf_shift_control.py "$run" --replay-queries 2000
for w in deep t2i; do
  python experiments/benchmark_ivf_workloads.py "$run" "$w" export --data "$data" --destination "$run/share"
done
```

Writes `results/deep.json`, `results/t2i.json` and `results/*-ranks.npz`. The PQ replay and shift control are folded into the export.

## Routers on fixed lists

Several representatives per list, one bias per list, and `ivf_tune` (Table 2). These read `$run` and never re-cluster.

```sh
python experiments/list_effective_dimension.py "$data" "$run" deep t2i --output results/list-effective-dimension.json
for w in deep t2i; do
  python experiments/tip_router.py "$data" "$run" "$w" --tips 1 2 4 8 --output "results/tip-router-$w.json"
  python experiments/power_router.py "$data" "$run" "$w" --output "results/power-router-$w.json"
done
python experiments/reanalyse_power_router.py "$run" --output results/power-router-reanalysis.json

for w in deep t2i; do
  python experiments/power_router_data.py "$data" "$run" "$gateb" "$w"
  python experiments/power_router_v2.py "$run" "$gateb" "$w" --seed 2026
done
python experiments/validate_ivf_tune.py "$run" "$gateb" --output results/ivf-tune-validation.json
python experiments/validate_ivf_tune.py "$run" "$gateb" --seed 2027 --data "$data" \
  --output results/ivf-tune-validation-seed2027.json
```

Gate B v2 logs are in `results/gate-b-v2/`.

## Rebuilding the lists with e-means

Needs a Faiss checkout with e-means:

```sh
src=~/.cache/centroid-gap/faiss
git clone https://github.com/facebookresearch/faiss.git "$src"
git -C "$src" checkout 2b648f0605f2ef3d14c4705350f8a0b214463878
python experiments/benchmark_emeans.py "$data" "$run" "$run/emeans" --faiss-checkout "$src"
```

Writes `results/emeans-*.json`.

## Large-margin cosine misses

```sh
python experiments/ivf_margin_misses.py "$data" ~/.cache/centroid-gap/cosine --workload both --threads 4
```

Outputs are in `results/cosine-margin/`.

## Spherical e-means replay

Replays the training theorems of Section 4 on the pinned trainer (single-threaded for byte-identical output):

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/replay_spherical_emeans.py --faiss-checkout "$src" > results/spherical-emeans.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/replay_spherical_emeans.py --faiss-checkout "$src" --random-starts > results/spherical-emeans-random.json
```

## Libraries and replication

- `ivf_pipeline.py`: stage prerequisites and verified downloads.
- `ivf_oracle.py`: the oracle and fixed-probe accounting.
- `fit_ivf_router.py`: router fitting and installation.
- `frozen_index.py`: loads a run directory and scores centroids.
- `routing_stats.py`: recall and work from list ranks, interpolation and the paired bootstrap.
- `emeans_ivf_eval.py`: the evaluation behind `benchmark_emeans.py`.
- `compare_replication.py`: diffs two result JSON files value by value.

`results/replication-x86/` holds an x86-64 rerun of the million-vector and cosine studies ([notes](../docs/verification.md#x86-64-faiss-replication)).
