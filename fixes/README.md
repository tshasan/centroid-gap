# Where the headroom is

Scripts behind the "Where the headroom is" paragraph and Table 3 of the paper. They read the cached indexes and query splits written by the `experiments-v1` pipeline (`~/.cache/centroid-gap`), keep the k-means lists and centroids fixed, and count work as in the paper (1,000 routing scores plus every scanned stored copy).

| Script | Measures | Output |
|---|---|---|
| `spill_eval.py` | Spilling to the second list, margin spilling, SOAR, query-guided copies | `results.json` |
| `mapped_assign.py` | Placing each vector where a ridge-predicted query would look | `mapped.json` |
| `budget_eval.py` | Per-query budget oracle and a score-gap stopping rule | `budget.json` |
| `stop_eval.py` | A list-radius stopping rule; per-query recall spread | `stop.json` |

```sh
python fixes/spill_eval.py deep t2i
python fixes/mapped_assign.py deep t2i
python fixes/budget_eval.py deep t2i
python fixes/stop_eval.py deep t2i
```
