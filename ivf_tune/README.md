# ivf-tune

Fits one bias per list for an existing Faiss IVF index, so it reaches the same recall with less work. No rebuild. The biases take 4 bytes per list and add no vector math at query time.

```sh
pip install -e "ivf_tune[cpu]"   # from the repository root; [cpu] pulls in faiss-cpu
```

```python
import faiss, ivf_tune

index = faiss.read_index("my_index.faiss")           # IndexIVF*, inner product or L2
tuning = ivf_tune.fit(index, sample_queries)         # a few thousand or more real queries
tuning.save("tuning.npz")

tuning = ivf_tune.Tuning.load("tuning.npz")
D, I = ivf_tune.search(index, tuning, queries, k=10) # same (D, I) as index.search
```

Command line: `python -m ivf_tune fit my_index.faiss sample_queries.npy -o tuning.npz`, then `python -m ivf_tune show tuning.npz`.

The biases live in `tuning.npz`, not in the index. They apply only when you search through `ivf_tune.search`; a plain `index.search` still routes by the centroids.

## How it works

1. Finds the exact top-k of each sample query (exhaustively for IVF-Flat; pass `base=` or `truth=` for compressed indexes).
2. Fits the biases with a softmax loss that charges each list its size, choosing the temperature on held-out queries.
3. Picks the smallest probe count whose one-sided 95% lower confidence bound on recall clears the target. The baseline gets the same treatment.
4. Keeps plain centroids (zero bias) if the biases save less than 1% on held-out queries, raise work on `shift_queries`, or save nothing at whole probe counts.

`tuning.report` records every candidate and the decision.

## Results

Work at exactly 95% recall@10 on 1,000-list IVF-Flat indexes over 1M vectors (paper, Table 2), with paired bootstrap 95% intervals:

| Index seed | DEEP | Text-to-Image | Image-query shift |
|---|---|---|---|
| 2026 | −6.0% [−7.6, −4.3] | −8.7% [−10.9, −6.5] | −1.2% [−3.5, +0.9] |
| 2027 | −6.6% [−8.3, −5.0] | −11.1% [−13.6, −8.6] | −2.8% [−5.8, −0.4] |

Batch time per query fell by about 6%. The default temperature grid was chosen from runs whose logs printed test results, so these are not a clean held-out test of that choice. The validation script and raw results are at tag [`experiments-v1`](https://github.com/tshasan/centroid-gap/tree/experiments-v1).

## Limits

- Expect single-digit savings.
- Sample queries must look like live traffic. Refit when traffic or the index changes; a tuning is refused on an index with different centroids.
- Measured only on IVF-Flat. For compressed codes the fit optimizes list coverage, an upper bound on recall.
- Python only. Each call has a fixed overhead of tens of microseconds, so batch your queries. `SearchParameters` are not passed through.
- Flat quantizers only; routing scores every centroid. Plain `faiss.IndexIVF` with inner product or L2; unwrap `IndexPreTransform` yourself.

## Tests

From `ivf_tune/` (about 10 s, synthetic indexes):

```sh
python -m unittest discover -s tests
```
