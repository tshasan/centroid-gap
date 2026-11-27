# ivf-tune

Make an existing Faiss IVF index reach the same recall with less work. No rebuild.

An IVF index picks which lists to scan by scoring each list's centroid. `ivf-tune` learns one number per list (a bias added to that score) from a sample of your real queries, and picks the probe count that reaches your recall target. The biases take 4 bytes per list and add no vector math at query time.

```python
import faiss, ivf_tune

index = faiss.read_index("my_index.faiss")          # any IndexIVF*, inner product or L2
tuning = ivf_tune.fit(index, sample_queries)         # a few thousand to ~100k real queries
tuning.save("tuning.npz")

D, I = ivf_tune.search(index, tuning, queries, k=10) # same (D, I) as index.search
```

The biases live in `tuning.npz`, not in the index. They take effect only when you search through `ivf_tune.search` from Python; a plain `index.search` call, or any C++, GPU or database code that calls Faiss directly, still routes by the plain centroids. See [Limits](#limits).

From the command line:

```sh
python -m ivf_tune fit my_index.faiss sample_queries.npy -o tuning.npz
python -m ivf_tune show tuning.npz
```

## Results on two million-vector benchmarks

1,000-list IVF-Flat indexes on 1M vectors, CPU Faiss 1.12, 6 threads, target recall@10 ≥ 0.95. The tuning was fitted on labelled sample queries. The numbers below are on test queries not used for fitting; see the second limit below for a caveat. "Centroids" uses the probe count `ivf-tune` picked for plain centroid routing.

| Workload | | nprobe | recall@10 | distance evaluations / query | batch throughput, ms / query (median of 7 rounds) |
|---|---|---:|---:|---:|---:|
| DEEP (image descriptors, L2) | centroids | 18 | 0.9521 | 19,924 | 0.108 |
| | **tuned** | 16 | 0.9516 | **18,581 (−6.7%)** | **0.102 (−6.1%)** |
| Text-to-Image (inner product) | centroids | 34 | 0.9498 | 34,512 | 0.404 |
| | **tuned** | 31 | 0.9507 | **31,989 (−7.3%)** | **0.380 (−5.8%)** |
| Text-to-Image, image-query shift | centroids | 34 | 0.9949 | 36,639 | 0.437 |
| | **tuned** | 31 | 0.9939 | **33,610 (−8.3%)** | **0.406 (−7.0%)** |

Those rows are read at whole probe counts, so a one-probe step (about 3% of the work) moves them. Interpolated to exactly 95% recall@10, with paired bootstrap 95% intervals, the work change is −6.0% [−7.7, −4.3] on DEEP and −8.7% [−10.9, −6.5] on Text-to-Image. The shift rows use the probe counts tuned on text queries, where image queries reach about 99.4% recall, so their −8.3% is not a same-recall comparison. Interpolated to 95% recall on the image queries, the change is −1.2% [−3.5, +0.9], which includes zero: no regression, and no clear gain. The ms column is batch throughput (whole test split per call, 6 threads), not single-query latency. The two arms were timed alternately over 7 rounds; the per-round time change ranged from −7.2% to −2.9% (DEEP), −6.3% to −4.8% (Text-to-Image) and −7.9% to −6.2% (shift).

A second index seed (2027, with the same fitting method) gave smaller or similar numbers at whole probe counts: DEEP 18 to 17 probes, work −2.2%, time −2.2% (rounds −2.8% to −1.1%); Text-to-Image 34 to 32 probes, work −6.6%, time −4.2% (rounds −4.8% to −3.9%); image shift −7.0% work. Interpolated to 95% recall: DEEP −6.6% [−8.3, −5.0], Text-to-Image −11.1% [−13.6, −8.6], image shift −2.8% [−5.8, −0.4]. The whole-probe DEEP result is small because the tuned arm stopped at 17 probes and overshot the target (0.9565), which is why the interpolated figure is the fairer comparison. Results: [`results/ivf-tune-validation-seed2027.json`](../../results/ivf-tune-validation-seed2027.json).

Fitting took about 2 minutes per index on 50,000 queries. Reproduce with [`experiments/validate_ivf_tune.py`](../../experiments/validate_ivf_tune.py); results are in [`results/ivf-tune-validation.json`](../../results/ivf-tune-validation.json).

## What it does

1. **Labels.** It finds the exact top-k of each sample query: exhaustively for IVF-Flat, or from `base=` vectors or `truth=` ids for compressed indexes. With `base=`, row i must be stored id i; if you added vectors with `add_with_ids`, also pass `base_ids=` (the id of each row). A full-probe spot check raises an error if the rows and ids don't line up.
2. **Fits biases.** It fits one bias per list with a size-aware softmax loss. Scanning a list costs its size, so the loss charges each list its size. A small grid of temperatures is fitted and the best one is chosen on held-out queries.
3. **Picks nprobe.** It chooses the smallest probe count whose one-sided 95% lower confidence bound on real recall@k clears the target. The baseline gets the same treatment, so you see a fair comparison. A probe count tuned on plain averages tends to land just below target on new queries; the margin is there for that.
4. **Refuses when the biases don't help.** If they save less than 1% on held-out queries, raise work on `shift_queries` from a different distribution, or fail to cut real search work once the probe count is rounded to a whole number, the tuning keeps plain centroids (all-zero bias) and still gives you the safe probe count.
5. **Fails loudly.** A target outside (0, 1], or one that even probing every list cannot reach (common with heavily compressed codes), raises `ValueError` instead of returning a full-scan probe count.

`tuning.report` records every candidate, the decision and the reason, and recall and work on a held-out check split.

## Limits

- **Expect single digits.** On the two benchmarks above the saving was 6–11% depending on index seed and how it is read. The Lean proofs say nothing about how much biases can save on your data; their factor-of-2 comparison holds only on one constructed family.
- **Not a clean test of the defaults.** The default temperature grid was chosen from Gate B v2 runs whose logs printed test results, so the numbers above are not a held-out test of that choice.
- **Your queries must look like your traffic.** The biases learn where queries go. If traffic changes, refit, and pass a sample of the new kind of query as `shift_queries` to guard against regressions.
- **Compressed indexes (IVF-PQ and others).** Fitting optimises list coverage, which is an upper bound on recall for compressed codes. The probe count is still chosen by real search recall on held-out sample queries, with a margin for sampling noise. The margin does not cover a shift between your sample and live traffic. So far it has been measured only on IVF-Flat.
- **Python only, batched queries.** Routing runs in Python around Faiss's `search_preassigned`. Each call costs a fixed tens of microseconds (49 µs against 8 µs native for one query on a 128-list index), so the saving shows up only when queries are batched. `SearchParameters` (id selectors, `max_codes`) are not passed through.
- **Flat quantizers.** Routing scores every centroid. With an HNSW or other non-flat quantizer this can cost more than the quantizer's own search; `ivf_tune` warns when it sees one.
- **Refit after rebuilding.** A tuning stores a fingerprint of the centroids and is refused on any other index, including the same object after retraining.
- **Threading.** `search` briefly sets `index.parallel_mode` during a call and restores it after. Don't change that setting from another thread at the same time.
- Only plain `faiss.IndexIVF` objects with inner-product or L2 metric are supported. Unwrap `IndexPreTransform` and apply the transform yourself.

## Tests

From this directory (about 10 s on a laptop, synthetic indexes only):

```sh
python -m unittest discover -s tests
```
