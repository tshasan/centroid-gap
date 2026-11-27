# TODO

Rule: run the cheapest test that could kill an idea before spending compute on it, and check the literature before proving anything new.

## Release

- [ ] Post the paper on arXiv (cs.DS; cross-list cs.IR, cs.LO); post the formalization on the Lean Zulip for review.

## Experiments

- [ ] **Clean held-out test of `ivf_tune`.** Fresh split; fix the temperature grid before looking at test. Pass: at least 5% less work on both datasets, no regression on image queries. Required before proposing a per-list bias upstream (Faiss, pgvector).
- [ ] **Spilling.** Store each vector in a second list (second-nearest centroid, or SOAR-style) and measure work at 95% recall@10, counting duplicate scans. Does it recover more of the oracle gap than routers on fixed lists?
- [ ] Ten million vectors: does the oracle gap widen with N? Run only after a method worth scaling exists.

## Proofs

- [ ] Close the planar gap's factor of two: is the worst ratio for optimal k-means $1+\sqrt N/4$ or $1+\sqrt N/2$?
- [ ] `RobustPrune`: the sharp threshold between cosine lead and α at which beam search must find the winner.
- [ ] One-vector floor with per-query tie-breaking (the Lean bound assumes a query-independent order).
- [ ] One-vector floor in $\mathbb R^2$ with polynomially bounded coefficients; is the fixed budget $\lceil (k+1)/2\rceil$ attainable there?
- [ ] Lean: convexity of the softmax bias fit; "load is free, accuracy is not" (proved on paper only; the notes are `docs/bias-router-theory.md` in git history, commit 23d5526).

## Upstream to Mathlib

Ask on the Lean Zulip before each one.

- [ ] Positive spanning sets: definition, and "positively spans $\mathbb R^d$ ⇒ at least $d+1$ vectors" (Davis 1954).
- [ ] Convexity of log-sum-exp, strict modulo constant shifts.
- [ ] Potential lemma for weighted digraphs (potentials exist iff no negative cycle).

## Not pursuing

More representatives per list (Gates A and KV-2 failed); GPU conic-hull pruning; LAION-scale runs.
