import IndexRecall.ProbeRanks

/-!
# Exact-scan IVF versus globally optimal k-means

The original, swapped and rerouted indexes use `k` routing distances and
one more distance per candidate. At every query in the interval the first
index needs all rows, and each of the other two needs exactly one.
-/

noncomputable section

namespace IndexRecall.GapFamily

open OptimalBuckets

abbrev Address (k : ℕ) := Fin k × Fin (core k + 1)

/-- Global optimality against arbitrary real centers and assignments, with
uniqueness of the posting lists up to relabeling. -/
theorem optimal_partition {k : ℕ} (hk : 2 ≤ k)
    (f : Fin k → Fin (core k + 1) → Fin k) (c : Fin k → ℝ) :
    rowLoss (k := k) site (mean site) ≤ assignmentLoss site f c ∧
      (assignmentLoss site f c ≤ rowLoss (k := k) site (mean site) →
        ∃ e : Equiv.Perm (Fin k), ∀ i t, f i t = e i) :=
  strict_optimal site anchor anchor_separated (core_close hk) (anchor_loss_lt hk)
    (nearest_anchor_row hk) f c

abbrev Returns {k : ℕ} (x : Fin k → Fin (core k + 1) → ℝ) (r : Fin k → ℝ)
    (q : ℝ) := FlatScan.Returns (fun a : Address k => (q - x a.1 a.2)^2) (fun i => (q - r i)^2)

abbrev distanceWork {k : ℕ} (r : Fin k → ℝ) (q : ℝ) :=
  FlatScan.work (Fin (core k + 1)) (fun i => (q - r i)^2)

abbrev originalNN {k : ℕ} (hk : 2 ≤ k) : Address k := (firstRow hk, lastSlot k)
abbrev swappedNN {k : ℕ} (hk : 2 ≤ k) : Address k := (secondRow hk, lastSlot k)

/-- The exact success thresholds hold pointwise, including for adaptive choices of `p`. -/
theorem probe_thresholds {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) (p : ℕ) :
    (Returns site (mean site) q p (originalNN hk) ↔ k ≤ p) ∧
    (Returns (swapped hk) (mean (swapped hk)) q p (swappedNN hk) ↔ 1 ≤ p) ∧
    (Returns site (rerouted hk) q p (originalNN hk) ↔ 1 ≤ p) := by
  have ho (b : Address k) (hb : b ≠ originalNN hk) :
      (q - site (originalNN hk).1 (originalNN hk).2)^2 < (q - site b.1 b.2)^2 := by
    simpa using uniqueNearest hk hq b.1 b.2 hb
  have hs (b : Address k) (hb : b ≠ swappedNN hk) :
      (q - swapped hk (swappedNN hk).1 (swappedNN hk).2)^2 <
        (q - swapped hk b.1 b.2)^2 := by
    simpa using swapped_uniqueNearest hk hq b.1 b.2 hb
  exact ⟨FlatScan.returns_iff _ p _ ho (baseline_rank hk hq),
    FlatScan.returns_iff _ p _ hs (swapped_rank hk hq),
    FlatScan.returns_iff _ p _ ho (rerouted_rank hk hq)⟩

/-- On any nonempty batch from the interval, a fixed probe budget gives recall zero or one. -/
theorem batch_recall {k : ℕ} (hk : 2 ≤ k) (qs : List ℝ) (hne : qs ≠ [])
    (hq : ∀ q ∈ qs, QueryInRange k q) (p : ℕ) :
    FlatScan.recall (fun q => Returns site (mean site) q p (originalNN hk)) qs =
        (if k ≤ p then 1 else 0) ∧
    FlatScan.recall (fun q => Returns (swapped hk) (mean (swapped hk)) q p (swappedNN hk)) qs =
        (if 1 ≤ p then 1 else 0) ∧
    FlatScan.recall (fun q => Returns site (rerouted hk) q p (originalNN hk)) qs =
        (if 1 ≤ p then 1 else 0) := by
  exact ⟨FlatScan.recall_of_constant hne _ (fun q hm => (probe_thresholds hk (hq q hm) p).1),
    FlatScan.recall_of_constant hne _ (fun q hm => (probe_thresholds hk (hq q hm) p).2.1),
    FlatScan.recall_of_constant hne _ (fun q hm => (probe_thresholds hk (hq q hm) p).2.2)⟩

/-- Routing distances plus the cardinality of the actual candidate set at each
index's smallest successful probe count. -/
theorem distance_counts {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    distanceWork (k := k) (mean site) q k = k + k * width k ∧
    distanceWork (mean (swapped hk)) q 1 = k + width k ∧
    distanceWork (rerouted hk) q 1 = k + width k := by
  have cost (r : Fin k → ℝ) (a : Fin k) (p : ℕ)
      (hi : Function.Injective (fun i => (q - r i)^2)) (hr : probeRank r q a = p) :
      distanceWork r q p = k + p * width k := by
    subst p
    simpa only [Fintype.card_fin, width] using FlatScan.work_at_rank (Fin (core k + 1)) hi a
  exact ⟨cost _ _ _ (baseline_untied hk hq) (baseline_rank hk hq),
    by simpa using cost _ _ _ (swapped_untied hk hq) (swapped_rank hk hq),
    by simpa using cost _ _ _ (rerouted_untied hk hq) (rerouted_rank hk hq)⟩

/-- Including the flat router, the cost factor is within 1/1024 of the number of buckets. -/
theorem total_ratio {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    let ratio := (distanceWork (k := k) (mean site) q k : ℝ) / distanceWork (mean (swapped hk)) q 1
    (k : ℝ) - 1 / 1024 < ratio ∧ ratio < k := by
  dsimp only
  obtain ⟨hb, hs, _⟩ := distance_counts hk hq
  rw [hb, hs]
  push_cast
  have hw : 0 < (width k : ℝ) := by exact_mod_cast width_pos k
  have hd : 0 < (k : ℝ) + width k := by positivity
  rw [lt_div_iff₀ hd, div_lt_iff₀ hd]
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hw' : (width k : ℝ) = (32 * (k : ℝ))^2 + 1 := by simp [width, core]
  rw [hw']
  constructor <;> nlinarith [mul_nonneg (show (0 : ℝ) ≤ k by positivity) (sub_nonneg.mpr hk')]

end IndexRecall.GapFamily

