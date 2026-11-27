import IndexRecall.PlanarOptimality
import IndexRecall.PlanarRanks

/-! # A sharp-order distance gap with multiple distinct answers -/

noncomputable section

namespace IndexRecall.PlanarGap

abbrev Returns {s : ℕ} (x : Layout s) (r : Label s → Point) (i : Fin s) :=
  FlatScan.Returns (fun a : Label s × Slot s => sqDist (query i) (x a.1 a.2))
    (fun j => sqDist (query i) (r j))

abbrev distanceWork {s : ℕ} (r : Label s → Point) (i : Fin s) :=
  FlatScan.work (Slot s) (fun j => sqDist (query i) (r j))

theorem database_size (s : ℕ) : Fintype.card (Label s × Slot s) = (size s)^2 := by
  rw [Fintype.card_prod, label_card, Fintype.card_fin, sq]

theorem distinct_answers {s : ℕ} (hs : 2 ≤ s) : Function.Injective (@tip s) := by
  intro i j he
  have h : ((i, false), firstSlot hs) = ((j, false), firstSlot hs) :=
    distinct_sites hs (by simpa using he)
  exact congrArg (fun a : Label s × Slot s => a.1.1) h

/-- Each query has its own unique true tip, with the same three probe thresholds. -/
theorem probe_thresholds {s : ℕ} (hs : 2 ≤ s) (i : Fin s) (p : ℕ) :
    (Returns site (mean site) i p ((i, false), firstSlot hs) ↔ s + 1 ≤ p) ∧
    (Returns swapped (mean swapped) i p ((i, true), firstSlot hs) ↔ 1 ≤ p) ∧
    (Returns site rerouted i p ((i, false), firstSlot hs) ↔ 1 ≤ p) := by
  have ho (b : Label s × Slot s) (hb : b ≠ ((i, false), firstSlot hs)) :
      sqDist (query i) (site (i, false) (firstSlot hs)) < sqDist (query i) (site b.1 b.2) := by
    simpa using uniqueNearest hs i b.1 b.2 hb
  have hw (b : Label s × Slot s) (hb : b ≠ ((i, true), firstSlot hs)) :
      sqDist (query i) (swapped (i, true) (firstSlot hs)) < sqDist (query i) (swapped b.1 b.2) := by
    simpa using swapped_uniqueNearest hs i b.1 b.2 hb
  exact ⟨FlatScan.returns_iff _ p _ ho (baseline_rank hs i),
    FlatScan.returns_iff _ p _ hw (swapped_rank hs i),
    FlatScan.returns_iff _ p _ ho (rerouted_rank hs i)⟩

theorem batch_recall {s : ℕ} (hs : 2 ≤ s) (qs : List (Fin s)) (hne : qs ≠ []) (p : ℕ) :
    FlatScan.recall (fun i => Returns site (mean site) i p ((i, false), firstSlot hs)) qs =
        (if s + 1 ≤ p then 1 else 0) ∧
    FlatScan.recall (fun i => Returns swapped (mean swapped) i p ((i, true), firstSlot hs)) qs =
        (if 1 ≤ p then 1 else 0) ∧
    FlatScan.recall (fun i => Returns site rerouted i p ((i, false), firstSlot hs)) qs =
        (if 1 ≤ p then 1 else 0) := by
  exact ⟨FlatScan.recall_of_constant hne _ (fun i _ => (probe_thresholds hs i p).1),
    FlatScan.recall_of_constant hne _ (fun i _ => (probe_thresholds hs i p).2.1),
    FlatScan.recall_of_constant hne _ (fun i _ => (probe_thresholds hs i p).2.2)⟩

theorem distance_counts {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    distanceWork (mean site) i (s+1) = size s + (s+1) * size s ∧
    distanceWork (mean swapped) i 1 = size s + size s ∧
    distanceWork rerouted i 1 = size s + size s := by
  have cost (r : Label s → Point) (a : Label s) (p : ℕ)
      (hi : Function.Injective (fun j => sqDist (query i) (r j))) (hr : probeRank r (query i) a = p) :
      distanceWork r i p = size s + p * size s := by
    subst p
    simpa only [label_card, Fintype.card_fin] using FlatScan.work_at_rank (Slot s) hi a
  exact ⟨cost _ _ _ (baseline_untied hs i) (baseline_rank hs i),
    by simpa using cost _ _ _ (swapped_untied hs i) (swapped_rank hs i),
    by simpa using cost _ _ _ (rerouted_untied hs i) (rerouted_rank hs i)⟩

/-- There are `size s` rows of width `size s`, hence this factor is `1 + √N/4`. -/
theorem total_ratio {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    (distanceWork (mean site) i (s+1) : ℝ) / distanceWork (mean swapped) i 1 =
      1 + (size s : ℝ) / 4 := by
  obtain ⟨hb, hw, _⟩ := distance_counts hs i
  rw [hb, hw]
  push_cast
  have hpos := size_cast_pos hs
  rw [div_eq_iff (by positivity)]
  push_cast [size]
  ring

end IndexRecall.PlanarGap

namespace IndexRecall.FlatScan

/-- Universal ceiling for flat, balanced indexes. `rootN` denotes the nonnegative
square root of `k * B`, and `p,t` are the two probe counts. -/
theorem balanced_work_cap {k B p t : ℕ} (hk : 0 < k) (hB : 0 < B) (hp : p ≤ k) (ht : 1 ≤ t)
    (rootN : ℝ) (hr : 0 ≤ rootN) (hN : rootN^2 = (k : ℝ) * B) :
    ((k : ℝ) + p * (B : ℝ)) / (k + t * (B : ℝ)) ≤ 1 + rootN / 2 := by
  have hp' : (p : ℝ) ≤ k := by exact_mod_cast hp
  have ht' : (1 : ℝ) ≤ t := by exact_mod_cast ht
  have hk' : 0 < (k : ℝ) := by exact_mod_cast hk
  have hB' : 0 < (B : ℝ) := by exact_mod_cast hB
  have hsum : 2 * rootN ≤ (k : ℝ) + B := by
    apply (sq_le_sq₀ (by positivity) (by positivity)).1
    nlinarith [sq_nonneg ((k : ℝ) - B)]
  apply (div_le_iff₀ (by positivity)).2
  calc
    _ ≤ (k : ℝ) + k * (B : ℝ) := add_le_add le_rfl (mul_le_mul_of_nonneg_right hp' hB'.le)
    _ ≤ (1 + rootN / 2) * (k + (B : ℝ)) := by
      nlinarith [mul_le_mul_of_nonneg_left hsum hr]
    _ ≤ _ := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      linarith [mul_le_mul_of_nonneg_right ht' hB'.le]

end IndexRecall.FlatScan

#print axioms IndexRecall.PlanarGap.strict_optimal
#print axioms IndexRecall.PlanarGap.distinct_sites
#print axioms IndexRecall.PlanarGap.distinct_answers
#print axioms IndexRecall.PlanarGap.probe_thresholds
#print axioms IndexRecall.PlanarGap.batch_recall
#print axioms IndexRecall.PlanarGap.distance_counts
#print axioms IndexRecall.PlanarGap.total_ratio
#print axioms IndexRecall.FlatScan.balanced_work_cap
