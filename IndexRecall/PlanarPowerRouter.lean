import IndexRecall.PlanarIVF

/-! # A power-diagram router closes the planar gap

Keep the unique k-means-optimal rows and their means as representatives. Score row `l` by
`‖q − mean l‖² − w l`, one real bias per row (a power-diagram router with the same `k` scores
as the plain centroid router). Then one probe finds every query's tip, so the full-recall
cost is `size s + size s`, the same as the representative-only fix `rerouted` and the
swapped layout. The mechanism: the decoy (upper) rows only outrank the true (lower) rows
because of height, a pure bias effect, while among the lower rows the centroid order is
already correct.
-/

noncomputable section

namespace IndexRecall.PlanarGap

/-- Power-diagram score of row `l` for query `i`: squared distance to the mean minus a bias. -/
def powerScore {s : ℕ} (w : Label s → ℝ) (i : Fin s) (l : Label s) : ℝ :=
  sqDist (query i) (mean site l) - w l

/-- Among lower rows the plain centroid order is already correct. -/
theorem lower_order {s : ℕ} (hs : 2 ≤ s) (i j : Fin s) (hj : j ≠ i) :
    sqDist (query i) (mean site (i, false)) < sqDist (query i) (mean site (j, false)) := by
  have hp : 40 ≤ pitch s := by unfold pitch; linarith [size_four hs]
  have hxy : ∀ k : Fin s, sqDist (query i) (mean site (k, false)) =
      ((query i).1 - (origin k + drift s))^2 + (height s - 2 / (size s : ℝ))^2 := by
    intro k
    simp [sqDist, mean_A hs, query]
  rw [hxy, hxy, add_lt_add_iff_right]
  have hq : (query i).1 - (origin i + drift s) = pitch s / 3 := by
    simp only [query]; ring
  rw [hq]
  rcases lt_or_gt_of_ne hj with h | h
  · -- j < i
    have hij : (j.val : ℝ) + 1 ≤ i.val := by exact_mod_cast (show j.val + 1 ≤ i.val from h)
    have hm := mul_le_mul_of_nonneg_right hij (show 0 ≤ pitch s by linarith)
    have hg : origin j + 3 * pitch s ≤ origin i := by unfold origin; linarith
    exact sq_lt_sq' (by linarith) (by linarith)
  · have hij : (i.val : ℝ) + 1 ≤ j.val := by exact_mod_cast (show i.val + 1 ≤ j.val from h)
    have hm := mul_le_mul_of_nonneg_right hij (show 0 ≤ pitch s by linarith)
    have hg : origin i + 3 * pitch s ≤ origin j := by unfold origin; linarith
    rw [← neg_sq ((query i).1 - (origin j + drift s))]
    exact sq_lt_sq' (by simp only [query]; linarith) (by simp only [query]; linarith)

/-- There is a bias vector under which every query's own lower row is the strict winner. -/
theorem exists_power_bias {s : ℕ} (hs : 2 ≤ s) :
    ∃ w : Label s → ℝ, ∀ i : Fin s, ∀ l : Label s, l ≠ (i, false) →
      powerScore w i (i, false) < powerScore w i l := by
  let f : Fin s × Fin s → ℝ := fun p =>
    sqDist (query p.1) (mean site (p.1, false)) - sqDist (query p.1) (mean site (p.2, true))
  have hB : ∀ i j : Fin s, f (i, j) ≤ ∑ p, |f p| := fun i j =>
    (le_abs_self _).trans (Finset.single_le_sum (f := fun p => |f p|)
      (fun p _ => abs_nonneg _) (Finset.mem_univ (i, j)))
  set B := ∑ p, |f p|
  refine ⟨fun l => if l.2 then 0 else B + 1, fun i l hl => ?_⟩
  obtain ⟨j, b⟩ := l
  cases b
  · have hj : j ≠ i := fun e => hl (by rw [e])
    have := lower_order hs i j hj
    simp only [powerScore]
    simp
    exact this
  · have := hB i j
    simp only [powerScore]
    simp only [Bool.false_eq_true, if_false, if_true, sub_zero]
    simp only at this
    linarith

/-- One probe, one bias per row: every query's tip is returned and the work is
`size s + size s`, the one-probe cost, against `size s + (s+1) * size s` for plain means. -/
theorem power_router_one_probe {s : ℕ} (hs : 2 ≤ s) :
    ∃ w : Label s → ℝ, ∀ i : Fin s,
      FlatScan.rank (powerScore w i) (i, false) = 1 ∧
      FlatScan.Returns (fun a : Label s × Slot s => sqDist (query i) (site a.1 a.2))
        (powerScore w i) 1 ((i, false), firstSlot hs) ∧
      FlatScan.work (Slot s) (powerScore w i) 1 = size s + size s := by
  obtain ⟨w, hw⟩ := exists_power_bias hs
  refine ⟨w, fun i => ?_⟩
  have hr := FlatScan.rank_first (score := powerScore w i) (i := (i, false)) (hw i)
  have hprobed : FlatScan.probed (powerScore w i) 1 = {(i, false)} := by
    ext l
    simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_singleton]
    constructor
    · intro h
      by_contra hne
      have hmem : (i, false) ∈ Finset.univ.filter
          (fun j => powerScore w i j < powerScore w i l) := by
        simpa using hw i l hne
      have : 0 < (Finset.univ.filter (fun j => powerScore w i j < powerScore w i l)).card :=
        Finset.card_pos.mpr ⟨_, hmem⟩
      unfold FlatScan.rank at h
      omega
    · rintro rfl; exact hr.le
  have hmin : ∀ b ≠ ((i, false), firstSlot hs),
      sqDist (query i) (site (i, false) (firstSlot hs)) < sqDist (query i) (site b.1 b.2) := by
    intro b hb
    simpa using uniqueNearest hs i b.1 b.2 hb
  have hret := FlatScan.returns_iff (τ := Slot s)
    (pointScore := fun a : Label s × Slot s => sqDist (query i) (site a.1 a.2))
    (powerScore w i) 1 ((i, false), firstSlot hs) hmin hr
  refine ⟨hr, hret.2 le_rfl, ?_⟩
  have hc : (FlatScan.candidates (Slot s) (powerScore w i) 1).card = size s := by
    rw [FlatScan.candidates, hprobed, Finset.card_product]
    simp [size, mul_comm]
  rw [FlatScan.work, hc]
  simp [size, mul_comm]


end IndexRecall.PlanarGap
