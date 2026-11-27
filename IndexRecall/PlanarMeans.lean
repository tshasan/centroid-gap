import IndexRecall.PlanarFamily

/-! # Means of the paired rows -/

noncomputable section

namespace IndexRecall.PlanarGap

private theorem sum_zero_slot {s : ℕ} (hs : 2 ≤ s) (c : ℝ) :
    (∑ t : Slot s, if t.val = 0 then c else 0) = c := by
  let first : Slot s := ⟨0, size_pos hs⟩
  have hzero (t : Slot s) : t.val = 0 ↔ t = first := by
    change t.val = first.val ↔ t = first
    exact Fin.ext_iff.symm
  simp [hzero]

private theorem mean_x {s : ℕ} (hs : 2 ≤ s) (a b : ℝ) :
    (∑ t : Slot s, (a + delta t + (if t.val = 0 then b else 0))) / (size s : ℝ) =
      a + drift s + b / (size s : ℝ) := by
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, sum_zero_slot hs]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, drift]
  field_simp [(size_cast_pos hs).ne']

theorem mean_A {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    mean site (i, false) = (origin i + drift s, 2 / (size s : ℝ)) := by
  have hx := mean_x hs (origin i) 0
  simp [mean, site, baseX, sum_zero_slot hs, show
    ((∑ t : Slot s, (origin i + delta t)) / (size s : ℝ)) = origin i + drift s by
      simpa using hx]

theorem mean_D {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    mean site (i, true) = (origin i + pitch s + drift s, 1) := by
  have hx := mean_x hs (origin i + pitch s) 0
  simp [mean, site, baseX, (size_cast_pos hs).ne', show
    ((∑ t : Slot s, (origin i + pitch s + delta t)) / (size s : ℝ)) =
      origin i + pitch s + drift s by simpa using hx]

theorem mean_swap_A {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    mean swapped (i, false) =
      (origin i + drift s + pitch s / (size s : ℝ), 1 / (size s : ℝ)) := by
  have hx (t : Slot s) : (swapped (i, false) t).1 =
      origin i + delta t + (if t.val = 0 then pitch s else 0) := by
    by_cases ht : t.val = 0 <;> simp [swapped, exchange, site, baseX, ht, add_right_comm]
  have hy (t : Slot s) : (swapped (i, false) t).2 = if t.val = 0 then 1 else 0 := by
    by_cases ht : t.val = 0 <;> simp [swapped, exchange, site, ht]
  simp [mean, hx, hy, mean_x hs, sum_zero_slot hs]

theorem mean_swap_D {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    mean swapped (i, true) =
      (origin i + pitch s + drift s - pitch s / (size s : ℝ), 1 + 1 / (size s : ℝ)) := by
  have hx (t : Slot s) : (swapped (i, true) t).1 =
      origin i + pitch s + delta t + (if t.val = 0 then -pitch s else 0) := by
    by_cases ht : t.val = 0 <;> simp [swapped, exchange, site, baseX, ht, add_right_comm]
  have hy (t : Slot s) : (swapped (i, true) t).2 = 1 + (if t.val = 0 then 1 else 0) := by
    by_cases ht : t.val = 0 <;> norm_num [swapped, exchange, site, ht]
  rw [mean, Prod.mk.injEq]
  constructor
  · simpa only [hx, neg_div, ← sub_eq_add_neg] using mean_x hs (origin i + pitch s) (-pitch s)
  · simp only [hy, Finset.sum_add_distrib, sum_zero_slot hs, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one, add_div,
      div_self (size_cast_pos hs).ne']

end IndexRecall.PlanarGap
