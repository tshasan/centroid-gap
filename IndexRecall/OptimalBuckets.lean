import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Global squared-error optimality for well-separated rows

If a competing solution has sufficiently low cost, each anchor has a distinct
nearby center. A strict nearest-center condition then forces any optimum to
keep each row together, and its squared error is minimized by the row mean.
-/

open scoped BigOperators

namespace IndexRecall.OptimalBuckets

variable {k m : ℕ}

/-- The exact arithmetic mean of the points in row `i`. -/
noncomputable def rowMean (x : Fin k → Fin (m + 1) → ℝ) (i : Fin k) : ℝ :=
  (∑ t, x i t) / ((m : ℝ) + 1)

/-- Squared error of an arbitrary assignment to the competing centers. -/
noncomputable def assignmentLoss (x : Fin k → Fin (m + 1) → ℝ)
    (f : Fin k → Fin (m + 1) → Fin k) (c : Fin k → ℝ) : ℝ :=
  ∑ i, ∑ t, (x i t - c (f i t)) ^ 2

/-- Squared error when every point is assigned to its own row. -/
noncomputable def rowLoss (x : Fin k → Fin (m + 1) → ℝ) (c : Fin k → ℝ) : ℝ :=
  assignmentLoss x (fun i _ => i) c

/-- The arithmetic mean minimizes squared error on a nonempty finite row. -/
theorem scalar_mean_le {n : ℕ} (hn : 0 < n) (x : Fin n → ℝ) (c : ℝ) :
    let μ := (∑ t, x t) / (n : ℝ)
    (∑ t, (x t - μ)^2) ≤ ∑ t, (x t - c)^2 := by
  intro μ
  have hsize : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : ∑ t, (x t - μ) = 0 := by
    rw [Finset.sum_sub_distrib]
    simp [μ, nsmul_eq_mul, mul_div_cancel₀, hsize]
  have hdecomp :
      (∑ t, (x t - c)^2) = (∑ t, (x t - μ)^2) + ∑ _t : Fin n, (μ - c)^2 := by
    calc
      _ = ∑ t, ((x t - μ)^2 + ((μ - c)^2 + (2 * (μ - c)) * (x t - μ))) :=
        Finset.sum_congr rfl (fun _ _ => by ring)
      _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hsum, mul_zero, add_zero]
  rw [hdecomp]
  exact le_add_of_nonneg_right (Finset.sum_nonneg fun _ _ => sq_nonneg _)

private theorem rowLoss_mean_le (x : Fin k → Fin (m + 1) → ℝ) (c : Fin k → ℝ) :
    rowLoss x (rowMean x) ≤ rowLoss x c := by
  unfold rowLoss assignmentLoss
  apply Finset.sum_le_sum
  intro i _
  simpa [rowMean] using scalar_mean_le (Nat.zero_lt_succ m) (x i) (c i)

private theorem center_matching (x : Fin k → Fin (m + 1) → ℝ) (a : Fin k → ℝ)
    (ha : ∀ i j, i ≠ j → 1 ≤ |a i - a j|)
    (hclose : ∀ i (t : Fin m), |x i t.castSucc - a i| ≤ 1 / 8)
    (f : Fin k → Fin (m + 1) → Fin k) (c : Fin k → ℝ)
    (hsmall : assignmentLoss x f c < (m : ℝ) / 64) :
    ∃ e : Equiv.Perm (Fin k), ∀ i, |c (e i) - a i| < 1 / 4 := by
  classical
  have hcenters (i : Fin k) : ∃ j, |c j - a i| < 1 / 4 := by
    by_contra! hfar
    have hpoint (t : Fin m) : (1 : ℝ) / 64 ≤
        (x i t.castSucc - c (f i t.castSucc)) ^ 2 := by
      have htriangle := abs_sub_le (c (f i t.castSucc)) (x i t.castSucc) (a i)
      rw [abs_sub_comm (c (f i t.castSucc)) (x i t.castSucc)] at htriangle
      have hdist : (1 : ℝ) / 8 ≤ |x i t.castSucc - c (f i t.castSucc)| := by
        linarith [hfar (f i t.castSucc), hclose i t]
      have hsquare := sq_le_sq.mpr (show |(1 : ℝ) / 8| ≤
          |x i t.castSucc - c (f i t.castSucc)| by
            rw [abs_of_nonneg (by norm_num)]
            exact hdist)
      norm_num at hsquare ⊢
      exact hsquare
    have hlarge : (m : ℝ) / 64 ≤ assignmentLoss x f c := calc
      (m : ℝ) / 64 = ∑ _t : Fin m, (1 : ℝ) / 64 := by
        simp [nsmul_eq_mul, div_eq_mul_inv]
      _ ≤ ∑ t : Fin m, (x i t.castSucc - c (f i t.castSucc)) ^ 2 :=
        Finset.sum_le_sum fun t _ => hpoint t
      _ ≤ ∑ t : Fin (m + 1), (x i t - c (f i t)) ^ 2 := by
        rw [Fin.sum_univ_castSucc]
        exact le_add_of_nonneg_right (sq_nonneg _)
      _ ≤ assignmentLoss x f c := by
        unfold assignmentLoss
        apply Finset.single_le_sum _ (Finset.mem_univ i)
        exact fun j _ => Finset.sum_nonneg fun t _ => sq_nonneg (x j t - c (f j t))
    exact hsmall.not_ge hlarge
  choose e he using hcenters
  have h_inj : Function.Injective e := by
    intro i j hij
    by_contra hne
    have htriangle : |a i - a j| ≤ |c (e i) - a i| + |c (e j) - a j| := by
      rw [← hij, abs_sub_comm (c (e i)) (a i)]
      exact abs_sub_le (a i) (c (e i)) (a j)
    linarith [ha i j hne, he i, he j]
  exact ⟨Equiv.ofBijective e h_inj.bijective_of_finite, he⟩

/-- The row means give a global minimum over all assignments and centers.
Every assignment attaining that minimum uses the same rows up to a permutation. -/
theorem strict_optimal (x : Fin k → Fin (m + 1) → ℝ) (a : Fin k → ℝ)
    (ha : ∀ i j, i ≠ j → 1 ≤ |a i - a j|)
    (hclose : ∀ i (t : Fin m), |x i t.castSucc - a i| ≤ 1 / 8)
    (hbound : rowLoss x a < (m : ℝ) / 64)
    (hnearest : ∀ i j t u v, i ≠ j → |u - a i| < 1 / 4 → |v - a j| < 1 / 4 →
      |x i t - u| < |x i t - v|)
    (f : Fin k → Fin (m + 1) → Fin k) (c : Fin k → ℝ) :
    rowLoss x (rowMean x) ≤ assignmentLoss x f c ∧
      (assignmentLoss x f c ≤ rowLoss x (rowMean x) →
        ∃ e : Equiv.Perm (Fin k), ∀ i t, f i t = e i) := by
  classical
  by_cases hsmall : assignmentLoss x f c < (m : ℝ) / 64
  · obtain ⟨e, he⟩ := center_matching x a ha hclose f c hsmall
    have hstrict (i : Fin k) (t : Fin (m + 1)) (hne : f i t ≠ e i) :
        (x i t - c (e i)) ^ 2 < (x i t - c (f i t)) ^ 2 := by
      obtain ⟨j, hj⟩ := e.surjective (f i t)
      have hij : i ≠ j := by
        intro h
        exact hne (h ▸ hj.symm)
      rw [← hj]
      exact sq_lt_sq.mpr (hnearest i j t (c (e i)) (c (e j)) hij (he i) (he j))
    have hweak (i : Fin k) (t : Fin (m + 1)) :
        (x i t - c (e i)) ^ 2 ≤ (x i t - c (f i t)) ^ 2 := by
      by_cases h : f i t = e i
      · rw [h]
      · exact (hstrict i t h).le
    have hcenter : rowLoss x (fun i => c (e i)) ≤ assignmentLoss x f c := by
      unfold rowLoss assignmentLoss
      exact Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun t _ => hweak i t
    refine ⟨(rowLoss_mean_le x _).trans hcenter, fun h => ⟨e, ?_⟩⟩
    intro i t
    by_contra hne
    have hbetter : rowLoss x (fun i => c (e i)) < assignmentLoss x f c := by
      unfold rowLoss assignmentLoss
      exact Finset.sum_lt_sum (fun j _ => Finset.sum_le_sum fun s _ => hweak j s)
        ⟨i, Finset.mem_univ i,
          Finset.sum_lt_sum (fun s _ => hweak i s) ⟨t, Finset.mem_univ t, hstrict i t hne⟩⟩
    exact ((rowLoss_mean_le x _).trans_lt hbetter).not_ge h
  · have hbetter := (rowLoss_mean_le x a).trans_lt (hbound.trans_le (le_of_not_gt hsmall))
    exact ⟨hbetter.le, fun h => (hbetter.not_ge h).elim⟩

end IndexRecall.OptimalBuckets
