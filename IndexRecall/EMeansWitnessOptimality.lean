import IndexRecall.SphericalOptimality
import IndexRecall.SphericalVectors

/-! # An explicit perturbation radius for the spherical k-means optimum -/

noncomputable section

open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

variable {k : ℕ}

/-- The optimized partition score moves by at most the sum of the document
displacements, uniformly over assignments of any sizes. -/
theorem quality_movement {E : Type*} [NormedAddCommGroup E] (x y : Fin k × Fin k → E)
    (f : Fin k × Fin k → Fin k) :
    |quality x f - quality y f| ≤ ∑ p, ‖x p - y p‖ :=
  SphericalGap.quality_movement x y f

/-- Every different partition of the repeated data trails by more than `1/20`. -/
theorem base_quality_gap (z : Fin k) (f : Fin k × Fin k → Fin k) (hf : ¬Relabeled f) :
    quality (Vectors.base z) f + 1 / 20 < quality (Vectors.base z) Prod.fst := by
  open Vectors in
    have hgap := repeated_quality_gap z (anchor z) (tip z) (anchor_unit z) (tip_unit z)
      (fun _ _ h => base_cross z h) f hf
    change quality (base z) f + (⟪tip z, anchor z z⟫_ℝ - 4 / 5) < quality (base z) Prod.fst at hgap
    rw [tip_anchor] at hgap
    have hs : (17 : ℝ) / 10 < Real.sqrt 3 := by
      apply (sq_lt_sq₀ (by norm_num) (Real.sqrt_nonneg _)).mp
      rw [Real.sq_sqrt (by norm_num)]
      norm_num
    linarith

/-- At `k²` addresses the allowed per-document displacement totals at most `1/100`.
Even after both partition scores move, the canonical partition still strictly wins. -/
theorem strict_quality_of_close (hk : 2 ≤ k) (z : Fin k) (x : Fin k × Fin k → Vectors.Space k)
    (hx : ∀ p, ‖x p - Vectors.base z p‖ ≤ 1 / (100 * (k : ℝ) ^ 2))
    (f : Fin k × Fin k → Fin k) (hf : ¬Relabeled f) :
    quality x f < quality x Prod.fst := by
  have hsum : (∑ p, ‖x p - Vectors.base z p‖) ≤ 1 / 100 := calc
    _ ≤ ∑ _p : Fin k × Fin k, 1 / (100 * (k : ℝ) ^ 2) := Finset.sum_le_sum fun p _ => hx p
    _ = _ := by
      have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast (show k ≠ 0 by omega)
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
        nsmul_eq_mul, Nat.cast_mul]
      field_simp
  have hmove (g : Fin k × Fin k → Fin k) :=
    abs_le.mp ((quality_movement x (Vectors.base z) g).trans hsum)
  linarith [(hmove f).2, (hmove Prod.fst).1, base_quality_gap z f hf]

end IndexRecall.SphericalGap.EMeans
