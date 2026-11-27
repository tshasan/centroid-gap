import IndexRecall.SphericalOptimality
import IndexRecall.SphericalRouting

/-! # Spherical k-means with arbitrary unit centers -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The spherical k-means loss for a labeled assignment and its representatives. -/
def sphericalLoss (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k) (c : Fin k → E) : ℝ :=
  ∑ p, (1 - ⟪x p, c (f p)⟫_ℝ)

/-- The normalized mean of each row, using `normalized_mean_eq` to cancel its size. -/
def meanDirection (x : Fin k × Fin k → E) (i : Fin k) : E := unitDir (rowSum x i)

theorem meanDirection_unit (x : Fin k × Fin k → E) (i : Fin k) (h : rowSum x i ≠ 0) :
    ‖meanDirection x i‖ = 1 := by
  simp [meanDirection, unitDir, norm_smul, norm_ne_zero_iff.mpr h]

/-- The canonical assignment and its normalized means are feasible and minimize
spherical k-means loss. A competing pair attains equality precisely when one
permutation carries both the memberships and the centers to the canonical pair. -/
structure UniqueOptimum (x : Fin k × Fin k → E) : Prop where
  unit_centers : ∀ i, ‖meanDirection x i‖ = 1
  minimal : ∀ (f : Fin k × Fin k → Fin k) (c : Fin k → E), (∀ j, ‖c j‖ = 1) →
    sphericalLoss x Prod.fst (meanDirection x) ≤ sphericalLoss x f c
  equality_iff : ∀ (f : Fin k × Fin k → Fin k) (c : Fin k → E), (∀ j, ‖c j‖ = 1) →
    (sphericalLoss x f c = sphericalLoss x Prod.fst (meanDirection x) ↔
      ∃ e : Equiv.Perm (Fin k), (∀ p, f p = e p.1) ∧ ∀ i, c (e i) = meanDirection x i)

private theorem row_score (x : Fin k × Fin k → E) (c : Fin k → E) :
    (∑ p, ⟪x p, c p.1⟫_ℝ) = ∑ i, ⟪rowSum x i, c i⟫_ℝ := by
  simp only [Fintype.sum_prod_type, rowSum, sum_inner]

private theorem canonical_score (x : Fin k × Fin k → E) (hn : ∀ i, rowSum x i ≠ 0) :
    (∑ p, ⟪x p, meanDirection x p.1⟫_ℝ) = quality x Prod.fst := by
  rw [row_score, quality]
  apply Finset.sum_congr rfl
  intro i _
  change ⟪rowSum x i, ‖rowSum x i‖⁻¹ • rowSum x i⟫_ℝ = _
  rw [real_inner_smul_right, real_inner_self_eq_norm_sq, clusterSum_row]
  change ‖rowSum x i‖⁻¹ * ‖rowSum x i‖ ^ 2 = ‖rowSum x i‖
  field_simp [norm_ne_zero_iff.mpr (hn i)]

/-- Strict optimality for the reduced assignment objective implies uniqueness
of the complete solution, including centers, whenever the winning sums are nonzero. -/
theorem uniqueOptimum_of_quality (x : Fin k × Fin k → E) (hn : ∀ i, rowSum x i ≠ 0)
    (hs : ∀ f, ¬Relabeled f → quality x f < quality x Prod.fst) : UniqueOptimum x := by
  have hbase := canonical_score x hn
  have hmax (f : Fin k × Fin k → Fin k) : quality x f ≤ quality x Prod.fst := by
    by_cases hf : Relabeled f
    · exact (quality_relabel x f hf).le
    · exact (hs f hf).le
  refine ⟨fun i => meanDirection_unit x i (hn i), ?_, ?_⟩
  · intro f c hc
    simp only [sphericalLoss, Finset.sum_sub_distrib, hbase]
    exact sub_le_sub_left ((center_score_le_quality x f c hc).trans (hmax f)) _
  · intro f c hc
    constructor
    · intro h
      have hscore : (∑ p, ⟪x p, c (f p)⟫_ℝ) = quality x Prod.fst := by
        simpa only [sphericalLoss, Finset.sum_sub_distrib, hbase, sub_right_inj] using h
      have hf : Relabeled f := by
        by_contra hf
        exact ((center_score_le_quality x f c hc).trans_lt (hs f hf)).ne hscore
      obtain ⟨e, he⟩ := hf
      have hrows : (∑ i, ⟪rowSum x i, c (e i)⟫_ℝ) = ∑ i, ‖rowSum x i‖ := calc
        _ = ∑ p, ⟪x p, c (e p.1)⟫_ℝ := (row_score x (fun i => c (e i))).symm
        _ = quality x Prod.fst := by simpa only [he] using hscore
        _ = _ := by simp only [quality, clusterSum_row, rowSum]
      have hle (i : Fin k) : ⟪rowSum x i, c (e i)⟫_ℝ ≤ ‖rowSum x i‖ := by
        simpa only [hc, mul_one] using real_inner_le_norm (rowSum x i) (c (e i))
      refine ⟨e, he, fun i => ?_⟩
      have hterm := (Finset.sum_eq_sum_iff_of_le (fun j _ => hle j)).mp hrows i (Finset.mem_univ i)
      have hdir := (inner_eq_norm_mul_iff_div (𝕜 := ℝ) (y := c (e i)) (hn i)).mp
        (by simpa [hc] using hterm)
      simpa [hc, meanDirection, unitDir] using hdir.symm
    · rintro ⟨e, he, hcenters⟩
      simp only [sphericalLoss, he, hcenters]

end IndexRecall.SphericalGap
