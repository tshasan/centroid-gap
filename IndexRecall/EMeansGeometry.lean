import IndexRecall.SphericalVectors
import IndexRecall.SphericalRouting
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.InnerProductSpace.Continuous

/-! # Stable assignment regions for spherical e-means

Each region constrains a raw center by all document cosines. Convexity controls
online averages, and a lower norm bound preserves the assignment after normalization.
-/

noncomputable section
open Filter
open scoped InnerProductSpace Topology

namespace IndexRecall.SphericalGap.EMeans

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Raw centers in this cell have a uniform angular separation between their own
row and the other documents. -/
def cell (x : Fin k × Fin k → E) (i : Fin k) : Set E :=
  {v | ‖v‖ ≤ 1 ∧ (∀ p, p.1 = i → 3 / 4 ≤ ⟪x p, v⟫_ℝ) ∧
    (∀ p, p.1 ≠ i → ⟪x p, v⟫_ℝ ≤ 1 / 3)}

theorem cell_convex (x : Fin k × Fin k → E) (i : Fin k) : Convex ℝ (cell x i) := by
  intro u hu v hv a b ha hb hab
  obtain ⟨hu, huOwn, huOther⟩ := hu
  obtain ⟨hv, hvOwn, hvOther⟩ := hv
  refine ⟨?_, ?_, ?_⟩
  · calc
      ‖a • u + b • v‖ ≤ ‖a • u‖ + ‖b • v‖ := norm_add_le _ _
      _ = a * ‖u‖ + b * ‖v‖ := by
        simp [norm_smul, Real.norm_eq_abs, abs_of_nonneg ha, abs_of_nonneg hb]
      _ ≤ a * 1 + b * 1 := add_le_add (mul_le_mul_of_nonneg_left hu ha)
        (mul_le_mul_of_nonneg_left hv hb)
      _ = 1 := by simpa using hab
  · intro p hp
    rw [inner_add_right, real_inner_smul_right, real_inner_smul_right]
    linarith [mul_le_mul_of_nonneg_left (huOwn p hp) ha, mul_le_mul_of_nonneg_left (hvOwn p hp) hb]
  · intro p hp
    rw [inner_add_right, real_inner_smul_right, real_inner_smul_right]
    linarith [mul_le_mul_of_nonneg_left (huOther p hp) ha, mul_le_mul_of_nonneg_left (hvOther p hp) hb]

theorem cell_norm_lower (x : Fin k × Fin k → E) (hu : ∀ p, ‖x p‖ = 1)
    {v : E} {i : Fin k} (hv : v ∈ cell x i) : 3 / 4 ≤ ‖v‖ :=
  (hv.2.1 (i,i) rfl).trans (by simpa only [hu, one_mul] using real_inner_le_norm (x (i,i)) v)

/-- A unit document chooses its own row uniquely when all normalized centers
come from the corresponding raw-center cells. -/
theorem owner_strict (x : Fin k × Fin k → E) (hu : ∀ p, ‖x p‖ = 1) (c : Fin k → E)
    (hc : ∀ i, c i ∈ cell x i) (p : Fin k × Fin k) (j : Fin k) (hj : j ≠ p.1) :
    ⟪x p, unitDir (c j)⟫_ℝ < ⟪x p, unitDir (c p.1)⟫_ℝ := by
  have hlow (i) := cell_norm_lower x hu (hc i)
  have hpos (i) : 0 < ‖c i‖ := by linarith [hlow i]
  have own : 3 / 4 ≤ ⟪x p, unitDir (c p.1)⟫_ℝ := by
    rw [unitDir, real_inner_smul_right, mul_comm, ← div_eq_mul_inv, le_div_iff₀ (hpos _)]
    linarith [(hc p.1).1, (hc p.1).2.1 p rfl]
  have other : ⟪x p, unitDir (c j)⟫_ℝ ≤ 4 / 9 := by
    rw [unitDir, real_inner_smul_right, mul_comm, ← div_eq_mul_inv, div_le_iff₀ (hpos _)]
    linarith [(hc j).2.2 p (Ne.symm hj), hlow j]
  linarith

open Vectors

private theorem base_same_inner (z : Fin k) {p b : Fin k × Fin k} (h : p.1 = b.1) :
    4 / 5 < ⟪base z p, base z b⟫_ℝ := by
  have ht : 4 / 5 < ⟪tip z, anchor z z⟫_ℝ := by
    rw [tip_anchor]
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]
  by_cases hp : p = (z,z)
  · subst p
    by_cases hb : b = (z,z)
    · norm_num [hb, base, real_inner_self_eq_norm_sq, tip_unit]
    · simpa only [base, if_pos rfl, if_neg hb, ← h] using ht
  · by_cases hb : b = (z,z)
    · subst b
      simpa only [base, if_neg hp, if_pos rfl, h, real_inner_comm (anchor z z) (tip z)] using ht
    · norm_num [base, hp, hb, h, real_inner_self_eq_norm_sq, anchor_unit]

private theorem dot_movement (u v a b : E) (hv : ‖v‖ = 1) (ha : ‖a‖ = 1) :
    |⟪u, v⟫_ℝ - ⟪a, b⟫_ℝ| ≤ ‖u - a‖ + ‖v - b‖ := calc
  _ = |⟪u - a, v⟫_ℝ + ⟪a, v - b⟫_ℝ| := by
    rw [inner_sub_left, inner_sub_right]
    congr 1
    ring
  _ ≤ |⟪u - a, v⟫_ℝ| + |⟪a, v - b⟫_ℝ| := abs_add_le _ _
  _ ≤ _ := by
    simpa only [hv, ha, mul_one, one_mul] using
      add_le_add (abs_real_inner_le_norm (u - a) v) (abs_real_inner_le_norm a (v - b))

private theorem normalized_near_anchor {a v : E} (ha : ‖a‖ = 1) (hv : ‖v - a‖ < 1 / 100) :
    ‖unitDir v‖ = 1 ∧ ‖unitDir v - a‖ < 1 / 50 := by
  have hn : ‖v‖ ≠ 0 := by
    intro hz
    have hv0 := norm_eq_zero.mp hz
    norm_num [hv0, ha] at hv
  have hu : ‖unitDir v‖ = 1 := by simp [unitDir, norm_smul, hn]
  have hrec : ‖v‖ • unitDir v = v := by simp [unitDir, smul_smul, hn]
  have hdelta : ‖unitDir v - v‖ ≤ ‖v - a‖ := calc
    _ = ‖(1 - ‖v‖) • unitDir v‖ := by rw [sub_smul, one_smul, hrec]
    _ = |1 - ‖v‖| := by rw [norm_smul, hu, mul_one, Real.norm_eq_abs]
    _ ≤ _ := by
      rw [abs_sub_comm]
      simpa only [ha] using abs_norm_sub_norm_le v a
  have htri := dist_triangle (unitDir v) v a
  simp only [dist_eq_norm] at htri
  exact ⟨hu, by linarith⟩

private theorem mem_cell_near (z : Fin k) (x : Fin k × Fin k → Space k)
    (hx : ∀ p, ‖x p - base z p‖ < 1 / 100) (w : Fin k × Fin k) (v : Space k)
    (hv : ‖v‖ = 1) (hw : ‖v - base z w‖ < 1 / 50) : v ∈ cell x w.1 := by
  have hmove (p : Fin k × Fin k) :
      |⟪x p, v⟫_ℝ - ⟪base z p, base z w⟫_ℝ| ≤ 3 / 100 := by
    linarith [dot_movement (x p) v (base z p) (base z w) hv (base_unit z p), hx p]
  refine ⟨hv.le, ?_, ?_⟩
  · intro p h
    linarith [(abs_le.mp (hmove p)).1, base_same_inner z h]
  · intro p h
    linarith [(abs_le.mp (hmove p)).2, base_cross_inner z h]

/-- A uniform bound on document displacement suffices for membership in the cells. -/
theorem cells_of_close (hk : 2 ≤ k) (z : Fin k) (x : Fin k × Fin k → Space k)
    (hu : ∀ p, ‖x p‖ = 1) (hx : ∀ p, ‖x p - base z p‖ < 1 / 100) :
    (∀ p, x p ∈ cell x p.1) ∧
      (∀ i (v : Space k), ‖v - anchor z i‖ < 1 / 100 → unitDir v ∈ cell x i) := by
  haveI : Nontrivial (Fin k) := Fin.nontrivial_iff_two_le.mpr hk
  obtain ⟨s, hsz⟩ := exists_ne z
  refine ⟨fun p => mem_cell_near z _ hx p (x p) (hu p) (by linarith [hx p]), ?_⟩
  intro i v hv
  obtain ⟨hu, hdist⟩ := normalized_near_anchor (anchor_unit z i) hv
  have ha : base z (i,s) = anchor z i := by simp [base, hsz]
  exact mem_cell_near z _ hx (i,s) (unitDir v) hu (by rwa [ha])

/-- For all sufficiently small positive perturbations, each document belongs to
its own cell. The normalized initial center also belongs whenever its raw value
is within `1/100` of the labeled anchor, uniformly in the number of lists. -/
theorem cells_near (hk : 2 ≤ k) (z : Fin k) :
    ∀ᶠ t in 𝓝[>] (0 : ℝ), (∀ p, site z t p ∈ cell (site z t) p.1) ∧
      (∀ i (v : Space k), ‖v - anchor z i‖ < 1 / 100 → unitDir v ∈ cell (site z t) i) := by
  have hnear : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ p, ‖site z t p - base z p‖ < 1 / 100 := by
    apply eventually_all.mpr
    intro p
    exact ((site_continuous z p).sub continuous_const).norm.continuousAt.eventually_lt
      continuousAt_const (by simp [site_zero])
  filter_upwards [hnear.filter_mono nhdsWithin_le_nhds] with t ht
  exact cells_of_close hk z _ (site_unit z t) ht

end IndexRecall.SphericalGap.EMeans
