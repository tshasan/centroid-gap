import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # Unit vectors for the spherical IVF example

The query, each row, and each document address have separate coordinate axes.
A rotation toward the address axis separates the copies of each anchor.
-/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.Vectors

abbrev Coord (k : ℕ) := Option (Fin k ⊕ (Fin k × Fin k))
abbrev Space (k : ℕ) := EuclideanSpace ℝ (Coord k)

variable {k : ℕ}

/-- A vector in the plane of the query axis and row axis `i`. -/
private def axial (i : Fin k) (s h : ℝ) : Space k :=
  EuclideanSpace.single none s + EuclideanSpace.single (some (.inl i)) h

def query (k : ℕ) : Space k := EuclideanSpace.single none 1

def anchor (z i : Fin k) : Space k :=
  axial i (if i = z then 0 else 2 / 5) (if i = z then 1 else Real.sqrt 21 / 5)

def tip (z : Fin k) : Space k := axial z (1 / 2) (Real.sqrt 3 / 2)

private theorem axial_inner (i j : Fin k) (s h t d : ℝ) :
    ⟪axial i s h, axial j t d⟫_ℝ = s * t + if i = j then h * d else 0 := by
  by_cases hij : i = j
  · subst j
    simp [axial, inner_add_left, inner_add_right, EuclideanSpace.inner_single_left]
  · simp [axial, inner_add_left, inner_add_right, EuclideanSpace.inner_single_left, hij]

theorem query_unit (k : ℕ) : ‖query k‖ = 1 := by simp [query]

theorem anchor_unit (z i : Fin k) : ‖anchor z i‖ = 1 := by
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [← real_inner_self_eq_norm_sq, anchor, axial_inner]
  by_cases hi : i = z
  · norm_num [hi]
  · simp [hi]
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 21)]

theorem tip_unit (z : Fin k) : ‖tip z‖ = 1 := by
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [← real_inner_self_eq_norm_sq, tip, axial_inner, if_pos rfl]
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]

theorem query_anchor (z i : Fin k) :
    ⟪query k, anchor z i⟫_ℝ = if i = z then 0 else 2 / 5 := by
  simp [query, anchor, axial, inner_add_right, EuclideanSpace.inner_single_left]

theorem query_tip (z : Fin k) : ⟪query k, tip z⟫_ℝ = 1 / 2 := by
  simp [query, tip, axial, inner_add_right, EuclideanSpace.inner_single_left]

theorem tip_anchor (z : Fin k) : ⟪tip z, anchor z z⟫_ℝ = Real.sqrt 3 / 2 := by
  simp [tip, anchor, axial_inner]

/-- The repeated instance has one tip at address `(z,z)`. -/
def base (z : Fin k) (p : Fin k × Fin k) : Space k :=
  if p = (z,z) then tip z else anchor z p.1

theorem base_unit (z : Fin k) (p : Fin k × Fin k) : ‖base z p‖ = 1 := by
  by_cases hp : p = (z,z) <;> simp [base, hp, tip_unit, anchor_unit]

/-- Repeated documents in different rows have cosine at most `1/5`. -/
theorem base_cross_inner (z : Fin k) {p b : Fin k × Fin k} (h : p.1 ≠ b.1) :
    ⟪base z p, base z b⟫_ℝ ≤ 1 / 5 := by
  by_cases hp : p = (z,z)
  · subst p
    have hz : b.1 ≠ z := Ne.symm h
    have hb : b ≠ (z,z) := fun he => hz (congrArg Prod.fst he)
    norm_num [base, hb, tip, anchor, axial_inner, hz, hz.symm]
  · by_cases hb : b = (z,z)
    · subst b
      have hz : p.1 ≠ z := h
      norm_num [base, hp, tip, anchor, axial_inner, hz]
    · simp only [base, if_neg hp, if_neg hb, anchor, axial_inner, if_neg h, add_zero]
      split_ifs <;> norm_num

/-- Every pair whose document addresses belong to different rows incurs a
strictly positive spherical loss exceeding `1/5`. -/
theorem base_cross (z : Fin k) {p b : Fin k × Fin k} (h : p.1 ≠ b.1) :
    ‖base z p + base z b‖ < 9 / 5 := by
  apply (sq_lt_sq₀ (norm_nonneg _) (by norm_num)).mp
  rw [norm_add_sq_real, base_unit, base_unit]
  linarith [base_cross_inner z h]

private def addressAxis (p : Fin k × Fin k) : Space k :=
  EuclideanSpace.single (some (.inr p)) 1

private theorem anchor_address_inner (z i : Fin k) (p : Fin k × Fin k) :
    ⟪anchor z i, addressAxis p⟫_ℝ = 0 := by
  simp [anchor, axial, addressAxis, inner_add_left, EuclideanSpace.inner_single_left]

-- Rational parametrization of the unit circle with value (1,0) at zero.
private def along (ε : ℝ) : ℝ := (1 - ε ^ 2) / (1 + ε ^ 2)
private def across (ε : ℝ) : ℝ := 2 * ε / (1 + ε ^ 2)

private theorem circle (ε : ℝ) : along ε ^ 2 + across ε ^ 2 = 1 := by
  have hd : 1 + ε ^ 2 ≠ 0 := by positivity
  unfold along across
  field_simp
  ring

/-- Rotate each repeated anchor toward its own address axis, keeping the tip fixed. -/
def site (z : Fin k) (ε : ℝ) (p : Fin k × Fin k) : Space k :=
  if p = (z,z) then tip z else along ε • anchor z p.1 + across ε • addressAxis p

/-- Documents in different rows share only the query coordinate. -/
theorem site_cross_inner (z : Fin k) (ε : ℝ) {p b : Fin k × Fin k} (h : p.1 ≠ b.1) :
    ⟪site z ε p, site z ε b⟫_ℝ =
      ⟪query k, site z ε p⟫_ℝ * ⟪query k, site z ε b⟫_ℝ := by
  have hpq : p ≠ b := fun he => h (congrArg Prod.fst he)
  by_cases hp : p = (z,z)
  · subst p
    have hb : b ≠ (z,z) := hpq.symm
    simp [site, hb, anchor, tip, axial, addressAxis, query, inner_add_left, inner_add_right,
      EuclideanSpace.inner_single_left, h, h.symm]
  · by_cases hb : b = (z,z)
    · subst b
      simp [site, hp, anchor, tip, axial, addressAxis, query, inner_add_left, inner_add_right,
        real_inner_smul_left, EuclideanSpace.inner_single_left, h]
      ring
    · simp [site, hp, hb, anchor, axial, addressAxis, query, inner_add_left, inner_add_right,
        real_inner_smul_left, real_inner_smul_right, EuclideanSpace.inner_single_left,
        h, hpq]
      ring_nf

theorem site_tip (z : Fin k) (ε : ℝ) : site z ε (z,z) = tip z := by simp [site]

theorem site_zero (z : Fin k) : site z 0 = base z := by
  funext p
  simp [site, base, along, across]

theorem site_unit (z : Fin k) (ε : ℝ) (p : Fin k × Fin k) : ‖site z ε p‖ = 1 := by
  by_cases hp : p = (z,z)
  · simpa [site, hp] using tip_unit z
  · rw [site, if_neg hp]
    apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
    rw [norm_add_sq_real, real_inner_smul_left, real_inner_smul_right, anchor_address_inner]
    simpa [norm_smul, anchor_unit, addressAxis, Real.norm_eq_abs, sq_abs] using circle ε

theorem site_continuous (z : Fin k) (p : Fin k × Fin k) :
    Continuous (fun ε => site z ε p) := by
  by_cases hp : p = (z,z)
  · simpa only [site, if_pos hp] using (continuous_const : Continuous (fun _ : ℝ => tip z))
  · simp only [site, if_neg hp]
    have hd : ∀ ε : ℝ, 1 + ε ^ 2 ≠ 0 := fun ε => by positivity
    have hu : Continuous along :=
      (continuous_const.sub (continuous_id.pow 2)).div (continuous_const.add (continuous_id.pow 2)) hd
    have hv : Continuous across :=
      (continuous_const.mul continuous_id).div (continuous_const.add (continuous_id.pow 2)) hd
    exact (hu.smul continuous_const).add (hv.smul continuous_const)

/-- Rotating an anchor toward its address axis can only reduce its nonnegative
query projection. The tip keeps a cosine advantage of at least `1/10`. -/
theorem site_margin (z : Fin k) (ε : ℝ) (p : Fin k × Fin k) (hp : p ≠ (z,z)) :
    ⟪query k, site z ε p⟫_ℝ + 1 / 10 ≤ ⟪query k, site z ε (z,z)⟫_ℝ := by
  have hproj : ⟪query k, addressAxis p⟫_ℝ = 0 := by
    simp [query, addressAxis, EuclideanSpace.inner_single_left]
  have hu : along ε ≤ 1 := by
    apply (div_le_one (show 0 < 1 + ε ^ 2 by positivity)).mpr
    linarith [sq_nonneg ε]
  rw [site_tip, query_tip, site, if_neg hp, inner_add_right,
    real_inner_smul_right, real_inner_smul_right, query_anchor, hproj]
  split_ifs <;> linarith

/-- Every point on the path with parameter in `[0,1]` has moved at most `4ε`. -/
theorem site_displacement (z : Fin k) {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1) (p : Fin k × Fin k) :
    ‖site z ε p - base z p‖ ≤ 4 * ε := by
  by_cases hp : p = (z,z)
  · simp only [site, base, if_pos hp, sub_self, norm_zero]
    positivity
  · have hd : 0 < 1 + ε ^ 2 := by positivity
    have hu : |along ε - 1| ≤ 2 * ε := by
      rw [along, abs_of_nonpos (by exact sub_nonpos.mpr ((div_le_one hd).mpr (by linarith [sq_nonneg ε])))]
      apply (neg_le_iff_add_nonneg).2
      field_simp
      nlinarith [mul_nonneg h0 (sub_nonneg.mpr h1), mul_nonneg h0 (sq_nonneg ε)]
    have hv : |across ε| ≤ 2 * ε := by
      unfold across
      rw [abs_of_nonneg (by positivity), div_le_iff₀ hd]
      nlinarith [mul_nonneg h0 (sq_nonneg ε)]
    simp only [site, base, if_neg hp]
    have heq : along ε • anchor z p.1 + across ε • addressAxis p - anchor z p.1 =
        (along ε - 1) • anchor z p.1 + across ε • addressAxis p := by
      rw [sub_smul, one_smul]
      abel
    rw [heq]
    refine (norm_add_le _ _).trans ?_
    simp only [norm_smul, Real.norm_eq_abs, anchor_unit, addressAxis, EuclideanSpace.norm_single,
      norm_one, mul_one]
    linarith

/-- Only the tip has a nonzero query coordinate in the target row. -/
theorem query_target (z : Fin k) (ε : ℝ) (j : Fin k) :
    ⟪query k, site z ε (z,j)⟫_ℝ = if j = z then 1 / 2 else 0 := by
  by_cases hj : j = z
  · simpa [hj, site_tip] using query_tip z
  · simp [site, hj, inner_add_right, addressAxis, query, EuclideanSpace.inner_single_left,
      anchor, axial]

private theorem site_coordinate (z : Fin k) (ε : ℝ) (p r : Fin k × Fin k) :
    (site z ε p) (some (.inr r)) =
      if p = (z,z) then 0 else if r = p then across ε else 0 := by
  by_cases hp : p = (z,z) <;> by_cases hr : r = p <;>
    simp [site, hp, hr, tip, anchor, axial, addressAxis, PiLp.add_apply, PiLp.smul_apply]

theorem site_injective (z : Fin k) {ε : ℝ} (hε : 0 < ε) : Function.Injective (site z ε) := by
  have hv : across ε ≠ 0 := ne_of_gt (by unfold across; positivity)
  have distinguish {p b : Fin k × Fin k} (h : p ≠ b) (hp : p ≠ (z,z))
      (he : site z ε p = site z ε b) : False := by
    have hc := congrArg (fun v : Space k => v (some (.inr p))) he
    simp [site_coordinate, h, hp, hv] at hc
  intro p b he
  by_contra h
  by_cases hp : p = (z,z)
  · exact distinguish (Ne.symm h) (fun hb => h (hp.trans hb.symm)) he.symm
  · exact distinguish h hp he

end IndexRecall.SphericalGap.Vectors
