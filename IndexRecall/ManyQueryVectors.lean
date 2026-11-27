import IndexRecall.SphericalCenters
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # One query and one tip in each of k orthogonal planes -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.ManyQuery

abbrev Space (k : ℕ) := EuclideanSpace ℝ (Fin k ⊕ Fin k)
variable {k : ℕ}

theorem dimension (k : ℕ) : Module.finrank ℝ (Space k) = 2 * k := by simp [Space, two_mul]

def e (i : Fin k) : Space k := EuclideanSpace.single (.inl i) 1
def f (i : Fin k) : Space k := EuclideanSpace.single (.inr i) 1
def plane (i : Fin k) (a b : ℝ) := a • e i + b • f i
def tip (i : Fin k) := plane i (Real.sqrt 3 / 2) (1 / 2)
def query (i : Fin k) := plane i (-1 / Real.sqrt 10) (3 / Real.sqrt 10)

theorem plane_inner (i j : Fin k) (a b c d : ℝ) :
    ⟪plane i a b, plane j c d⟫_ℝ = if i = j then a*c + b*d else 0 := by
  by_cases h : i = j
  · subst j
    simp [plane, e, f, inner_add_left, inner_add_right, real_inner_smul_left,
      real_inner_smul_right, EuclideanSpace.inner_single_left]
    ring
  · simp [plane, e, f, inner_add_left, inner_add_right, real_inner_smul_left,
      real_inner_smul_right, EuclideanSpace.inner_single_left, h]

theorem e_unit (i : Fin k) : ‖e i‖ = 1 := by simp [e]
theorem f_unit (i : Fin k) : ‖f i‖ = 1 := by simp [f]
theorem plane_one_zero (i : Fin k) : plane i 1 0 = e i := by simp [plane]

private theorem plane_norm_sq (i : Fin k) (a b : ℝ) : ‖plane i a b‖ ^ 2 = a^2 + b^2 := by
  rw [← real_inner_self_eq_norm_sq, plane_inner, if_pos rfl]
  ring

theorem tip_unit (i : Fin k) : ‖tip i‖ = 1 := by
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [tip, plane_norm_sq]
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]

theorem query_unit (i : Fin k) : ‖query i‖ = 1 := by
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [query, plane_norm_sq, div_pow, div_pow, Real.sq_sqrt (by norm_num)]
  norm_num

theorem e_tip (i : Fin k) : ⟪e i, tip i⟫_ℝ = Real.sqrt 3 / 2 := by
  rw [← plane_one_zero, tip, plane_inner]
  simp

def base (p : Fin k × Fin k) : Space k := if p.2 = p.1 then tip p.1 else e p.1

theorem base_unit (p : Fin k × Fin k) : ‖base p‖ = 1 := by
  unfold base
  split <;> simp [tip_unit, e_unit]

theorem base_same_nonneg (i a b : Fin k) : 0 ≤ ⟪base (i,a), base (i,b)⟫_ℝ := by
  have he : 0 ≤ ⟪e i, tip i⟫_ℝ := by rw [e_tip]; positivity
  have ht : 0 ≤ ⟪tip i, e i⟫_ℝ := by rwa [real_inner_comm]
  by_cases ha : a = i <;> by_cases hb : b = i <;>
    simp [base, ha, hb, real_inner_self_eq_norm_sq, tip_unit, e_unit, he, ht]

theorem base_cross {i j : Fin k} (h : i ≠ j) (a b : Fin k) :
    ⟪base (i,a), base (j,b)⟫_ℝ = 0 := by
  unfold base
  split_ifs <;> simp [tip, ← plane_one_zero, plane_inner, h]

theorem base_rowSum (i : Fin k) :
    rowSum base i = ((k - 1 : ℕ) : ℝ) • e i + tip i := by
  unfold rowSum
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i)]
  have hs : (∑ j ∈ Finset.univ.erase i, base (i,j)) = ∑ _j ∈ Finset.univ.erase i, e i :=
    Finset.sum_congr rfl fun j hj => by simp [base, (Finset.mem_erase.mp hj).1]
  rw [hs]
  simp [base, Nat.cast_smul_eq_nsmul]

theorem base_row_lower (i : Fin k) :
    (k : ℝ) - (1 - Real.sqrt 3 / 2) ≤ ‖rowSum base i‖ := by
  calc
    _ = ⟪e i, rowSum base i⟫_ℝ := by
      rw [base_rowSum, inner_add_right, real_inner_smul_right, real_inner_self_eq_norm_sq,
        e_unit, one_pow, mul_one, e_tip, Nat.cast_sub (Fin.pos i), Nat.cast_one]
      ring
    _ ≤ _ := by simpa [e_unit] using real_inner_le_norm (e i) (rowSum base i)

/-- The second coordinate distinguishes the ordinary documents within each plane. -/
def ordinary (i : Fin k) (s : ℝ) := plane i (Real.sqrt (1 - s^2)) (-s)
def tilt (t : ℝ) (j : Fin k) : ℝ := t * ((j.val : ℝ) + 1) / k
def site (t : ℝ) (p : Fin k × Fin k) : Space k :=
  if p.2 = p.1 then tip p.1 else ordinary p.1 (tilt t p.2)

theorem site_tip (t : ℝ) (i : Fin k) : site t (i,i) = tip i := by simp [site]
theorem site_zero : site (k := k) 0 = base := by
  funext p
  simp [site, base, ordinary, tilt, plane_one_zero]

private theorem tilt_bounds {t : ℝ} (ht : 0 ≤ t) (j : Fin k) :
    0 ≤ tilt t j ∧ tilt t j ≤ t := by
  have hk : 0 < (k : ℝ) := by exact_mod_cast Fin.pos j
  have hj : (j.val : ℝ) + 1 ≤ k := by exact_mod_cast j.isLt
  exact ⟨by unfold tilt; positivity, (div_le_iff₀ hk).mpr (mul_le_mul_of_nonneg_left hj ht)⟩

private theorem root_bounds {s : ℝ} (h0 : 0 ≤ s) (h1 : s ≤ 1) :
    0 ≤ 1 - s^2 ∧ 1 - s ≤ Real.sqrt (1 - s^2) ∧ Real.sqrt (1 - s^2) ≤ 1 := by
  have h : 0 ≤ 1 - s^2 := by nlinarith [mul_nonneg (sub_nonneg.mpr h1) (by linarith : 0 ≤ 1 + s)]
  refine ⟨h, (sq_le_sq₀ (sub_nonneg.mpr h1) (Real.sqrt_nonneg _)).mp ?_,
    (sq_le_sq₀ (Real.sqrt_nonneg _) zero_le_one).mp ?_⟩ <;> rw [Real.sq_sqrt h]
  · nlinarith [mul_nonneg h0 (sub_nonneg.mpr h1)]
  · nlinarith [sq_nonneg s]

theorem site_unit {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (p : Fin k × Fin k) : ‖site t p‖ = 1 := by
  by_cases h : p.2 = p.1
  · simpa [site, h] using tip_unit p.1
  have hs := tilt_bounds h0 p.2
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [site, if_neg h, ordinary, plane_norm_sq, Real.sq_sqrt (root_bounds hs.1 (hs.2.trans h1)).1]
  ring

theorem site_close {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (p : Fin k × Fin k) :
    ‖site t p - base p‖ ≤ 2 * t := by
  by_cases h : p.2 = p.1
  · simp [site, base, h, h0]
  let s := tilt t p.2
  have hs := tilt_bounds h0 p.2
  obtain ⟨_, hlo, hhi⟩ := root_bounds hs.1 (hs.2.trans h1)
  have he : site t p - base p =
      (Real.sqrt (1 - s^2) - 1) • e p.1 + (-s) • f p.1 := by
    simp only [site, base, if_neg h, ordinary, plane, sub_smul, one_smul, s]
    abel
  rw [he]
  apply (norm_add_le _ _).trans
  simp only [norm_smul, Real.norm_eq_abs, e_unit, f_unit, mul_one,
    abs_of_nonpos (sub_nonpos.mpr hhi), abs_neg, abs_of_nonneg hs.1, s]
  linarith [hs.2]

private theorem site_cross (t : ℝ) {p b : Fin k × Fin k} (h : p.1 ≠ b.1) :
    ⟪site t p, site t b⟫_ℝ = 0 := by
  unfold site
  split_ifs <;> simp [tip, ordinary, plane_inner, h]

theorem site_injective {t : ℝ} (ht : 0 < t) (h1 : t ≤ 1) : Function.Injective (site (k := k) t) := by
  rintro ⟨i,a⟩ ⟨j,b⟩ he
  have hij : i = j := by
    by_contra h
    have hc := site_cross t (p := (i,a)) (b := (j,b)) h
    rw [he, real_inner_self_eq_norm_sq, site_unit ht.le h1] at hc
    norm_num at hc
  subst j
  have coord := congrArg (fun v => ⟪plane i 0 1, v⟫_ℝ) he
  have hneg (j : Fin k) : -tilt t j ≤ 0 := neg_nonpos.mpr (tilt_bounds ht.le j).1
  by_cases ha : a = i <;> by_cases hb : b = i <;>
    simp [site, ha, hb, tip, ordinary, plane_inner] at coord ⊢
  · linarith [hneg b]
  · linarith [hneg a]
  · apply Fin.ext
    have hk : (k : ℝ) ≠ 0 := by exact_mod_cast (Fin.pos i).ne'
    unfold tilt at coord
    have hh := mul_left_cancel₀ ht.ne' ((div_left_inj' hk).mp coord)
    exact_mod_cast add_right_cancel hh

def tipScore : ℝ := ((3 - Real.sqrt 3) / 2) / Real.sqrt 10

theorem tipScore_bounds : 9 / 100 < tipScore ∧ tipScore < 1 / Real.sqrt 10 := by
  have hp : 0 < Real.sqrt 10 := by positivity
  have h3 := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have h10 := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 10)
  constructor
  · apply (lt_div_iff₀ hp).mpr
    linarith [show Real.sqrt 3 < 2 by nlinarith, show Real.sqrt 10 < 4 by nlinarith]
  · apply (div_lt_div_iff_of_pos_right hp).mpr
    nlinarith [Real.sqrt_nonneg 3]

/-- The table of all tips is a single function of the list label. -/
theorem query_tip (i j : Fin k) : ⟪query i, tip j⟫_ℝ = if i = j then tipScore else 0 := by
  rw [query, tip, plane_inner]
  by_cases h : i = j
  · simp only [if_pos h, tipScore]
    ring
  · simp [h]

theorem tips_first (i j : Fin k) (hj : j ≠ i) :
    ⟪query i, tip j⟫_ℝ < ⟪query i, tip i⟫_ℝ := by
  rw [query_tip, query_tip, if_neg hj.symm, if_pos rfl]
  linarith [tipScore_bounds.1]

private theorem query_foreign (i : Fin k) (t : ℝ) (p : Fin k × Fin k) (hp : p.1 ≠ i) :
    ⟪query i, site t p⟫_ℝ = 0 := by
  unfold site
  split <;> simp [query, tip, ordinary, plane_inner, hp.symm]

/-- Tilting towards negative f keeps an ordinary document's score at most its
unperturbed score, since sqrt(1-s²) + 3s ≥ 1 on the chosen interval. -/
private theorem query_ordinary {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (i j : Fin k) (hj : j ≠ i) :
    ⟪query i, site t (i,j)⟫_ℝ ≤ -(1 / Real.sqrt 10) := by
  let s := tilt t j
  have hs := tilt_bounds h0 j
  have hr := (root_bounds hs.1 (hs.2.trans h1)).2.1
  calc
    _ = -(Real.sqrt (1 - s^2) + 3*s) / Real.sqrt 10 := by
      simp [site, hj, query, ordinary, plane_inner, s]
      ring
    _ ≤ _ := by
      rw [← neg_div]
      apply (div_le_div_iff_of_pos_right (by positivity)).mpr
      dsimp [s]
      linarith [hs.1]

theorem margin {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (i : Fin k) (p : Fin k × Fin k)
    (hp : p ≠ (i,i)) : ⟪query i, site t p⟫_ℝ + 9 / 100 < ⟪query i, site t (i,i)⟫_ℝ := by
  rw [site_tip, query_tip, if_pos rfl]
  obtain ⟨r,j⟩ := p
  by_cases hr : r = i
  · subst r
    have hj : j ≠ i := fun h => hp (by rw [h])
    linarith [query_ordinary h0 h1 i j hj, tipScore_bounds.1,
      show (0 : ℝ) ≤ 1 / Real.sqrt 10 by positivity]
  · rw [query_foreign i t (r,j) hr, zero_add]
    exact tipScore_bounds.1

theorem own_sum_negative (hk : 2 ≤ k) {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (i : Fin k) :
    ⟪query i, rowSum (site t) i⟫_ℝ < 0 := by
  have h := Finset.sum_le_sum (s := Finset.univ.erase i)
    (fun j hj => query_ordinary h0 h1 i j (Finset.mem_erase.mp hj).1)
  simp only [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul] at h
  have hk1 : (1 : ℝ) ≤ ((k - 1 : ℕ) : ℝ) := by exact_mod_cast (show 1 ≤ k - 1 by omega)
  have hneg := mul_le_mul_of_nonpos_right hk1 (neg_nonpos.mpr (by positivity : 0 ≤ 1 / Real.sqrt 10))
  rw [rowSum, inner_sum, ← Finset.sum_erase_add _ _ (Finset.mem_univ i), site_tip, query_tip, if_pos rfl]
  linarith [tipScore_bounds.2]

theorem rowSum_ne_zero (hk : 2 ≤ k) {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (i : Fin k) :
    rowSum (site t) i ≠ 0 := by
  intro h
  simpa [h] using own_sum_negative hk h0 h1 i

theorem means_last (hk : 2 ≤ k) {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (i j : Fin k) (hj : j ≠ i) :
    ⟪query i, meanDirection (site t) i⟫_ℝ < ⟪query i, meanDirection (site t) j⟫_ℝ := by
  have hf : ⟪query i, rowSum (site t) j⟫_ℝ = 0 := by
    rw [rowSum, inner_sum]
    exact Finset.sum_eq_zero fun b _ => query_foreign i t (j,b) hj
  simp only [meanDirection, unitDir, real_inner_smul_right, hf, mul_zero]
  exact mul_neg_of_pos_of_neg (inv_pos.mpr (norm_pos_iff.mpr (rowSum_ne_zero hk h0 h1 i)))
    (own_sum_negative hk h0 h1 i)

end IndexRecall.SphericalGap.ManyQuery
