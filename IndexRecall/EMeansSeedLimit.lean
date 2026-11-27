import IndexRecall.EMeansSeedCounting
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Nat.Factorial.BigOperators

/-! # Limiting probability of one ordinary target seed -/

noncomputable section

open Filter Finset
open scoped Topology

namespace IndexRecall.SphericalGap.EMeans

/-- The chance that all `r` draws come from `a` of the `b` documents. -/
private lemma choose_ratio_product (a b r : ℕ) :
    (a.choose r : ℝ) / (b.choose r : ℝ) =
      ∏ i ∈ range r, ((a - i : ℕ) : ℝ) / ((b - i : ℕ) : ℝ) := by
  rw [prod_div_distrib, ← Nat.cast_prod, ← Nat.cast_prod, ← Nat.descFactorial_eq_prod_range,
    ← Nat.descFactorial_eq_prod_range]
  simp only [Nat.descFactorial_eq_factorial_mul_choose, Nat.cast_mul]
  exact (mul_div_mul_left _ _ (Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero r))).symm

/-- The probability of missing the target row entirely. -/
private def allForeign (k : ℕ) : ℝ :=
  ((k ^ 2 - k).choose k : ℝ) / ((k ^ 2).choose k : ℝ)

private lemma seedProbability_eq_allForeign {k : ℕ} (hk : 2 ≤ k) :
    seedProbability k = (k : ℝ) / (k - 1) * allForeign k := by
  have hsize : 2 * k ≤ k ^ 2 := by simpa [pow_two] using Nat.mul_le_mul_right k hk
  have hsq : k ≤ k ^ 2 := by omega
  have hpred : 1 ≤ k := by omega
  have hforeign : k - 1 ≤ k ^ 2 - k := by omega
  have hstep := Nat.choose_succ_right_eq (k ^ 2 - k) (k - 1)
  rw [Nat.sub_add_cancel hpred] at hstep
  have hcast := congrArg (Nat.cast : ℕ → ℝ) hstep
  simp only [Nat.cast_mul, Nat.cast_sub hforeign, Nat.cast_sub hsq, Nat.cast_sub hpred,
    Nat.cast_pow, Nat.cast_one] at hcast
  have hk' : (1 : ℝ) < k := by exact_mod_cast (show 1 < k by omega)
  have htotal : (↑((k ^ 2).choose k) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.choose_pos hsq).ne'
  rw [seedProbability, allForeign, Nat.cast_sub hpred, Nat.cast_one]
  field_simp [htotal, (sub_pos.mpr hk').ne']
  nlinarith only [hcast]

/-- Uniform bounds on the conditional probability at each draw. -/
private lemma draw_bounds {x y : ℝ} (hx : 2 < x) (hy₀ : 0 ≤ y) (hy : y < x) :
    Real.exp (-(x - 2)⁻¹) ≤ (x ^ 2 - x - y) / (x ^ 2 - y) ∧
      (x ^ 2 - x - y) / (x ^ 2 - y) ≤ Real.exp (-x⁻¹) := by
  have h₀ : 0 < x := by linarith
  have h₁ : 0 < x - 1 := by linarith
  have h₂ : 0 < x - 2 := by linarith
  have hd : 0 < x ^ 2 - y := by nlinarith [mul_pos h₀ h₁]
  constructor
  · calc
      Real.exp (-(x - 2)⁻¹) = (Real.exp ((x - 2)⁻¹))⁻¹ := Real.exp_neg _
      _ ≤ ((x - 2)⁻¹ + 1)⁻¹ := inv_anti₀ (by positivity) (Real.add_one_le_exp _)
      _ = (x - 2) / (x - 1) := by field_simp; ring
      _ ≤ (x ^ 2 - x - y) / (x ^ 2 - y) := (div_le_div_iff₀ h₁ hd).mpr (by nlinarith)
  · calc
      (x ^ 2 - x - y) / (x ^ 2 - y) ≤ (x - 1) / x :=
        (div_le_div_iff₀ hd h₀).mpr (by nlinarith)
      _ = -x⁻¹ + 1 := by field_simp; ring
      _ ≤ Real.exp (-x⁻¹) := Real.add_one_le_exp _

private lemma allForeign_bounds {k : ℕ} (hk : 3 ≤ k) :
    Real.exp (-((k : ℝ) / (k - 2))) ≤ allForeign k ∧ allForeign k ≤ Real.exp (-1) := by
  have hsize : 2 * k ≤ k ^ 2 := by
    simpa [pow_two] using Nat.mul_le_mul_right k (show 2 ≤ k by omega)
  have hsq : k ≤ k ^ 2 := by omega
  have hx : (2 : ℝ) < k := by exact_mod_cast (show 2 < k by omega)
  have hk' : (k : ℝ) ≠ 0 := by linarith
  have hd (i : ℕ) (hi : i ∈ range k) :
      Real.exp (-((k : ℝ) - 2)⁻¹) ≤ ((k ^ 2 - k - i : ℕ) : ℝ) / ((k ^ 2 - i : ℕ) : ℝ) ∧
      ((k ^ 2 - k - i : ℕ) : ℝ) / ((k ^ 2 - i : ℕ) : ℝ) ≤ Real.exp (-(k : ℝ)⁻¹) := by
    have hi' := mem_range.mp hi
    rw [Nat.cast_sub (show i ≤ k ^ 2 - k by omega), Nat.cast_sub hsq,
      Nat.cast_sub (show i ≤ k ^ 2 by omega), Nat.cast_pow]
    exact draw_bounds hx (by positivity) (by exact_mod_cast hi')
  rw [allForeign, choose_ratio_product]
  constructor
  · calc
      Real.exp (-((k : ℝ) / (k - 2))) = ∏ _i ∈ range k, Real.exp (-((k : ℝ) - 2)⁻¹) := by
        rw [prod_const, card_range, ← Real.exp_nat_mul]
        congr 1
        ring
      _ ≤ _ := prod_le_prod (fun _ _ => (Real.exp_pos _).le) (fun i hi => (hd i hi).1)
  · calc
      _ ≤ ∏ _i ∈ range k, Real.exp (-(k : ℝ)⁻¹) :=
        prod_le_prod (fun _ _ => by positivity) (fun i hi => (hd i hi).2)
      _ = Real.exp (-1) := by
        rw [prod_const, card_range, ← Real.exp_nat_mul, mul_neg, mul_inv_cancel₀ hk']

/-- The certified ordinary-seed event has limiting probability `exp (-1)`. -/
theorem tendsto_seedProbability : Tendsto seedProbability atTop (𝓝 (Real.exp (-1))) := by
  have hratio (c : ℝ) : Tendsto (fun k : ℕ => (k : ℝ) / (k - c)) atTop (𝓝 1) := by
    simpa only [sub_eq_add_neg] using tendsto_natCast_div_add_atTop (-c)
  have hlo : Tendsto (fun k : ℕ => Real.exp (-((k : ℝ) / (k - 2)))) atTop (𝓝 (Real.exp (-1))) :=
    Real.continuous_exp.continuousAt.tendsto.comp (hratio 2).neg
  have hforeign : Tendsto allForeign atTop (𝓝 (Real.exp (-1))) :=
    tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo tendsto_const_nhds
      ((eventually_ge_atTop 3).mono fun _ hk => (allForeign_bounds hk).1)
      ((eventually_ge_atTop 3).mono fun _ hk => (allForeign_bounds hk).2)
  simpa only [one_mul] using ((hratio 1).mul hforeign).congr'
    ((eventually_ge_atTop 2).mono fun _ hk => (seedProbability_eq_allForeign hk).symm)

end IndexRecall.SphericalGap.EMeans
