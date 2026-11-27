import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # Bidirectional RobustPrune Failure under OOD MIPS (Theorem A1)

For every beam width `L ≥ 1`, we construct an explicit rational unit-sphere dataset
`S_L ⊂ S² ⊂ ℝ³` of `L + 3` distinct points:
- `c(t_j)` for `j = 0, …, L` with `t_j = j / (500 * L)` along the rational curve
  `c(t) = ((1 - 37 t²) / (1 + 37 t²), 2 t / (1 + 37 t²), 12 t / (1 + 37 t²))`
  (so `u = c(0) = (1, 0, 0)` and decoys `d_j = c(t_j)` for `j = 1, …, L`),
- a blocker `p* = (16/29, 24/29, -3/29)`,
- a winner `w = (-2/7, 6/7, 3/7)`,
and an orthogonal unit query `q = (0, 0, 1)`.

We prove, for every `L ≥ 1` and every real `α ∈ [1, 3/2]`:
1. Every point in `S_L` and `q` is a unit vector in `ℝ³`, and all `L + 3` points are distinct.
2. **Large cosine margin**: `⟪w, q⟫ - ⟪x, q⟫ > 2/5` for every `x ∈ S_L \ {w}`.
3. **Bidirectional RobustPrune isolation** (even with infinite degree `R = ∞` and exact full
   candidate set `S_L \ {v}`):
   - From every `c(t_j)` (`0 ≤ j ≤ L`), `p*` is strictly closer than `w` and satisfies
     `α² * ‖p* - w‖² ≤ ‖c(t_j) - w‖²`, so `p*` prunes `w` (for every `α ∈ [1, 3/2]`).
   - From `w`, `p*` is the unique strict nearest neighbor in `S_L \ {w}` and satisfies
     `α² * ‖p* - c(t_j)‖² ≤ ‖w - c(t_j)‖²`, so `p*` prunes every `c(t_j)`.
   - Hence in the **symmetrized** (undirected) RobustPrune graph `N_sym`, `w` has *only* `p*`
     as a neighbor: `N_sym(w) = {p*}`.
4. **Decoy-chain trapping for beam search of width `L`**:
   - For every `j < L`, `c(t_{j+1})` is closer to `c(t_j)` than `p*`, and `c(t_{j+1})`
     dominates `p*` in `q`-score (`⟪c(t_{j+1}), q⟫ > ⟪c(t_j), q⟫ ≥ 0 > ⟪p*, q⟫`).
   - Starting from `u = c(0)` with beam width `L`, if the beam visits all `L` decoys
     `{d_1, …, d_L}`, those `L` decoys strictly outrank both `u` and `p*`, so any
     top-`L` beam drops `p*` and has no path to `w` (recall@1 = 0).
-/

noncomputable section

open scoped RealInnerProductSpace BigOperators

namespace IndexRecall.RobustPruneFailure

abbrev Vec3 := EuclideanSpace ℝ (Fin 3)

/-- Helper to build a vector in `ℝ³` from three coordinates. -/
def vec3 (x y z : ℝ) : Vec3 :=
  WithLp.toLp 2 ![x, y, z]

@[simp] theorem vec3_apply_0 (x y z : ℝ) : vec3 x y z 0 = x := rfl
@[simp] theorem vec3_apply_1 (x y z : ℝ) : vec3 x y z 1 = y := rfl
@[simp] theorem vec3_apply_2 (x y z : ℝ) : vec3 x y z 2 = z := rfl

theorem inner_vec3 (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
    ⟪vec3 x₁ y₁ z₁, vec3 x₂ y₂ z₂⟫ = x₁ * x₂ + y₁ * y₂ + z₁ * z₂ := by
  simp only [EuclideanSpace.inner_eq_star_dotProduct, star_trivial, dotProduct,
    Fin.sum_univ_three, vec3, WithLp.ofLp_toLp, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  ring

theorem norm_sq_vec3 (x y z : ℝ) :
    ‖vec3 x y z‖ ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_vec3]
  ring

theorem dist_sq_vec3 (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
    ‖vec3 x₁ y₁ z₁ - vec3 x₂ y₂ z₂‖ ^ 2 =
      (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 + (z₁ - z₂) ^ 2 := by
  have hsub : vec3 x₁ y₁ z₁ - vec3 x₂ y₂ z₂ = vec3 (x₁ - x₂) (y₁ - y₂) (z₁ - z₂) := by
    ext i; fin_cases i <;> rfl
  rw [hsub, norm_sq_vec3]

/-- Rational unit-sphere curve `c(t)` with `c(0) = (1, 0, 0)`. -/
def curvePt (t : ℝ) : Vec3 :=
  vec3 ((1 - 37 * t ^ 2) / (1 + 37 * t ^ 2))
       (2 * t / (1 + 37 * t ^ 2))
       (12 * t / (1 + 37 * t ^ 2))

/-- Blocker point `p* = (16/29, 24/29, -3/29)` on `S²`. -/
def pStar : Vec3 := vec3 (16 / 29) (24 / 29) (-3 / 29)

/-- Winner point `w = (-2/7, 6/7, 3/7)` on `S²`. -/
def winner : Vec3 := vec3 (-2 / 7) (6 / 7) (3 / 7)

/-- Orthogonal unit query `q = (0, 0, 1)`. -/
def query : Vec3 := vec3 0 0 1

/-- Parameter `t_j = j / (500 * L)` for `j ∈ Fin (L + 1)`. -/
def param (L : ℕ) (j : ℕ) : ℝ := (j : ℝ) / (500 * (L : ℝ))

private theorem denom_pos (t : ℝ) : 0 < 1 + 37 * t ^ 2 := by positivity

private theorem denom_ne_zero (t : ℝ) : 1 + 37 * t ^ 2 ≠ 0 := (denom_pos t).ne'

/-! ### Unit-sphere identities -/

theorem norm_curvePt (t : ℝ) : ‖curvePt t‖ = 1 := by
  have hsq : ‖curvePt t‖ ^ 2 = 1 := by
    rw [curvePt, norm_sq_vec3]
    have hd := denom_ne_zero t
    field_simp [hd]
    ring
  nlinarith [norm_nonneg (curvePt t)]

theorem norm_pStar : ‖pStar‖ = 1 := by
  have hsq : ‖pStar‖ ^ 2 = 1 := by rw [pStar, norm_sq_vec3]; norm_num
  nlinarith [norm_nonneg pStar]

theorem norm_winner : ‖winner‖ = 1 := by
  have hsq : ‖winner‖ ^ 2 = 1 := by rw [winner, norm_sq_vec3]; norm_num
  nlinarith [norm_nonneg winner]

theorem norm_query : ‖query‖ = 1 := by
  have hsq : ‖query‖ ^ 2 = 1 := by rw [query, norm_sq_vec3]; norm_num
  nlinarith [norm_nonneg query]

/-! ### Exact rational distance-squared identities -/

theorem dist_sq_curve_pStar (t : ℝ) :
    ‖curvePt t - pStar‖ ^ 2 =
      2 * (13 - 12 * t + 1665 * t ^ 2) / (29 * (1 + 37 * t ^ 2)) := by
  rw [curvePt, pStar, dist_sq_vec3]
  have hd := denom_ne_zero t
  field_simp [hd]
  ring

theorem dist_sq_curve_winner (t : ℝ) :
    ‖curvePt t - winner‖ ^ 2 =
      2 * (9 - 48 * t + 185 * t ^ 2) / (7 * (1 + 37 * t ^ 2)) := by
  rw [curvePt, winner, dist_sq_vec3]
  have hd := denom_ne_zero t
  field_simp [hd]
  ring

theorem dist_sq_pStar_winner :
    ‖pStar - winner‖ ^ 2 = 200 / 203 := by
  rw [pStar, winner, dist_sq_vec3]
  norm_num

theorem dist_sq_curve_curve (a b : ℝ) :
    ‖curvePt a - curvePt b‖ ^ 2 =
      148 * (b - a) ^ 2 / ((1 + 37 * a ^ 2) * (1 + 37 * b ^ 2)) := by
  rw [curvePt, curvePt, dist_sq_vec3]
  have hda := denom_ne_zero a
  have hdb := denom_ne_zero b
  field_simp [hda, hdb]
  ring

/-! ### Inner products with query `q = (0, 0, 1)` and cosine margin -/

theorem inner_curvePt_query (t : ℝ) :
    ⟪curvePt t, query⟫ = 12 * t / (1 + 37 * t ^ 2) := by
  rw [curvePt, query, inner_vec3]; ring

theorem inner_pStar_query : ⟪pStar, query⟫ = -3 / 29 := by
  rw [pStar, query, inner_vec3]; norm_num

theorem inner_winner_query : ⟪winner, query⟫ = 3 / 7 := by
  rw [winner, query, inner_vec3]; norm_num

theorem param_bounds {L j : ℕ} (hL : 1 ≤ L) (hj : j ≤ L) :
    0 ≤ param L j ∧ param L j ≤ 1 / 500 := by
  unfold param
  have hLpos : (0 : ℝ) < L := Nat.cast_pos.mpr (by omega)
  have hjle : (j : ℝ) ≤ L := Nat.cast_le.mpr hj
  constructor
  · positivity
  · rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    linarith

theorem param_strictMono {L j₁ j₂ : ℕ} (hL : 1 ≤ L) (h : j₁ < j₂) :
    param L j₁ < param L j₂ := by
  unfold param
  have hLpos : (0 : ℝ) < L := Nat.cast_pos.mpr (by omega)
  exact div_lt_div_of_pos_right (Nat.cast_lt.mpr h) (by positivity)

/-- Score along `c(t)` is strictly increasing on `[0, 1/500]`. -/
theorem inner_curvePt_query_strictMono {a b : ℝ} (ha0 : 0 ≤ a) (hab : a < b) (hb : b ≤ 1 / 500) :
    ⟪curvePt a, query⟫ < ⟪curvePt b, query⟫ := by
  rw [inner_curvePt_query, inner_curvePt_query]
  have hda := denom_pos a
  have hdb := denom_pos b
  rw [div_lt_div_iff₀ hda hdb]
  have hdiff : 12 * b * (1 + 37 * a ^ 2) - 12 * a * (1 + 37 * b ^ 2) =
      12 * (b - a) * (1 - 37 * a * b) := by ring
  have hprod : 37 * a * b < 1 := by nlinarith
  nlinarith

/-- Winner beats every curve point on `[0, 1/500]` by strictly more than `2/5`. -/
theorem winner_margin_curve {t : ℝ} (ht : t ≤ 1 / 500) :
    2 / 5 < ⟪winner, query⟫ - ⟪curvePt t, query⟫ := by
  rw [inner_winner_query, inner_curvePt_query]
  have hd := denom_pos t
  have hle : 12 * t / (1 + 37 * t ^ 2) ≤ 12 / 500 := by
    rw [div_le_div_iff₀ hd (by norm_num)]
    nlinarith
  linarith

/-- Winner beats `p*` by strictly more than `2/5`. -/
theorem winner_margin_pStar : 2 / 5 < ⟪winner, query⟫ - ⟪pStar, query⟫ := by
  rw [inner_winner_query, inner_pStar_query]; norm_num

/-! ### Key pruning & ordering inequalities on `t ∈ [0, 1/500]` -/

/-- `p*` is strictly closer to `c(t)` than `w` is, for all `t ∈ [0, 1/500]`. -/
theorem dist_curve_pStar_lt_winner {t : ℝ} (ht0 : 0 ≤ t) (ht : t ≤ 1 / 500) :
    ‖curvePt t - pStar‖ ^ 2 < ‖curvePt t - winner‖ ^ 2 := by
  rw [dist_sq_curve_pStar, dist_sq_curve_winner]
  have hd := denom_pos t
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have hfac : 2 * (9 - 48 * t + 185 * t ^ 2) * (29 * (1 + 37 * t ^ 2)) -
      2 * (13 - 12 * t + 1665 * t ^ 2) * (7 * (1 + 37 * t ^ 2)) =
      2 * (1 + 37 * t ^ 2) * (170 - 1308 * t - 6290 * t ^ 2) := by ring
  have hmid : 0 < 170 - 1308 * t - 6290 * t ^ 2 := by nlinarith
  nlinarith

/-- Forward RobustPrune dominance: for all `α ∈ [1, 3/2]` and `t ∈ [0, 1/500]`,
`α² * ‖p* - w‖² ≤ ‖c(t) - w‖²`. Hence `p*` prunes `w` from `c(t)`! -/
theorem pStar_prunes_winner_from_curve {t α : ℝ} (ht0 : 0 ≤ t) (ht : t ≤ 1 / 500)
    (hα0 : 0 ≤ α) (hα : α ≤ 3 / 2) :
    α ^ 2 * ‖pStar - winner‖ ^ 2 ≤ ‖curvePt t - winner‖ ^ 2 := by
  rw [dist_sq_pStar_winner, dist_sq_curve_winner]
  have hd := denom_pos t
  have hαsq : α ^ 2 ≤ (3 / 2 : ℝ) ^ 2 := by nlinarith
  have h1 : α ^ 2 * (200 / 203 : ℝ) ≤ (450 / 203 : ℝ) := by nlinarith
  have h2 : (450 / 203 : ℝ) ≤ 2 * (9 - 48 * t + 185 * t ^ 2) / (7 * (1 + 37 * t ^ 2)) := by
    rw [div_le_div_iff₀ (by norm_num) (by positivity)]
    nlinarith
  linarith

/-- Reverse nearest-neighbor property: from `w`, `p*` is strictly closer than every `c(t)`. -/
theorem dist_winner_pStar_lt_curve {t : ℝ} (ht : t ≤ 1 / 500) :
    ‖winner - pStar‖ ^ 2 < ‖winner - curvePt t‖ ^ 2 := by
  rw [norm_sub_rev winner pStar, norm_sub_rev winner (curvePt t),
    dist_sq_pStar_winner, dist_sq_curve_winner]
  have hd := denom_pos t
  rw [div_lt_div_iff₀ (by norm_num) (by positivity)]
  nlinarith

/-- Reverse RobustPrune dominance: for all `α ∈ [1, 3/2]` and `t ∈ [0, 1/500]`,
`α² * ‖p* - c(t)‖² ≤ ‖w - c(t)‖²`. Hence `p*` prunes every `c(t)` from `w`! -/
theorem pStar_prunes_curve_from_winner {t α : ℝ} (ht0 : 0 ≤ t) (ht : t ≤ 1 / 500)
    (hα0 : 0 ≤ α) (hα : α ≤ 3 / 2) :
    α ^ 2 * ‖pStar - curvePt t‖ ^ 2 ≤ ‖winner - curvePt t‖ ^ 2 := by
  rw [norm_sub_rev pStar (curvePt t), norm_sub_rev winner (curvePt t),
    dist_sq_curve_pStar, dist_sq_curve_winner]
  have hd := denom_pos t
  have hαsq : α ^ 2 ≤ 9 / 4 := by nlinarith
  have hnonneg : 0 ≤ 2 * (13 - 12 * t + 1665 * t ^ 2) / (29 * (1 + 37 * t ^ 2)) := by
    apply div_nonneg _ (by positivity)
    nlinarith
  have h1 : α ^ 2 * (2 * (13 - 12 * t + 1665 * t ^ 2) / (29 * (1 + 37 * t ^ 2))) ≤
      (9 / 4) * (2 * (13 - 12 * t + 1665 * t ^ 2) / (29 * (1 + 37 * t ^ 2))) :=
    mul_le_mul_of_nonneg_right hαsq hnonneg
  have h2 : (9 / 4) * (2 * (13 - 12 * t + 1665 * t ^ 2) / (29 * (1 + 37 * t ^ 2))) ≤
      2 * (9 - 48 * t + 185 * t ^ 2) / (7 * (1 + 37 * t ^ 2)) := by
    rw [mul_div_assoc', div_le_div_iff₀ (by positivity) (by positivity)]
    have hfac : 2 * (9 - 48 * t + 185 * t ^ 2) * (29 * (1 + 37 * t ^ 2)) -
        9 / 4 * (2 * (13 - 12 * t + 1665 * t ^ 2)) * (7 * (1 + 37 * t ^ 2)) =
        (1 + 37 * t ^ 2) * (225 / 2 - 2406 * t - 83435 / 2 * t ^ 2) := by ring
    have hmid : 0 ≤ 225 / 2 - 2406 * t - 83435 / 2 * t ^ 2 := by nlinarith
    nlinarith
  linarith

/-- Any two curve points `c(a), c(b)` on `[0, 1/500]` are much closer to each other than to `p*`. -/
theorem dist_curve_curve_lt_pStar {a b : ℝ} (ha0 : 0 ≤ a) (ha : a ≤ 1 / 500)
    (hb0 : 0 ≤ b) (hb : b ≤ 1 / 500) :
    ‖curvePt a - curvePt b‖ ^ 2 < ‖curvePt a - pStar‖ ^ 2 := by
  rw [dist_sq_curve_curve, dist_sq_curve_pStar]
  have hda := denom_pos a
  have hdb := denom_pos b
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have hba : (b - a) ^ 2 ≤ (1 / 500 : ℝ) ^ 2 := by nlinarith
  have hdb1 : 1 ≤ 1 + 37 * b ^ 2 := by nlinarith
  have hnum : 12 ≤ 13 - 12 * a + 1665 * a ^ 2 := by nlinarith
  have hleft : 148 * (b - a) ^ 2 * 29 ≤ 148 * (1 / 500 : ℝ) ^ 2 * 29 := by nlinarith
  have hright : 24 ≤ 2 * (13 - 12 * a + 1665 * a ^ 2) * (1 + 37 * b ^ 2) := by nlinarith
  nlinarith

/-- Curve points cannot prune `p*` from `c(a)` for `α ∈ [1, 3/2]`, wait: we need the exact
reverse! Can any curve point `c(b)` prune `w`? Actually, even better: for `a < b`, we check
whether `c(b)` can prune `p*` from `c(a)` — we want `p*` to SURVIVE pruning from `c(a)`
OR be pruned by some `c(b)` which ALSO prunes `w`!
Wait: if `c(b)` prunes `p*` from `c(a)`, then `‖c(b) - p*‖ ≤ ‖c(a) - p*‖ / α`, and we prove
the **ratio dominance** lemma below: `c(b)` ALSO prunes `w` from `c(a)`! -/
theorem ratio_dominance {a b : ℝ} (ha0 : 0 ≤ a) (hab : a < b) (hb : b ≤ 1 / 500) :
    ‖curvePt b - winner‖ ^ 2 * ‖curvePt a - pStar‖ ^ 2 <
      ‖curvePt a - winner‖ ^ 2 * ‖curvePt b - pStar‖ ^ 2 := by
  rw [dist_sq_curve_winner a, dist_sq_curve_pStar a,
      dist_sq_curve_winner b, dist_sq_curve_pStar b]
  have hda := denom_pos a
  have hdb := denom_pos b
  have h1 : (2 * (9 - 48 * a + 185 * a ^ 2) / (7 * (1 + 37 * a ^ 2))) *
      (2 * (13 - 12 * b + 1665 * b ^ 2) / (29 * (1 + 37 * b ^ 2))) -
      (2 * (9 - 48 * b + 185 * b ^ 2) / (7 * (1 + 37 * b ^ 2))) *
      (2 * (13 - 12 * a + 1665 * a ^ 2) / (29 * (1 + 37 * a ^ 2))) =
      4 * (b - a) * (516 + 12580 * (a + b) - 77700 * a * b) /
        (203 * (1 + 37 * a ^ 2) * (1 + 37 * b ^ 2)) := by
    field_simp [hda.ne', hdb.ne']
    ring
  have hpos : 0 < 4 * (b - a) * (516 + 12580 * (a + b) - 77700 * a * b) /
      (203 * (1 + 37 * a ^ 2) * (1 + 37 * b ^ 2)) := by
    apply div_pos _ (by positivity)
    have hba : 0 < b - a := by linarith
    have hab_small : 77700 * a * b < 1 := by nlinarith
    nlinarith
  linarith

/-- For `a < b` on `[0, 1/500]`, `‖c(b) - p*‖ > ‖c(a) - p*‖`? Wait: check the sign!
At `t = 0`: `13 / 29 ≈ 0.448276`. Derivative at `0` is `-12 < 0`, so `‖c(b) - p*‖` initially
DECREASES slightly as `b` increases from `0` to `1/500`!
Specifically, for `b < a`: `‖c(a) - p*‖ < ‖c(b) - p*‖`? Let's verify exact monotonicity
of `‖c(t) - p*‖²` on `[0, 1/500]`:
`(13 - 12t + 1665t²)/(1 + 37t²)` has derivative `(-12 + 3330t)(1+37t²) - 74t(13-12t+1665t²) < 0`
on `[0, 1/500]`. So `‖c(b) - p*‖ < ‖c(a) - p*‖` for `a < b`!
And when `a > b`, `‖c(b) - p*‖ > ‖c(a) - p*‖`, so `c(b)` CANNOT prune `p*` from `c(a)` for any `α ≥ 1`! -/
theorem dist_curve_pStar_strictAnti {a b : ℝ} (ha0 : 0 ≤ a) (hab : a < b) (hb : b ≤ 1 / 500) :
    ‖curvePt b - pStar‖ ^ 2 < ‖curvePt a - pStar‖ ^ 2 := by
  rw [dist_sq_curve_pStar, dist_sq_curve_pStar]
  have hda := denom_pos a
  have hdb := denom_pos b
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have hdiff : 2 * (13 - 12 * a + 1665 * a ^ 2) * (29 * (1 + 37 * b ^ 2)) -
      2 * (13 - 12 * b + 1665 * b ^ 2) * (29 * (1 + 37 * a ^ 2)) =
      58 * (b - a) * (12 - 1184 * (a + b) - 444 * a * b) := by ring
  have hba : 0 < b - a := by linarith
  have hmid : 0 < 12 - 1184 * (a + b) - 444 * a * b := by nlinarith
  nlinarith

/-! ### Formal Exact RobustPrune Model (Infinite Degree `R = ∞`, Full Candidate Set) -/

/-- Exact sequential DiskANN `RobustPrune(u, V, α)` with unbounded degree `R = ∞`.
Given candidates ordered by increasing Euclidean distance to `u`, `robustPruneLoop` scans
candidates and keeps `p` iff no already-kept neighbor `p'` satisfies `α * ‖p' - p‖ ≤ ‖u - p‖`. -/
def robustPruneLoop (u : Vec3) (α : ℝ) : List Vec3 → List Vec3 → List Vec3
  | [], kept => kept
  | p :: rest, kept =>
      if ∃ p' ∈ kept, α ^ 2 * ‖p' - p‖ ^ 2 ≤ ‖u - p‖ ^ 2 then
        robustPruneLoop u α rest kept
      else
        robustPruneLoop u α rest (kept ++ [p])

/-- Monotonicity: once an element is in `kept`, it remains in `robustPruneLoop u α rest kept`. -/
theorem mem_robustPruneLoop_of_mem_kept {u : Vec3} {α : ℝ} {rest kept : List Vec3} {x : Vec3}
    (hx : x ∈ kept) : x ∈ robustPruneLoop u α rest kept := by
  induction rest generalizing kept with
  | nil => exact hx
  | cons p ps ih =>
    unfold robustPruneLoop
    split_ifs
    · exact ih hx
    · exact ih (List.mem_append_left _ hx)

/-- If some already-kept element `p' ∈ kept` prunes `x`, then `x` cannot be newly added
by `robustPruneLoop u α rest kept` (so if `x ∉ kept`, then `x ∉ robustPruneLoop u α rest kept`). -/
theorem not_mem_robustPruneLoop_of_pruned {u : Vec3} {α : ℝ} {rest kept : List Vec3} {x : Vec3}
    (hx_not : x ∉ kept)
    (hpruned : ∃ p' ∈ kept, α ^ 2 * ‖p' - x‖ ^ 2 ≤ ‖u - x‖ ^ 2) :
    x ∉ robustPruneLoop u α rest kept := by
  induction rest generalizing kept with
  | nil => exact hx_not
  | cons p ps ih =>
    unfold robustPruneLoop
    split_ifs with h
    · exact ih hx_not hpruned
    · have hne : p ≠ x := by
        rintro rfl
        exact h hpruned
      apply ih
      · simp [hx_not, Ne.symm hne]
      · rcases hpruned with ⟨p', hp', hle⟩
        exact ⟨p', List.mem_append_left _ hp', hle⟩

/-- Every element in `robustPruneLoop u α rest kept` is in `kept` or `rest`. -/
theorem mem_robustPruneLoop_subset {u : Vec3} {α : ℝ} {rest kept : List Vec3} {x : Vec3}
    (hx : x ∈ robustPruneLoop u α rest kept) : x ∈ kept ∨ x ∈ rest := by
  induction rest generalizing kept with
  | nil => exact Or.inl hx
  | cons p ps ih =>
    unfold robustPruneLoop at hx
    split_ifs at hx with h
    · rcases ih hx with h1 | h2
      · exact Or.inl h1
      · exact Or.inr (List.mem_cons_of_mem _ h2)
    · rcases ih hx with h1 | h2
      · rcases List.mem_append.mp h1 with h1a | h1b
        · exact Or.inl h1a
        · simp only [List.mem_singleton] at h1b
          exact Or.inr (h1b ▸ List.mem_cons_self)
      · exact Or.inr (List.mem_cons_of_mem _ h2)

/-- Key invariant of `robustPruneLoop`: if `q_el` appears in `rest` (not yet in `kept`), then
at the moment `q_el` is processed, either `q_el` gets added to `kept` OR some earlier element
already in `kept` pruned `q_el`! Equivalently, in the final output `out = robustPruneLoop u α rest kept`,
either `q_el ∈ out` or `∃ p' ∈ out, α² * ‖p' - q_el‖² ≤ ‖u - q_el‖²`. -/
theorem robustPruneLoop_mem_or_pruned {u : Vec3} {α : ℝ} {rest kept : List Vec3} {x : Vec3}
    (hx : x ∈ rest) :
    x ∈ robustPruneLoop u α rest kept ∨
      ∃ p' ∈ robustPruneLoop u α rest kept, α ^ 2 * ‖p' - x‖ ^ 2 ≤ ‖u - x‖ ^ 2 := by
  induction rest generalizing kept with
  | nil => simp at hx
  | cons p ps ih =>
    unfold robustPruneLoop
    rcases List.mem_cons.mp hx with rfl | hx_ps
    · split_ifs with h
      · rcases h with ⟨p', hp', hle⟩
        exact Or.inr ⟨p', mem_robustPruneLoop_of_mem_kept hp', hle⟩
      · exact Or.inl (mem_robustPruneLoop_of_mem_kept (by simp))
    · split_ifs with h
      · exact ih hx_ps
      · exact ih hx_ps

/-! ### Main Theorem A1a: Exact Bidirectional Isolation of `w` under RobustPrune -/

/-- The dataset `S_L` consists of `{c(t_j) : 0 ≤ j ≤ L} ∪ {p*, w}`. -/
def isCurvePt (L : ℕ) (v : Vec3) : Prop :=
  ∃ j ≤ L, v = curvePt (param L j)

def inDataset (L : ℕ) (v : Vec3) : Prop :=
  isCurvePt L v ∨ v = pStar ∨ v = winner

/-- All curve points are distinct from `p*` and `w`, and `p* ≠ w`. -/
theorem curvePt_ne_pStar {t : ℝ} (ht0 : 0 ≤ t) :
    curvePt t ≠ pStar := by
  intro heq
  have h1 : ⟪curvePt t, query⟫ = ⟪pStar, query⟫ := congrArg (fun v => ⟪v, query⟫) heq
  rw [inner_curvePt_query, inner_pStar_query] at h1
  have hd := denom_pos t
  have hnonneg : 0 ≤ 12 * t / (1 + 37 * t ^ 2) := by positivity
  linarith

theorem curvePt_ne_winner {t : ℝ} (ht : t ≤ 1 / 500) :
    curvePt t ≠ winner := by
  intro heq
  have h1 := winner_margin_curve ht
  rw [heq] at h1
  linarith

theorem pStar_ne_winner : pStar ≠ winner := by
  intro heq
  have h1 := winner_margin_pStar
  rw [heq] at h1
  linarith

/-- **Theorem A1 (Forward Pruning):**
From any curve vertex `u = c(t_j)` (`0 ≤ j ≤ L`), let `cand` be *any* candidate list
of points from `S_L \ {u}` ordered by non-decreasing distance to `u` (`Pairwise (fun x y => ‖u - x‖² ≤ ‖u - y‖²)`)
that contains `p*`. Then for *every* `α ∈ [1, 3/2]`, `winner ∉ robustPruneLoop u α cand []`. -/
theorem robustPrune_from_curve_drops_winner
    {L j : ℕ} (hL : 1 ≤ L) (hj : j ≤ L) {α : ℝ} (hα1 : 1 ≤ α) (hα2 : α ≤ 3 / 2)
    (cand : List Vec3)
    (hcand_in : ∀ v ∈ cand, inDataset L v)
    (hcand_ne : curvePt (param L j) ∉ cand)
    (hcand_sorted : cand.Pairwise (fun x y => ‖curvePt (param L j) - x‖ ^ 2 ≤ ‖curvePt (param L j) - y‖ ^ 2))
    (hpStar_in : pStar ∈ cand) :
    winner ∉ robustPruneLoop (curvePt (param L j)) α cand [] := by
  set u := curvePt (param L j)
  set a := param L j
  have ⟨ha0, ha⟩ := param_bounds hL hj
  -- Split cand at pStar: cand = pre ++ pStar :: post.
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hpStar_in
  -- Notice winner cannot appear in pre because ‖u - p*‖² < ‖u - w‖² and cand is sorted!
  have hw_not_pre : winner ∉ pre := by
    intro hw_pre
    rw [hsplit] at hcand_sorted
    have hpw := (List.pairwise_append.mp hcand_sorted).2.2 winner hw_pre pStar List.mem_cons_self
    linarith [dist_curve_pStar_lt_winner ha0 ha]
  -- Evaluate robustPruneLoop u α (pre ++ pStar :: post) [] step by step.
  have hloop_append : ∀ l₁ l₂ kept,
      robustPruneLoop u α (l₁ ++ l₂) kept =
        robustPruneLoop u α l₂ (robustPruneLoop u α l₁ kept) := by
    intro l₁ l₂
    induction l₁ with
    | nil => intro kept; rfl
    | cons x xs ih =>
      intro kept
      simp only [List.cons_append, robustPruneLoop]
      split_ifs <;> exact ih _
  rw [hsplit, hloop_append]
  set kept₁ := robustPruneLoop u α pre []
  -- winner ∉ kept₁ because kept₁ ⊆ pre and winner ∉ pre.
  have hw_not_kept₁ : winner ∉ kept₁ := by
    intro hmem
    rcases mem_robustPruneLoop_subset hmem with h1 | h2
    · simp at h1
    · exact hw_not_pre h2
  -- Now process pStar :: post starting from kept₁.
  unfold robustPruneLoop
  split_ifs with hpruned_pStar
  · -- Case 1: Some p' ∈ kept₁ pruned pStar: α² * ‖p' - p*‖² ≤ ‖u - p*‖².
    -- Since p' ∈ kept₁ ⊆ pre, p' ∈ cand, p' ≠ winner, and p' ≠ pStar (wait: could p' = pStar?
    -- If p' = pStar, then pStar ∈ kept₁ already prunes winner directly!).
    rcases hpruned_pStar with ⟨p', hp'_kept, hle_p'⟩
    have hp'_pre : p' ∈ pre := by
      rcases mem_robustPruneLoop_subset hp'_kept with h1 | h2
      · simp at h1
      · exact h2
    have hp'_cand : p' ∈ cand := by rw [hsplit]; exact List.mem_append_left _ hp'_pre
    -- Prove α² * ‖p' - w‖² ≤ ‖u - w‖², so p' ∈ kept₁ directly prunes winner!
    have hp'_prunes_w : α ^ 2 * ‖p' - winner‖ ^ 2 ≤ ‖u - winner‖ ^ 2 := by
      rcases hcand_in p' hp'_cand with ⟨k, hkL, rfl⟩ | rfl | rfl
      · -- p' = curvePt b for b = param L k.
        set b := param L k
        have ⟨hb0, hb⟩ := param_bounds hL hkL
        -- Must have a < b? Check the 3 cases for a vs b:
        rcases lt_trichotomy a b with hab | hab_eq | hba
        · -- Case a < b: ratio_dominance gives ‖c(b) - w‖² * ‖c(a) - p*‖² < ‖c(a) - w‖² * ‖c(b) - p*‖²
          have hrat := ratio_dominance ha0 hab hb
          have hap_pos : 0 < ‖curvePt a - pStar‖ ^ 2 := by
            rw [dist_sq_curve_pStar]
            have hd := denom_pos a
            apply div_pos _ (by positivity)
            nlinarith
          have hαsq_pos : 0 < α ^ 2 := by nlinarith
          have hstep1 : α ^ 2 * (‖curvePt b - winner‖ ^ 2 * ‖curvePt a - pStar‖ ^ 2) <
              α ^ 2 * (‖curvePt a - winner‖ ^ 2 * ‖curvePt b - pStar‖ ^ 2) :=
            mul_lt_mul_of_pos_left hrat hαsq_pos
          have haw_nonneg : 0 ≤ ‖curvePt a - winner‖ ^ 2 := sq_nonneg _
          have hstep2 : α ^ 2 * ‖curvePt b - pStar‖ ^ 2 * ‖curvePt a - winner‖ ^ 2 ≤
              ‖curvePt a - pStar‖ ^ 2 * ‖curvePt a - winner‖ ^ 2 :=
            mul_le_mul_of_nonneg_right hle_p' haw_nonneg
          have hcomb : (α ^ 2 * ‖curvePt b - winner‖ ^ 2) * ‖curvePt a - pStar‖ ^ 2 <
              ‖curvePt a - winner‖ ^ 2 * ‖curvePt a - pStar‖ ^ 2 := by linarith
          exact le_of_lt (lt_of_mul_lt_mul_right hcomb hap_pos.le)
        · -- Case a = b: impossible since curvePt b = curvePt a = u ∉ cand!
          rw [← hab_eq] at hp'_cand
          exact absurd hp'_cand hcand_ne
        · -- Case b < a: impossible because ‖c(b) - p*‖² > ‖c(a) - p*‖² and α ≥ 1!
          have hanti := dist_curve_pStar_strictAnti hb0 hba ha
          have hbp_nonneg : 0 ≤ ‖curvePt b - pStar‖ ^ 2 := sq_nonneg _
          have hαsq : 1 ≤ α ^ 2 := by nlinarith
          nlinarith
      · -- p' = pStar: directly prunes winner by pStar_prunes_winner_from_curve!
        exact pStar_prunes_winner_from_curve ha0 ha (by linarith) hα2
      · -- p' = winner: impossible since winner ∉ pre!
        exact absurd hp'_pre hw_not_pre
    exact not_mem_robustPruneLoop_of_pruned hw_not_kept₁ ⟨p', hp'_kept, hp'_prunes_w⟩
  · -- Case 2: pStar is NOT pruned, so pStar is added to kept₁!
    have hw_not_kept₂ : winner ∉ kept₁ ++ [pStar] := by
      simp [hw_not_kept₁, pStar_ne_winner.symm]
    exact not_mem_robustPruneLoop_of_pruned hw_not_kept₂
      ⟨pStar, by simp, pStar_prunes_winner_from_curve ha0 ha (by linarith) hα2⟩

/-- **Theorem A1 (Reverse Pruning from `w`):**
From `winner`, let `cand` be *any* candidate list of points from `S_L \ {winner}` ordered by
non-decreasing distance to `winner` that contains `p*`.
Then for *every* `α ∈ [1, 3/2]`, every element of `robustPruneLoop winner α cand []` equals `p*`!
(In particular, `winner` connects *only* to `p*` and to *no* curve point `c(t_j)`.) -/
theorem robustPrune_from_winner_only_pStar
    {L : ℕ} (hL : 1 ≤ L) {α : ℝ} (hα1 : 1 ≤ α) (hα2 : α ≤ 3 / 2)
    (cand : List Vec3)
    (hcand_in : ∀ v ∈ cand, inDataset L v)
    (hcand_ne : winner ∉ cand)
    (hcand_sorted : cand.Pairwise (fun x y => ‖winner - x‖ ^ 2 ≤ ‖winner - y‖ ^ 2))
    (hpStar_in : pStar ∈ cand) :
    ∀ v ∈ robustPruneLoop winner α cand [], v = pStar := by
  -- Every element v ∈ cand other than p* has ‖w - p*‖² < ‖w - v‖².
  -- Thus the prefix before p* can only contain copies of p* (or be empty), so p* is at the head!
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hpStar_in
  have hpre_eq : ∀ v ∈ pre, v = pStar := by
    intro v hv
    have hv_cand : v ∈ cand := by rw [hsplit]; exact List.mem_append_left _ hv
    rcases hcand_in v hv_cand with ⟨k, hkL, rfl⟩ | rfl | rfl
    · exfalso
      rw [hsplit] at hcand_sorted
      have hle := (List.pairwise_append.mp hcand_sorted).2.2 (curvePt (param L k)) hv pStar List.mem_cons_self
      have ⟨hk0, hk⟩ := param_bounds hL hkL
      linarith [dist_winner_pStar_lt_curve hk]
    · rfl
    · exact absurd hv_cand hcand_ne
  -- Show by induction that if all remaining elements are in S_L \ {winner}, and pStar ∈ kept,
  -- then no curve point can be added!
  have hno_curve_added : ∀ rest kept, (∀ v ∈ kept, v = pStar) → pStar ∈ kept →
      (∀ v ∈ rest, inDataset L v ∧ v ≠ winner) →
      ∀ v ∈ robustPruneLoop winner α rest kept, v = pStar := by
    intro rest
    induction rest with
    | nil =>
      intro kept hkept _ _ v hv
      exact hkept v hv
    | cons x xs ih =>
      intro kept hkept hp_kept hrest v hv
      unfold robustPruneLoop at hv
      split_ifs at hv with h
      · exact ih kept hkept hp_kept (fun y hy => hrest y (List.mem_cons_of_mem _ hy)) v hv
      · have hx_eq : x = pStar := by
          rcases (hrest x List.mem_cons_self).1 with ⟨k, hkL, rfl⟩ | rfl | rfl
          · exfalso
            have ⟨hk0, hk⟩ := param_bounds hL hkL
            exact h ⟨pStar, hp_kept, pStar_prunes_curve_from_winner hk0 hk (by linarith) hα2⟩
          · rfl
          · exact absurd rfl (hrest winner List.mem_cons_self).2
        apply ih (kept ++ [x])
        · intro y hy
          rcases List.mem_append.mp hy with hy1 | hy2
          · exact hkept y hy1
          · simp only [List.mem_singleton] at hy2; exact hy2.trans hx_eq
        · exact List.mem_append_left _ hp_kept
        · exact fun y hy => hrest y (List.mem_cons_of_mem _ hy)
        · exact hv
  have hloop_append : ∀ l₁ l₂ kept,
      robustPruneLoop winner α (l₁ ++ l₂) kept =
        robustPruneLoop winner α l₂ (robustPruneLoop winner α l₁ kept) := by
    intro l₁ l₂
    induction l₁ with
    | nil => intro kept; rfl
    | cons x xs ih =>
      intro kept
      simp only [List.cons_append, robustPruneLoop]
      split_ifs <;> exact ih _
  rw [hsplit, hloop_append]
  set kept₁ := robustPruneLoop winner α pre []
  have hkept₁_eq : ∀ v ∈ kept₁, v = pStar := by
    intro v hv
    rcases mem_robustPruneLoop_subset hv with h1 | h2
    · simp at h1
    · exact hpre_eq v h2
  have hpost_in : ∀ v ∈ post, inDataset L v ∧ v ≠ winner := by
    intro v hv
    have hv_cand : v ∈ cand := by rw [hsplit]; exact List.mem_append_right _ (List.mem_cons_of_mem _ hv)
    exact ⟨hcand_in v hv_cand, fun h => hcand_ne (h ▸ hv_cand)⟩
  unfold robustPruneLoop
  split_ifs with h
  · -- Some p' ∈ kept₁ pruned pStar; since kept₁ only contains pStar, pStar ∈ kept₁!
    rcases h with ⟨p', hp', _⟩
    rw [hkept₁_eq p' hp'] at hp'
    exact hno_curve_added post kept₁ hkept₁_eq hp' hpost_in
  · -- pStar is appended to kept₁.
    apply hno_curve_added post (kept₁ ++ [pStar])
    · intro y hy
      rcases List.mem_append.mp hy with hy1 | hy2
      · exact hkept₁_eq y hy1
      · simp only [List.mem_singleton] at hy2; exact hy2
    · simp
    · exact hpost_in

/-- From `winner`, `p*` is the first candidate and is always kept, so it is in
`robustPruneLoop winner α cand []`. -/
theorem robustPrune_from_winner_keeps_pStar
    {L : ℕ} (hL : 1 ≤ L) {α : ℝ}
    (cand : List Vec3)
    (hcand_in : ∀ v ∈ cand, inDataset L v)
    (hcand_ne : winner ∉ cand)
    (hcand_sorted : cand.Pairwise (fun x y => ‖winner - x‖ ^ 2 ≤ ‖winner - y‖ ^ 2))
    (hpStar_in : pStar ∈ cand) :
    pStar ∈ robustPruneLoop winner α cand [] := by
  cases cand with
  | nil => simp at hpStar_in
  | cons x rest =>
    have hx_eq : x = pStar := by
      rcases List.mem_cons.mp hpStar_in with h | h
      · exact h.symm
      · rcases hcand_in x List.mem_cons_self with ⟨k, hkL, rfl⟩ | rfl | rfl
        · exfalso
          have hle := (List.pairwise_cons.mp hcand_sorted).1 pStar h
          have ⟨_, hk⟩ := param_bounds hL hkL
          linarith [dist_winner_pStar_lt_curve hk]
        · rfl
        · exact absurd List.mem_cons_self hcand_ne
    subst hx_eq
    unfold robustPruneLoop
    rw [if_neg (by simp)]
    exact mem_robustPruneLoop_of_mem_kept (by simp)

/-! ### Decoy-Chain Nearest-Neighbor Property & Beam-Search Trapping (Theorem A1b) -/

/-- Monotonicity of curve distance along `[0, 1/500]`:
for `0 ≤ a < b < c ≤ 1/500`, `‖c(a) - c(b)‖² < ‖c(a) - c(c)‖²`. -/
theorem dist_curve_curve_strictMono_right {a b c : ℝ}
    (ha0 : 0 ≤ a) (hab : a < b) (hbc : b < c) :
    ‖curvePt a - curvePt b‖ ^ 2 < ‖curvePt a - curvePt c‖ ^ 2 := by
  rw [dist_sq_curve_curve, dist_sq_curve_curve]
  have hda := denom_pos a
  have hdb := denom_pos b
  have hdc := denom_pos c
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have hfac : 148 * (c - a) ^ 2 * ((1 + 37 * a ^ 2) * (1 + 37 * b ^ 2)) -
      148 * (b - a) ^ 2 * ((1 + 37 * a ^ 2) * (1 + 37 * c ^ 2)) =
      148 * (1 + 37 * a ^ 2) * (c - b) *
        ((c + b - 2 * a) + 37 * (a * (b * (c - a) + c * (b - a)))) := by
    ring
  have hcb : 0 < c - b := by linarith
  have hgap : 0 < c + b - 2 * a := by linarith
  have hba : 0 < b - a := by linarith
  have hca : 0 < c - a := by linarith
  have hb0 : 0 ≤ b := by linarith
  have hc0 : 0 ≤ c := by linarith
  have hbound : 0 ≤ a * (b * (c - a) + c * (b - a)) := by positivity
  have hpos : 0 < 148 * (1 + 37 * a ^ 2) * (c - b) *
      ((c + b - 2 * a) + 37 * (a * (b * (c - a) + c * (b - a)))) := by
    have h1 : 0 < 148 * (1 + 37 * a ^ 2) * (c - b) := by positivity
    have h2 : 0 < (c + b - 2 * a) + 37 * (a * (b * (c - a) + c * (b - a))) := by linarith
    exact mul_pos h1 h2
  linarith

/-- Adjacent forward edge survival: from `c(t_j)` (`j < L`), the next decoy `c(t_{j+1})`
cannot be pruned by `c(t_k)` for any `k > j+1` (since `c(t_{j+1})` is closer to `c(t_j)`)
nor by any `c(t_k)` with `k < j` (since `‖c(t_k) - c(t_{j+1})‖ > ‖c(t_j) - c(t_{j+1})‖`!).
Hence in exact sorted `RobustPrune` (`α ≥ 1`), the edge `c(t_j) → c(t_{j+1})` ALWAYS survives! -/
theorem dist_curve_left_gt_adjacent {a b c : ℝ}
    (ha0 : 0 ≤ a) (hab : a < b) (hbc : b < c) :
    ‖curvePt b - curvePt c‖ ^ 2 < ‖curvePt a - curvePt c‖ ^ 2 := by
  rw [dist_sq_curve_curve, dist_sq_curve_curve]
  have hda := denom_pos a
  have hdb := denom_pos b
  have hdc := denom_pos c
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  have hfac : 148 * (c - a) ^ 2 * ((1 + 37 * b ^ 2) * (1 + 37 * c ^ 2)) -
      148 * (c - b) ^ 2 * ((1 + 37 * a ^ 2) * (1 + 37 * c ^ 2)) =
      148 * (1 + 37 * c ^ 2) * (b - a) *
        ((2 * c - a - b) + 37 * (c * (a * (c - b) + b * (c - a)))) := by
    ring
  have hba : 0 < b - a := by linarith
  have hgap : 0 < 2 * c - a - b := by linarith
  have hcb : 0 < c - b := by linarith
  have hca : 0 < c - a := by linarith
  have hb0 : 0 ≤ b := by linarith
  have hc0 : 0 ≤ c := by linarith
  have hbound : 0 ≤ c * (a * (c - b) + b * (c - a)) := by positivity
  have hpos : 0 < 148 * (1 + 37 * c ^ 2) * (b - a) *
      ((2 * c - a - b) + 37 * (c * (a * (c - b) + b * (c - a)))) := by
    have h1 : 0 < 148 * (1 + 37 * c ^ 2) * (b - a) := by positivity
    have h2 : 0 < (2 * c - a - b) + 37 * (c * (a * (c - b) + b * (c - a))) := by linarith
    exact mul_pos h1 h2
  linarith

/-- **Forward Decoy-Chain Edge Survival:**
For every `j < L` and every `α ≥ 1`, in `robustPruneLoop (c(t_j)) α cand []` over any
distance-sorted candidate list `cand ⊆ S_L \ {c(t_j)}` containing `c(t_{j+1})`,
the forward decoy edge `c(t_{j+1})` ALWAYS survives pruning! -/
theorem robustPrune_from_curve_keeps_next_decoy
    {L j : ℕ} (hL : 1 ≤ L) (hj : j < L) {α : ℝ} (hα1 : 1 ≤ α)
    (cand : List Vec3)
    (hcand_in : ∀ v ∈ cand, inDataset L v)
    (hcand_ne : curvePt (param L j) ∉ cand)
    (hcand_sorted : cand.Pairwise (fun x y => ‖curvePt (param L j) - x‖ ^ 2 ≤ ‖curvePt (param L j) - y‖ ^ 2))
    (hnext_in : curvePt (param L (j + 1)) ∈ cand) :
    curvePt (param L (j + 1)) ∈ robustPruneLoop (curvePt (param L j)) α cand [] := by
  set u := curvePt (param L j)
  set nxt := curvePt (param L (j + 1))
  have ⟨hj0, hjle⟩ := param_bounds hL (Nat.le_of_lt hj)
  have ⟨hn0, hnle⟩ := param_bounds hL (Nat.succ_le_of_lt hj)
  have hjn_lt : param L j < param L (j + 1) := param_strictMono hL (Nat.lt_succ_self j)
  -- Split cand at nxt: cand = pre ++ nxt :: post.
  obtain ⟨pre, post, hsplit⟩ := List.append_of_mem hnext_in
  have hloop_append : ∀ l₁ l₂ kept,
      robustPruneLoop u α (l₁ ++ l₂) kept =
        robustPruneLoop u α l₂ (robustPruneLoop u α l₁ kept) := by
    intro l₁ l₂
    induction l₁ with
    | nil => intro kept; rfl
    | cons x xs ih =>
      intro kept
      simp only [List.cons_append, robustPruneLoop]
      split_ifs <;> exact ih _
  rw [hsplit, hloop_append]
  set kept₁ := robustPruneLoop u α pre []
  -- Prove NO element p' ∈ kept₁ can prune nxt (unless p' = nxt itself, in which case nxt ∈ kept₁ already!).
  by_cases halready : nxt ∈ kept₁
  · exact mem_robustPruneLoop_of_mem_kept halready
  · unfold robustPruneLoop
    rw [if_neg]
    · exact mem_robustPruneLoop_of_mem_kept (by simp)
    · rintro ⟨p', hp'_kept, hprune⟩
      have hp'_pre : p' ∈ pre := by
        rcases mem_robustPruneLoop_subset hp'_kept with h1 | h2
        · simp at h1
        · exact h2
      have hp'_cand : p' ∈ cand := by rw [hsplit]; exact List.mem_append_left _ hp'_pre
      have hdist_le : ‖u - p'‖ ^ 2 ≤ ‖u - nxt‖ ^ 2 := by
        rw [hsplit] at hcand_sorted
        exact (List.pairwise_append.mp hcand_sorted).2.2 p' hp'_pre nxt List.mem_cons_self
      have hαsq : 1 ≤ α ^ 2 := by nlinarith
      rcases hcand_in p' hp'_cand with ⟨k, hkL, rfl⟩ | rfl | rfl
      · -- p' = curvePt (param L k). Compare k with j and j+1.
        have ⟨hk0, hkle⟩ := param_bounds hL hkL
        by_cases hkj : k < j
        · -- k < j < j+1: dist_curve_left_gt_adjacent gives ‖u - nxt‖² < ‖c(t_k) - nxt‖²!
          have hlt := dist_curve_left_gt_adjacent hk0 (param_strictMono hL hkj) hjn_lt
          have hnn : 0 ≤ ‖curvePt (param L k) - nxt‖ ^ 2 := sq_nonneg _
          nlinarith
        · by_cases hkeq : k = j
          · subst hkeq; exact hcand_ne hp'_cand
          · by_cases hkn : k = j + 1
            · subst hkn; exact halready hp'_kept
            · -- k > j + 1: contradicts hdist_le by dist_curve_curve_strictMono_right!
              have hgt : j + 1 < k := by omega
              have hlt := dist_curve_curve_strictMono_right hj0 hjn_lt (param_strictMono hL hgt)
              linarith
      · -- p' = pStar: impossible since ‖u - nxt‖² < ‖u - p*‖² contradicts hdist_le!
        have hlt := dist_curve_curve_lt_pStar hj0 hjle hn0 hnle
        linarith
      · -- p' = winner: impossible since ‖u - nxt‖² < ‖u - p*‖² < ‖u - w‖²!
        have hlt1 := dist_curve_curve_lt_pStar hj0 hjle hn0 hnle
        have hlt2 := dist_curve_pStar_lt_winner hj0 hjle
        linarith

/-! ### Main Packaged Theorem A1: Bidirectional Isolation + Decoy Dominance -/

/-- **Theorem A1 (Bidirectional RobustPrune Failure on `S²` for all `L ≥ 1` and `α ∈ [1, 3/2]`):**
Let `N_dir : Vec3 → List Vec3` be the exact `RobustPrune(v, S_L \ {v}, α)` out-neighborhood
for any `α ∈ [1, 3/2]` (unbounded degree `R = ∞`), and let `N_sym` be its symmetrization
(undirected graph: `y ∈ N_sym(x)` iff `y ∈ N_dir(x)` or `x ∈ N_dir(y)`). Then:
1. Every point in `S_L` and `query` has unit norm `1`.
2. `winner` leads every other point in `S_L` by cosine margin strictly `> 2/5`.
3. In the **symmetrized** graph `N_sym`, `winner` has *no* edge to `u = c(0)` or any decoy
   `d_j = c(t_j)`: its only possible neighbor in `S_L \ {winner}` is `p*`.
4. The forward decoy chain `c(t_0) → c(t_1) → ⋯ → c(t_L)` exists in `N_dir`, and all `L`
   decoys `{c(t_1), …, c(t_L)}` strictly outrank both `u = c(0)` and `p*` on `query`.
   Consequently, any width-`L` top-scoring beam that reaches `{c(t_1), …, c(t_L)}` evicts `p*`
   before ever expanding `p*` and has **zero** edges to `winner` (recall@1 = 0). -/
theorem bidirectional_robustPrune_failure
    {L : ℕ} (hL : 1 ≤ L) {α : ℝ} (hα1 : 1 ≤ α) (hα2 : α ≤ 3 / 2)
    (cand : Vec3 → List Vec3)
    (hcand_in : ∀ v, ∀ x ∈ cand v, inDataset L x)
    (hcand_ne : ∀ v, v ∉ cand v)
    (hcand_sorted : ∀ v, (cand v).Pairwise (fun x y => ‖v - x‖ ^ 2 ≤ ‖v - y‖ ^ 2))
    (hcand_complete : ∀ v, inDataset L v → ∀ x, inDataset L x → x ≠ v → x ∈ cand v) :
    let N_dir : Vec3 → List Vec3 := fun v => robustPruneLoop v α (cand v) []
    -- (1) All points in S_L and query are unit vectors
    (∀ x, inDataset L x → ‖x‖ = 1) ∧ ‖query‖ = 1 ∧
    -- (2) Winner beats every other point in S_L by cosine margin > 2/5
    (∀ x, inDataset L x → x ≠ winner → 2 / 5 < ⟪winner, query⟫ - ⟪x, query⟫) ∧
    -- (3) Symmetrized isolation: no curve point c(t_j) is adjacent to winner in either direction
    (∀ j ≤ L, winner ∉ N_dir (curvePt (param L j)) ∧ curvePt (param L j) ∉ N_dir winner) ∧
    -- (4) Forward decoy chain exists and every decoy d_j (1 ≤ j ≤ L) strictly beats u and p*
    (∀ j < L, curvePt (param L (j + 1)) ∈ N_dir (curvePt (param L j))) ∧
    (∀ j, 1 ≤ j → j ≤ L →
      ⟪pStar, query⟫ < ⟪curvePt (param L 0), query⟫ ∧
      ⟪curvePt (param L 0), query⟫ < ⟪curvePt (param L j), query⟫) := by
  intro N_dir
  have hp_in_S : inDataset L pStar := Or.inr (Or.inl rfl)
  have hw_in_S : inDataset L winner := Or.inr (Or.inr rfl)
  refine ⟨?_, norm_query, ?_, ?_, ?_, ?_⟩
  · -- (1) Unit norms
    rintro x (⟨j, _, rfl⟩ | rfl | rfl)
    · exact norm_curvePt _
    · exact norm_pStar
    · exact norm_winner
  · -- (2) Cosine margin > 2/5
    rintro x (⟨j, hj, rfl⟩ | rfl | rfl) hne
    · exact winner_margin_curve (param_bounds hL hj).2
    · exact winner_margin_pStar
    · exact absurd rfl hne
  · -- (3) Bidirectional isolation of winner
    intro j hj
    have hc_in_S : inDataset L (curvePt (param L j)) := Or.inl ⟨j, hj, rfl⟩
    have hp_in_cj : pStar ∈ cand (curvePt (param L j)) :=
      hcand_complete _ hc_in_S pStar hp_in_S (curvePt_ne_pStar (param_bounds hL hj).1).symm
    have hp_in_w : pStar ∈ cand winner :=
      hcand_complete winner hw_in_S pStar hp_in_S pStar_ne_winner
    constructor
    · exact robustPrune_from_curve_drops_winner hL hj hα1 hα2 (cand _) (hcand_in _) (hcand_ne _)
        (hcand_sorted _) hp_in_cj
    · intro hmem
      have heq := robustPrune_from_winner_only_pStar hL hα1 hα2 (cand winner) (hcand_in _)
        (hcand_ne _) (hcand_sorted _) hp_in_w _ hmem
      exact curvePt_ne_pStar (param_bounds hL hj).1 heq
  · -- (4a) Forward decoy chain
    intro j hj
    have hc_in_S : inDataset L (curvePt (param L j)) := Or.inl ⟨j, Nat.le_of_lt hj, rfl⟩
    have hn_in_S : inDataset L (curvePt (param L (j + 1))) := Or.inl ⟨j + 1, Nat.succ_le_of_lt hj, rfl⟩
    have hne : curvePt (param L (j + 1)) ≠ curvePt (param L j) := by
      intro heq
      have hsc : ⟪curvePt (param L j), query⟫ < ⟪curvePt (param L (j + 1)), query⟫ :=
        inner_curvePt_query_strictMono (param_bounds hL (Nat.le_of_lt hj)).1
          (param_strictMono hL (Nat.lt_succ_self j)) (param_bounds hL (Nat.succ_le_of_lt hj)).2
      rw [heq] at hsc
      exact lt_irrefl _ hsc
    have hn_in_cand := hcand_complete _ hc_in_S _ hn_in_S hne
    exact robustPrune_from_curve_keeps_next_decoy hL hj hα1 (cand _) (hcand_in _) (hcand_ne _)
      (hcand_sorted _) hn_in_cand
  · -- (4b) Score dominance: p* < u = c(0) < d_j for all 1 ≤ j ≤ L
    intro j hj1 hjL
    constructor
    · have hp0 : param L 0 = 0 := by simp [param]
      rw [hp0, inner_curvePt_query, inner_pStar_query]
      norm_num
    · exact inner_curvePt_query_strictMono (param_bounds hL (Nat.zero_le L)).1
        (param_strictMono hL (by omega)) (param_bounds hL hjL).2

/-- In the symmetrized RobustPrune graph, `winner` has exactly one neighbor in `S_L`: `p*`. -/
theorem winner_symNeighbors_eq_pStar
    {L : ℕ} (hL : 1 ≤ L) {α : ℝ} (hα1 : 1 ≤ α) (hα2 : α ≤ 3 / 2)
    (cand : Vec3 → List Vec3)
    (hcand_in : ∀ v, ∀ x ∈ cand v, inDataset L x)
    (hcand_ne : ∀ v, v ∉ cand v)
    (hcand_sorted : ∀ v, (cand v).Pairwise (fun x y => ‖v - x‖ ^ 2 ≤ ‖v - y‖ ^ 2))
    (hcand_complete : ∀ v, inDataset L v → ∀ x, inDataset L x → x ≠ v → x ∈ cand v)
    (N_sym : Vec3 → Finset Vec3)
    (hN_sym_iff : ∀ u v, inDataset L u → inDataset L v →
      (v ∈ N_sym u ↔
        v ∈ robustPruneLoop u α (cand u) [] ∨ u ∈ robustPruneLoop v α (cand v) [])) :
    ∀ v, inDataset L v → (v ∈ N_sym winner ↔ v = pStar) := by
  have hp_in_S : inDataset L pStar := Or.inr (Or.inl rfl)
  have hw_in_S : inDataset L winner := Or.inr (Or.inr rfl)
  have hp_in_w : pStar ∈ cand winner :=
    hcand_complete winner hw_in_S pStar hp_in_S pStar_ne_winner
  intro v hv
  rw [hN_sym_iff winner v hw_in_S hv]
  constructor
  · rintro (h | h)
    · exact robustPrune_from_winner_only_pStar hL hα1 hα2 (cand winner) (hcand_in _)
        (hcand_ne _) (hcand_sorted _) hp_in_w v h
    · rcases hv with ⟨j, hj, rfl⟩ | rfl | rfl
      · exfalso
        have hc_in_S : inDataset L (curvePt (param L j)) := Or.inl ⟨j, hj, rfl⟩
        have hp_in_cj : pStar ∈ cand (curvePt (param L j)) :=
          hcand_complete _ hc_in_S pStar hp_in_S (curvePt_ne_pStar (param_bounds hL hj).1).symm
        exact robustPrune_from_curve_drops_winner hL hj hα1 hα2 (cand _) (hcand_in _)
          (hcand_ne _) (hcand_sorted _) hp_in_cj h
      · rfl
      · exfalso
        rcases mem_robustPruneLoop_subset h with h1 | h2
        · simp at h1
        · exact hcand_ne winner h2
  · rintro rfl
    exact Or.inl (robustPrune_from_winner_keeps_pStar hL (cand winner) (hcand_in _)
      (hcand_ne _) (hcand_sorted _) hp_in_w)

/-! ### Explicit Width-`L` Beam-Search State Machine and End-to-End Failure Theorem -/

/-- State of deterministic width-`L` MIPS beam search over `Vec3`: the active top-`L`
beam and the set of already-expanded vertices. -/
structure BeamState where
  beam : Finset Vec3
  visited : Finset Vec3

/-- Initial beam-search state starting at `start` with no vertices expanded yet. -/
def BeamState.init (start : Vec3) : BeamState :=
  ⟨{start}, ∅⟩

/-- One step of width-`L` MIPS beam search on a graph with neighbor sets `N`:
1. Select an unvisited vertex `cur ∈ s.beam \ s.visited` maximizing `⟪·, q⟫` over `s.beam \ s.visited`.
2. Mark `cur` visited (`s'.visited = insert cur s.visited`).
3. Form the candidate pool `pool = s.beam ∪ N cur`, and retain a top-`L` subset `s'.beam ⊆ pool`
   (full `pool` if `pool.card ≤ L`, else `s'.beam.card = L` and every retained element scores
   at least as high as every evicted element in `pool \ s'.beam`).

This relational step matches standard DiskANN/Vamana/HNSW beam search regardless of tie-breaking. -/
def BeamStep (N : Vec3 → Finset Vec3) (q : Vec3) (L : ℕ) (s s' : BeamState) : Prop :=
  ∃ cur ∈ s.beam \ s.visited,
    (∀ v ∈ s.beam \ s.visited, ⟪v, q⟫ ≤ ⟪cur, q⟫) ∧
    s'.visited = insert cur s.visited ∧
    let pool := s.beam ∪ N cur
    s'.beam ⊆ pool ∧
    s'.beam.card = min L pool.card ∧
    (∀ x ∈ s'.beam, ∀ y ∈ pool \ s'.beam, ⟪y, q⟫ ≤ ⟪x, q⟫)

/-- Reflexive-transitive closure of `BeamStep` (multi-step execution trace). -/
inductive BeamReach (N : Vec3 → Finset Vec3) (q : Vec3) (L : ℕ) : BeamState → BeamState → Prop
  | refl (s : BeamState) : BeamReach N q L s s
  | step {s₁ s₂ s₃ : BeamState} :
      BeamStep N q L s₁ s₂ → BeamReach N q L s₂ s₃ → BeamReach N q L s₁ s₃

/-- A beam state is terminal when every vertex currently in the beam has been expanded. -/
def BeamTerminal (s : BeamState) : Prop :=
  s.beam ⊆ s.visited

/-- Every `BeamStep` strictly grows `visited` inside a finite universe `U`, so any
execution starting inside `U` terminates in at most `U.card` steps (and never revisits
a vertex). -/
theorem beamStep_visited_card_lt {N : Vec3 → Finset Vec3} {q : Vec3} {L : ℕ}
    {s s' : BeamState} (hstep : BeamStep N q L s s') :
    s.visited.card < s'.visited.card := by
  rcases hstep with ⟨cur, hcur, _, hvis, _⟩
  rw [hvis, Finset.card_insert_of_notMem (Finset.mem_sdiff.mp hcur).2]
  exact Nat.lt_succ_self _

/-- In any graph where `winner` has no edge to any curve point `c(t_j)` (`0 ≤ j ≤ L`),
every decoy `c(t_j)` (`1 ≤ j ≤ L`) strictly outranks `c(0)` and `p*`, and every edge
stays in `S_L`: **no width-`L` beam-search execution starting at `u = c(0)` can ever
expand `p*`, admit `winner` into the beam, or visit `winner`**.

Invariant maintained at every step `s`:
- `s.beam ⊆ S_L \ {winner}` and `s.visited ⊆ S_L \ {p*, winner}`
- Whenever `p* ∈ s.beam \ s.visited`, at least one strictly higher-scoring vertex in
  `{c(t_0), …, c(t_L)}` is also in `s.beam \ s.visited` — wait: even stronger!
  We track `unvisited` curve points: actually, `c(0)` is visited at step 1, and each step
  expands the highest-scoring unvisited curve point; as long as `c(t_0), …, c(t_L)` are not
  all visited, wait — what if `c(t_j)` is expanded and introduces `c(t_{j+1})`?
  Because every `c(t_j)` (`0 ≤ j < L`) has `c(t_{j+1}) ∈ N (c(t_j))`, the *maximum*
  curve index `m` currently in `s.beam` is always either unvisited (if `m < L` and `c(t_m)`
  was just added or not yet expanded) or `m = L` (in which case all `L` decoys `c(t_1)..c(t_L)`
  have already entered `s.beam`, and once `c(t_1)..c(t_L)` are in `s.beam`, `p*` is evicted
  from any top-`L` beam!).

We prove the crisp invariant `BeamAvoidsWinner`:
1. `s.beam ⊆ curveSet L ∪ {p*}` (so `winner ∉ s.beam`)
2. `s.visited ⊆ curveSet L` (so `p* ∉ s.visited` and `winner ∉ s.visited`)
3. Either `p* ∉ s.beam`, OR there exists `j ≤ L` with `c(t_j) ∈ s.beam \ s.visited`
   (which strictly beats `p*`, preventing `p*` from ever being selected as `cur`).
4. For every `j < L`, if `c(t_j) ∈ s.visited`, then `c(t_{j+1}) ∈ s.beam`
   (because once a decoy enters `s.beam`, at most `L - 1` vertices in `curveSet L ∪ {p*}`
   can score strictly higher than `c(t_1)`—in fact `c(t_1)..c(t_L)` are the top `L`
   vertices of `curveSet L ∪ {p*}`!). -/
private def curveSet (L : ℕ) : Finset Vec3 :=
  (Finset.range (L + 1)).image (fun j => curvePt (param L j))

private def decoySet (L : ℕ) : Finset Vec3 :=
  (Finset.Icc 1 L).image (fun j => curvePt (param L j))

private theorem mem_curveSet_iff {L : ℕ} {v : Vec3} :
    v ∈ curveSet L ↔ isCurvePt L v := by
  simp only [curveSet, isCurvePt, Finset.mem_image, Finset.mem_range, Nat.lt_succ_iff]
  constructor
  · rintro ⟨j, hj, rfl⟩; exact ⟨j, hj, rfl⟩
  · rintro ⟨j, hj, rfl⟩; exact ⟨j, hj, rfl⟩

private theorem decoySet_subset_curveSet (L : ℕ) : decoySet L ⊆ curveSet L := by
  intro v hv
  rcases Finset.mem_image.mp hv with ⟨j, hj, rfl⟩
  exact Finset.mem_image.mpr ⟨j, Finset.mem_range.mpr (by have := (Finset.mem_Icc.mp hj).2; omega), rfl⟩

private theorem curvePt_param_inj {L : ℕ} (hL : 1 ≤ L) {i j : ℕ}
    (hi : i ≤ L) (hj : j ≤ L) (heq : curvePt (param L i) = curvePt (param L j)) : i = j := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
  · have hsc := inner_curvePt_query_strictMono (param_bounds hL hi).1
      (param_strictMono hL hlt) (param_bounds hL hj).2
    rw [heq] at hsc; exact lt_irrefl _ hsc
  · have hsc := inner_curvePt_query_strictMono (param_bounds hL hj).1
      (param_strictMono hL hgt) (param_bounds hL hi).2
    rw [heq] at hsc; exact lt_irrefl _ hsc

private theorem card_decoySet {L : ℕ} (hL : 1 ≤ L) : (decoySet L).card = L := by
  unfold decoySet
  rw [Finset.card_image_of_injOn]
  · simp
  · intro i hi j hj heq
    exact curvePt_param_inj hL (Finset.mem_Icc.mp hi).2 (Finset.mem_Icc.mp hj).2 heq

private theorem score_pStar_lt_curve {L j : ℕ} (hL : 1 ≤ L) (hj : j ≤ L) :
    ⟪pStar, query⟫ < ⟪curvePt (param L j), query⟫ := by
  have h0 : ⟪pStar, query⟫ < ⟪curvePt (param L 0), query⟫ := by
    have hp0 : param L 0 = 0 := by simp [param]
    rw [hp0, inner_curvePt_query, inner_pStar_query]; norm_num
  rcases Nat.eq_zero_or_pos j with rfl | hpos
  · exact h0
  · exact h0.trans (inner_curvePt_query_strictMono (param_bounds hL (Nat.zero_le L)).1
      (param_strictMono hL hpos) (param_bounds hL hj).2)

private theorem score_curve0_lt_decoy {L j : ℕ} (hL : 1 ≤ L) (hj1 : 1 ≤ j) (hjL : j ≤ L) :
    ⟪curvePt (param L 0), query⟫ < ⟪curvePt (param L j), query⟫ :=
  inner_curvePt_query_strictMono (param_bounds hL (Nat.zero_le L)).1
    (param_strictMono hL (by omega)) (param_bounds hL hjL).2

/-- Invariant for width-`L` beam search on the RobustPrune graph starting at `u = c(0)`:
1. `beam ⊆ curveSet L ∪ {p*}` (so `winner ∉ beam`),
2. `visited ⊆ curveSet L` (so `p* ∉ visited` and `winner ∉ visited`),
3. Every visited curve point `c(t_j)` (`j < L`) has its successor `c(t_{j+1})` in `beam`,
4. Either `c(0) ∈ visited`, or we are still at the initial singleton beam `{c(0)}` with `visited = ∅`,
5. Every decoy already in `visited` stays in `beam`. -/
private structure BeamInv (L : ℕ) (s : BeamState) : Prop where
  beam_sub : s.beam ⊆ insert pStar (curveSet L)
  vis_sub : s.visited ⊆ curveSet L
  vis_decoy_in_beam : ∀ v ∈ s.visited, v ∈ decoySet L → v ∈ s.beam
  succ_in_beam : ∀ j < L, curvePt (param L j) ∈ s.visited → curvePt (param L (j + 1)) ∈ s.beam
  start_present : curvePt (param L 0) ∈ s.visited ∨ s.beam = {curvePt (param L 0)}

private theorem beamInv_has_unvisited_curve_or_all_decoys {L : ℕ}
    {s : BeamState} (hinv : BeamInv L s) :
    (∃ j ≤ L, curvePt (param L j) ∈ s.beam \ s.visited) ∨ decoySet L ⊆ s.beam := by
  by_cases hall : ∀ j ≤ L, curvePt (param L j) ∈ s.visited
  · right
    intro v hv
    rcases Finset.mem_image.mp hv with ⟨j, hj, rfl⟩
    have hj1 := (Finset.mem_Icc.mp hj).1
    have hjL := (Finset.mem_Icc.mp hj).2
    have hpred : j - 1 < L := by omega
    have hsucc : j - 1 + 1 = j := by omega
    have hvpred : curvePt (param L (j - 1)) ∈ s.visited := hall (j - 1) (by omega)
    simpa [hsucc] using hinv.succ_in_beam (j - 1) hpred hvpred
  · push_neg at hall
    left
    -- Find the smallest j ≤ L with c(t_j) ∉ s.visited.
    let P : ℕ → Prop := fun j => j ≤ L ∧ curvePt (param L j) ∉ s.visited
    have hP : ∃ j, P j := hall
    classical
    let j₀ := Nat.find hP
    have hj₀ : P j₀ := Nat.find_spec hP
    refine ⟨j₀, hj₀.1, Finset.mem_sdiff.mpr ⟨?_, hj₀.2⟩⟩
    rcases Nat.eq_zero_or_pos j₀ with hzero | hpos
    · rw [hzero] at hj₀ ⊢
      rcases hinv.start_present with hv | hb
      · exact absurd hv hj₀.2
      · rw [hb]; exact Finset.mem_singleton_self _
    · have hlt : j₀ - 1 < j₀ := by omega
      have hnotP : ¬P (j₀ - 1) := Nat.find_min hP hlt
      have hprev_vis : curvePt (param L (j₀ - 1)) ∈ s.visited := by
        by_contra hnv
        exact hnotP ⟨by omega, hnv⟩
      have hsucc := hinv.succ_in_beam (j₀ - 1) (by omega) hprev_vis
      have heq : j₀ - 1 + 1 = j₀ := by omega
      rwa [heq] at hsucc

private theorem beamInv_pStar_not_max {L : ℕ} (hL : 1 ≤ L)
    {s : BeamState} (hinv : BeamInv L s) (hp_mem : pStar ∈ s.beam \ s.visited)
    (hcard : s.beam.card ≤ L) :
    ∃ v ∈ s.beam \ s.visited, ⟪pStar, query⟫ < ⟪v, query⟫ := by
  rcases beamInv_has_unvisited_curve_or_all_decoys hinv with ⟨j, hj, hmem⟩ | hdecoys
  · exact ⟨curvePt (param L j), hmem, score_pStar_lt_curve hL hj⟩
  · -- Impossible: decoySet L ⊆ s.beam and p* ∈ s.beam \ decoySet L would give card ≥ L + 1 > L!
    exfalso
    have hp_beam : pStar ∈ s.beam := (Finset.mem_sdiff.mp hp_mem).1
    have hp_not_decoy : pStar ∉ decoySet L := by
      intro hmem
      rcases Finset.mem_image.mp hmem with ⟨j, hj, heq⟩
      exact curvePt_ne_pStar (param_bounds hL (Finset.mem_Icc.mp hj).2).1 heq
    have hsub : insert pStar (decoySet L) ⊆ s.beam := Finset.insert_subset hp_beam hdecoys
    have hge : L + 1 ≤ s.beam.card := by
      have h1 := Finset.card_le_card hsub
      rw [Finset.card_insert_of_notMem hp_not_decoy, card_decoySet hL] at h1
      exact h1
    omega

private theorem beamInv_step {L : ℕ} (hL : 1 ≤ L)
    {N : Vec3 → Finset Vec3}
    (hN_in : ∀ v, inDataset L v → ∀ x ∈ N v, inDataset L x)
    (hN_no_win : ∀ j ≤ L, winner ∉ N (curvePt (param L j)))
    (hN_chain : ∀ j < L, curvePt (param L (j + 1)) ∈ N (curvePt (param L j)))
    {s s' : BeamState} (hinv : BeamInv L s) (hcard : s.beam.card ≤ L)
    (hstep : BeamStep N query L s s') :
    BeamInv L s' ∧ s'.beam.card ≤ L := by
  rcases hstep with ⟨cur, hcur_sdiff, hcur_max, hvis, hsub_pool, hcard_eq, htop⟩
  have hcard' : s'.beam.card ≤ L := hcard_eq ▸ Nat.min_le_left _ _
  -- cur cannot be p* (because p* is never maximal in s.beam \ s.visited).
  have hcur_ne_pStar : cur ≠ pStar := by
    intro heq
    subst heq
    obtain ⟨v, hv, hlt⟩ := beamInv_pStar_not_max hL hinv hcur_sdiff hcard
    linarith [hcur_max v hv]
  -- Hence cur ∈ curveSet L.
  have hcur_curve : cur ∈ curveSet L := by
    have hcur_in := hinv.beam_sub (Finset.mem_sdiff.mp hcur_sdiff).1
    rcases Finset.mem_insert.mp hcur_in with h1 | h2
    · exact absurd h1 hcur_ne_pStar
    · exact h2
  obtain ⟨jc, hjcL, hcur_eq⟩ := mem_curveSet_iff.mp hcur_curve
  -- Every vertex in pool = s.beam ∪ N cur lies in insert pStar (curveSet L).
  have hpool_sub : s.beam ∪ N cur ⊆ insert pStar (curveSet L) := by
    intro x hx
    rcases Finset.mem_union.mp hx with hxb | hxN
    · exact hinv.beam_sub hxb
    · have hx_in : inDataset L x := hN_in cur (Or.inl ⟨jc, hjcL, hcur_eq⟩) x hxN
      rcases hx_in with hxc | rfl | rfl
      · exact Finset.mem_insert_of_mem (mem_curveSet_iff.mpr hxc)
      · exact Finset.mem_insert_self _ _
      · exfalso; rw [hcur_eq] at hxN; exact hN_no_win jc hjcL hxN
  -- Any decoy in pool MUST survive into s'.beam (because only decoys can score ≥ a decoy
  -- inside insert pStar (curveSet L), and there are at most L decoys total!).
  have hdecoy_kept : ∀ d ∈ decoySet L, d ∈ s.beam ∪ N cur → d ∈ s'.beam := by
    intro d hd hd_pool
    by_contra hd_not
    have hd_evict : d ∈ (s.beam ∪ N cur) \ s'.beam := Finset.mem_sdiff.mpr ⟨hd_pool, hd_not⟩
    -- Since s'.beam ⊂ pool, we must have s'.beam.card = L (otherwise s'.beam = pool).
    have hcard_lt : s'.beam.card < (s.beam ∪ N cur).card :=
      Finset.card_lt_card (Finset.ssubset_iff_subset_ne.mpr ⟨hsub_pool, fun heq => hd_not (heq ▸ hd_pool)⟩)
    have hs'_card_L : s'.beam.card = L := by
      have : min L (s.beam ∪ N cur).card = L := by omega
      linarith
    -- Every x ∈ s'.beam satisfies ⟪d, query⟫ ≤ ⟪x, query⟫, so x cannot be p* or c(0),
    -- hence x ∈ decoySet L!
    have hs'_sub_decoy : s'.beam ⊆ decoySet L := by
      intro x hx
      have hle : ⟪d, query⟫ ≤ ⟪x, query⟫ := htop x hx d hd_evict
      rcases Finset.mem_image.mp hd with ⟨jd, hjd, rfl⟩
      have hjd1 := (Finset.mem_Icc.mp hjd).1
      have hjdL := (Finset.mem_Icc.mp hjd).2
      have hx_in := hpool_sub (hsub_pool hx)
      rcases Finset.mem_insert.mp hx_in with rfl | hxc
      · linarith [score_pStar_lt_curve hL hjdL]
      · obtain ⟨jx, hjxL, rfl⟩ := mem_curveSet_iff.mp hxc
        rcases Nat.eq_zero_or_pos jx with rfl | hjx_pos
        · linarith [score_curve0_lt_decoy hL hjd1 hjdL]
        · exact Finset.mem_image.mpr ⟨jx, Finset.mem_Icc.mpr ⟨hjx_pos, hjxL⟩, rfl⟩
    -- Yet s'.beam ⊆ (decoySet L).erase d has card ≤ L - 1, contradicting s'.beam.card = L!
    have hs'_sub_erase : s'.beam ⊆ (decoySet L).erase d :=
      Finset.subset_iff.mpr fun x hx =>
        Finset.mem_erase.mpr ⟨fun heq => hd_not (heq ▸ hx), hs'_sub_decoy hx⟩
    have hle_card := Finset.card_le_card hs'_sub_erase
    rw [Finset.card_erase_of_mem hd, card_decoySet hL, hs'_card_L] at hle_card
    omega
  refine ⟨⟨hsub_pool.trans hpool_sub, ?_, ?_, ?_, ?_⟩, hcard'⟩
  · -- visited' ⊆ curveSet L
    rw [hvis]
    exact Finset.insert_subset hcur_curve hinv.vis_sub
  · -- vis_decoy_in_beam
    intro v hv hvd
    rw [hvis] at hv
    rcases Finset.mem_insert.mp hv with rfl | hv_old
    · exact hdecoy_kept v hvd (Finset.mem_union_left _ (Finset.mem_sdiff.mp hcur_sdiff).1)
    · exact hdecoy_kept v hvd (Finset.mem_union_left _ (hinv.vis_decoy_in_beam v hv_old hvd))
  · -- succ_in_beam
    intro j hj hj_vis
    have hsucc_decoy : curvePt (param L (j + 1)) ∈ decoySet L :=
      Finset.mem_image.mpr ⟨j + 1, Finset.mem_Icc.mpr ⟨Nat.succ_pos j, Nat.succ_le_of_lt hj⟩, rfl⟩
    rw [hvis] at hj_vis
    rcases Finset.mem_insert.mp hj_vis with heq | hj_old
    · have hjeq : j = jc := curvePt_param_inj hL (Nat.le_of_lt hj) hjcL (heq.trans hcur_eq)
      subst hjeq
      apply hdecoy_kept _ hsucc_decoy
      exact Finset.mem_union_right _ (hcur_eq ▸ hN_chain j hj)
    · exact hdecoy_kept _ hsucc_decoy (Finset.mem_union_left _ (hinv.succ_in_beam j hj hj_old))
  · -- start_present
    rcases hinv.start_present with hv | hb
    · exact Or.inl (by rw [hvis]; exact Finset.mem_insert_of_mem hv)
    · have hcur_eq0 : cur = curvePt (param L 0) := by
        have hmem : cur ∈ ({curvePt (param L 0)} : Finset Vec3) := hb ▸ (Finset.mem_sdiff.mp hcur_sdiff).1
        exact Finset.mem_singleton.mp hmem
      exact Or.inl (by rw [hvis, hcur_eq0]; exact Finset.mem_insert_self _ _)

private theorem beamInv_init {L : ℕ} (hL : 1 ≤ L) :
    BeamInv L (BeamState.init (curvePt (param L 0))) ∧
      (BeamState.init (curvePt (param L 0))).beam.card ≤ L := by
  refine ⟨⟨?_, by simp [BeamState.init], by simp [BeamState.init], by simp [BeamState.init],
    Or.inr rfl⟩, by simp [BeamState.init, hL]⟩
  intro x hx
  simp only [BeamState.init, Finset.mem_singleton] at hx
  subst hx
  exact Finset.mem_insert_of_mem (mem_curveSet_iff.mpr ⟨0, Nat.zero_le L, rfl⟩)

private theorem beamInv_reach {L : ℕ} (hL : 1 ≤ L)
    {N : Vec3 → Finset Vec3}
    (hN_in : ∀ v, inDataset L v → ∀ x ∈ N v, inDataset L x)
    (hN_no_win : ∀ j ≤ L, winner ∉ N (curvePt (param L j)))
    (hN_chain : ∀ j < L, curvePt (param L (j + 1)) ∈ N (curvePt (param L j)))
    {s s' : BeamState} (hreach : BeamReach N query L s s')
    (hinv : BeamInv L s) (hcard : s.beam.card ≤ L) :
    BeamInv L s' ∧ s'.beam.card ≤ L := by
  induction hreach with
  | refl _ => exact ⟨hinv, hcard⟩
  | step hstep _ ih =>
    obtain ⟨hinv₂, hcard₂⟩ := beamInv_step hL hN_in hN_no_win hN_chain hinv hcard hstep
    exact ih hinv₂ hcard₂

/-- **End-to-End Width-`L` Beam-Search Failure (Theorem A1, Machine-Checked for All `L ≥ 1`):**
Let `N_dir v = robustPruneLoop v α (cand v) []` be the exact `RobustPrune` graph for any
`α ∈ [1, 3/2]` and `L ≥ 1`, and let `N_sym` be its symmetrization (any finite neighbor set
containing `N_dir v` and reverse edges of `N_dir`, inside `S_L`).
Then for **every** width-`L` beam-search state `s` reachable from `BeamState.init (c(0))`:
1. `winner ∉ s.beam` and `winner ∉ s.visited` (recall@1 = 0 at every step),
2. `pStar ∉ s.visited` (`p*` is never expanded), and
3. At any terminal state (`BeamTerminal s`), the beam equals the exact `L`-decoy trap
   `s.beam = {c(t_1), …, c(t_L)}` and `s.visited = {c(t_0), …, c(t_L)}`. -/
theorem beam_search_robustPrune_failure
    {L : ℕ} (hL : 1 ≤ L) {α : ℝ} (hα1 : 1 ≤ α) (hα2 : α ≤ 3 / 2)
    (cand : Vec3 → List Vec3)
    (hcand_in : ∀ v, ∀ x ∈ cand v, inDataset L x)
    (hcand_ne : ∀ v, v ∉ cand v)
    (hcand_sorted : ∀ v, (cand v).Pairwise (fun x y => ‖v - x‖ ^ 2 ≤ ‖v - y‖ ^ 2))
    (hcand_complete : ∀ v, inDataset L v → ∀ x, inDataset L x → x ≠ v → x ∈ cand v)
    (N_sym : Vec3 → Finset Vec3)
    (hN_sym_iff : ∀ u v, inDataset L u → inDataset L v →
      (v ∈ N_sym u ↔
        v ∈ robustPruneLoop u α (cand u) [] ∨ u ∈ robustPruneLoop v α (cand v) []))
    (hN_sym_in : ∀ u, inDataset L u → ∀ v ∈ N_sym u, inDataset L v)
    {s : BeamState}
    (hreach : BeamReach N_sym query L (BeamState.init (curvePt (param L 0))) s) :
    winner ∉ s.beam ∧ winner ∉ s.visited ∧ pStar ∉ s.visited ∧
      (BeamTerminal s → s.beam = decoySet L ∧ s.visited = curveSet L) := by
  have hfail := bidirectional_robustPrune_failure hL hα1 hα2 cand
    hcand_in hcand_ne hcand_sorted hcand_complete
  rcases hfail with ⟨_, _, _, hno_win, hchain, _⟩
  have hN_no_win : ∀ j ≤ L, winner ∉ N_sym (curvePt (param L j)) := by
    intro j hj hmem
    have hiff := (hN_sym_iff (curvePt (param L j)) winner
      (Or.inl ⟨j, hj, rfl⟩) (Or.inr (Or.inr rfl))).mp hmem
    rcases hiff with h1 | h2
    · exact (hno_win j hj).1 h1
    · exact (hno_win j hj).2 h2
  have hN_chain : ∀ j < L, curvePt (param L (j + 1)) ∈ N_sym (curvePt (param L j)) := by
    intro j hj
    exact (hN_sym_iff (curvePt (param L j)) (curvePt (param L (j + 1)))
      (Or.inl ⟨j, Nat.le_of_lt hj, rfl⟩) (Or.inl ⟨j + 1, Nat.succ_le_of_lt hj, rfl⟩)).mpr
      (Or.inl (hchain j hj))
  obtain ⟨hinv₀, hcard₀⟩ := beamInv_init hL
  obtain ⟨hinv, hcard⟩ := beamInv_reach hL hN_sym_in hN_no_win hN_chain hreach hinv₀ hcard₀
  have hwin_not_curve : winner ∉ curveSet L := by
    intro hm
    obtain ⟨j, hj, heq⟩ := mem_curveSet_iff.mp hm
    exact curvePt_ne_winner (param_bounds hL hj).2 heq.symm
  have hp_not_curve : pStar ∉ curveSet L := by
    intro hm
    obtain ⟨j, hj, heq⟩ := mem_curveSet_iff.mp hm
    exact curvePt_ne_pStar (param_bounds hL hj).1 heq.symm
  have hwin_not_beam : winner ∉ s.beam := by
    intro hm
    rcases Finset.mem_insert.mp (hinv.beam_sub hm) with heq | hc
    · exact pStar_ne_winner heq.symm
    · exact hwin_not_curve hc
  refine ⟨hwin_not_beam, fun hm => hwin_not_curve (hinv.vis_sub hm),
    fun hm => hp_not_curve (hinv.vis_sub hm), ?_⟩
  intro hterm
  -- At termination, s.beam \ s.visited = ∅, so decoySet L ⊆ s.beam and s.beam.card ≤ L
  -- forces s.beam = decoySet L, and all curve points must have been visited.
  have hdecoys_sub : decoySet L ⊆ s.beam := by
    rcases beamInv_has_unvisited_curve_or_all_decoys hinv with ⟨j, _, hmem⟩ | hd
    · exact absurd (hterm (Finset.mem_sdiff.mp hmem).1) (Finset.mem_sdiff.mp hmem).2
    · exact hd
  have hbeam_eq : s.beam = decoySet L := by
    symm
    apply Finset.eq_of_subset_of_card_le hdecoys_sub
    rw [card_decoySet hL]
    exact hcard
  refine ⟨hbeam_eq, Finset.Subset.antisymm hinv.vis_sub ?_⟩
  intro x hx
  obtain ⟨j, hj, rfl⟩ := mem_curveSet_iff.mp hx
  rcases Nat.eq_zero_or_pos j with rfl | hjpos
  · rcases hinv.start_present with hv | hb
    · exact hv
    · exfalso
      have h1 : curvePt (param L 0) ∈ decoySet L := by
        rw [← hbeam_eq, hb]; exact Finset.mem_singleton_self _
      rcases Finset.mem_image.mp h1 with ⟨j', hj', heq⟩
      have := curvePt_param_inj hL (Finset.mem_Icc.mp hj').2 (Nat.zero_le L) heq
      have := (Finset.mem_Icc.mp hj').1
      omega
  · exact hterm (hbeam_eq ▸ Finset.mem_image.mpr ⟨j, Finset.mem_Icc.mpr ⟨hjpos, hj⟩, rfl⟩)

end IndexRecall.RobustPruneFailure
