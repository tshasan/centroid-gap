import IndexRecall.OneVectorImpossibility

/-! # The one-vector impossibility in the plane (`d = 2`)

`IndexRecall.OneVectorImpossibility` places the `k` lists on `k` orthogonal axes
of `ℝ^{k+1}`. The lower bound there uses only one fact about the queries: the
two queries of list `i` are antipodal, so the linear part of any affine score
cancels in their sum. This file separates that convexity argument from
dimension.

1. **Dimension-free lower bound** (`antipodal_sum_tieRank_ge`,
   `antipodal_answered_le`, `antipodal_fullRecall_budget_ge`): in *every* real
   inner product space, for *any* vectors `u_0, …, u_{k-1}` with queries `±u_i`
   owned by list `i`, every one-vector affine router (arbitrary `r_j`, `b_j`,
   tie-breaking) has `k (k + 3) ≤ 2 ∑ rank`, and full Recall@1 at a fixed budget
   `p` forces `k + 1 ≤ 2 p`.

2. **A planar database** (`circle_doc_unit`, `circle_doc_injective`,
   `circle_query_margin`): in `ℝ²`, `k` lists of `m ≥ 2` pairwise distinct unit
   documents on rational points of the unit circle. List `i` holds the antipodal
   pair `±u(i/k)`, where `u(t) = ((1 - t²)/(1 + t²), 2t/(1 + t²))`, plus `m - 2`
   fillers. Each query `±u(i/k)` has a unique nearest document with cosine
   margin at least `1/(32 k²)`. (A margin of order `1/k²` is the best possible
   for `2k` distinct query directions on a circle.)

3. **Two tips still recover everything in `ℝ²`** (`circle_twoTip_returns_all`,
   `circle_twoTip_sum_work`), while every one-vector affine or power-diagram
   router needs average rank at least `(k + 3)/4` and total work
   `4k² + m k (k + 3) ≤ 2 ∑ work` (`circle_affine_sum_work_ge`,
   `circle_powerDist_sum_work_ge`).

4. **The bound is tight in `ℝ²` too** (`circle_witness_sum_tieRank_eq`): the
   router `r_j = 4k (4k + 2)^j u(j/k)^⊥`, `b_j = (4k + 2)^j` attains
   `2 ∑ rank = k (k + 3)` exactly. This witness uses exponentially large
   coefficients; whether polynomially bounded ones suffice in `ℝ²` is open here.
-/

noncomputable section

open scoped RealInnerProductSpace BigOperators Classical

namespace IndexRecall.OneVectorCircle

open IndexRecall.OneVectorImpossibility

/-! ### Part 1: the antipodal lower bound in any inner product space -/

section Generic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The `2 * k` antipodal queries `(i, true) ↦ u i`, `(i, false) ↦ -u i`. -/
def antiQuery {k : ℕ} (u : Fin k → E) (q : Fin k × Bool) : E := querySign q.2 • u q.1

/-- Negated one-vector affine score `-(⟪x, r j⟫ + b j)` (smaller is better). -/
def affScore {k : ℕ} (r : Fin k → E) (b : Fin k → ℝ) (x : E) (j : Fin k) : ℝ :=
  -(⟪x, r j⟫ + b j)

/-- Power-diagram score `‖x - c j‖² - w j` (smaller is better). -/
def powScore {k : ℕ} (c : Fin k → E) (w : Fin k → ℝ) (x : E) (j : Fin k) : ℝ :=
  ‖x - c j‖ ^ 2 - w j

theorem powScore_eq_affScore {k : ℕ} (c : Fin k → E) (w : Fin k → ℝ) {x : E}
    (hx : ‖x‖ = 1) (j : Fin k) :
    powScore c w x j = affScore (fun l => (2 : ℝ) • c l) (fun l => w l - ‖c l‖ ^ 2 - 1) x j := by
  unfold powScore affScore
  rw [norm_sub_sq_real, hx, real_inner_smul_right]
  ring

/-- Antipodal pairing in any inner product space: if `i ≺ j` in `(b, lt)`, list
`j` beats list `i` on at least one of `±u i`. -/
theorem antipodal_beats_of_biasOrder {k : ℕ} (u : Fin k → E) (r : Fin k → E) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) {i j : Fin k} (hij : biasOrder b lt i j) :
    beats (affScore r b (antiQuery u (i, false))) lt j i ∨
    beats (affScore r b (antiQuery u (i, true))) lt j i := by
  have hsum :
      (affScore r b (antiQuery u (i, false)) j - affScore r b (antiQuery u (i, false)) i) +
      (affScore r b (antiQuery u (i, true)) j - affScore r b (antiQuery u (i, true)) i) =
      2 * (b i - b j) := by
    simp only [affScore, antiQuery, querySign, Bool.false_eq_true, ↓reduceIte,
      neg_smul, one_smul, inner_neg_left]
    ring
  rcases hij with hlt | ⟨heq, hlt_tie⟩
  · by_contra h
    push_neg at h
    have h1 : ¬ affScore r b (antiQuery u (i, false)) j <
        affScore r b (antiQuery u (i, false)) i := fun hc => h.1 (Or.inl hc)
    have h2 : ¬ affScore r b (antiQuery u (i, true)) j <
        affScore r b (antiQuery u (i, true)) i := fun hc => h.2 (Or.inl hc)
    linarith
  · rw [heq, sub_self, mul_zero] at hsum
    by_cases h1 : affScore r b (antiQuery u (i, false)) j ≤
        affScore r b (antiQuery u (i, false)) i
    · rcases lt_or_eq_of_le h1 with hlt1 | heq1
      · exact Or.inl (Or.inl hlt1)
      · exact Or.inl (Or.inr ⟨heq1, hlt_tie⟩)
    · have h2 : affScore r b (antiQuery u (i, true)) j <
          affScore r b (antiQuery u (i, true)) i := by linarith
      exact Or.inr (Or.inl h2)

/-- **Dimension-free fixed-budget ceiling**: for any vectors `u` in any inner
product space, any one-vector affine router answers at most `k + 2 p - 1` of the
`2 k` antipodal queries within budget `p ≥ 1`. -/
theorem antipodal_answered_le {k : ℕ} (u : Fin k → E) (r : Fin k → E) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt]
    {p : ℕ} (hp : 1 ≤ p) :
    (Finset.univ.filter (fun q : Fin k × Bool =>
      tieRank (affScore r b (antiQuery u q)) lt q.1 ≤ p)).card ≤ k + 2 * p - 1 := by
  let ok (i : Fin k) (s : Bool) : Prop :=
    tieRank (affScore r b (antiQuery u (i, s))) lt i ≤ p
  let G : Finset (Fin k) := Finset.univ.filter (fun i => ok i false ∧ ok i true)
  have hsplit :
      (Finset.univ.filter (fun q : Fin k × Bool =>
        tieRank (affScore r b (antiQuery u q)) lt q.1 ≤ p)).card ≤ k + G.card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    have hstep :
        (∑ i : Fin k, ∑ s : Bool, if ok i s then (1 : ℕ) else 0) ≤
        ∑ i : Fin k, (1 + if i ∈ G then 1 else 0) := by
      refine Finset.sum_le_sum fun i _ => ?_
      rw [Fintype.sum_bool]
      have hmem : (i ∈ G) ↔ (ok i false ∧ ok i true) := by
        simp [G]
      by_cases hf : ok i false <;> by_cases ht : ok i true
      · have hG : i ∈ G := hmem.mpr ⟨hf, ht⟩
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => ht (hmem.mp h).2
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => hf (hmem.mp h).1
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => hf (hmem.mp h).1
        simp only [hf, ht, hG, ↓reduceIte]
        omega
    have hsum_eq : (∑ i : Fin k, (1 + if i ∈ G then (1 : ℕ) else 0)) = k + G.card := by
      rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, mul_one, Finset.card_univ,
        Fintype.card_fin, Finset.sum_boole, Nat.cast_id]
      simp [Finset.filter_mem_eq_inter]
    exact hstep.trans_eq hsum_eq
  have hG : G.card ≤ 2 * p - 1 := by
    by_cases hne : G.Nonempty
    · rcases exists_biasOrder_min b lt G hne with ⟨i₀, hi₀, hmin⟩
      have hi₀_ok : ok i₀ false ∧ ok i₀ true := (Finset.mem_filter.mp hi₀).2
      let Bf := Finset.univ.filter (fun j =>
        beats (affScore r b (antiQuery u (i₀, false))) lt j i₀)
      let Bt := Finset.univ.filter (fun j =>
        beats (affScore r b (antiQuery u (i₀, true))) lt j i₀)
      have hsub : G.erase i₀ ⊆ Bf ∪ Bt := by
        intro j hj
        have hjne : j ≠ i₀ := (Finset.mem_erase.mp hj).1
        have hjG : j ∈ G := (Finset.mem_erase.mp hj).2
        rcases antipodal_beats_of_biasOrder u r b lt (hmin j hjG hjne) with hbf | hbt
        · exact Finset.mem_union_left Bt (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hbf⟩)
        · exact Finset.mem_union_right Bf (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hbt⟩)
      have hcard : (G.erase i₀).card ≤ Bf.card + Bt.card :=
        (Finset.card_le_card hsub).trans (Finset.card_union_le Bf Bt)
      rw [Finset.card_erase_of_mem hi₀] at hcard
      have hGpos : 1 ≤ G.card := Finset.one_le_card.mpr hne
      have hBf : 1 + Bf.card ≤ p := hi₀_ok.1
      have hBt : 1 + Bt.card ≤ p := hi₀_ok.2
      omega
    · rw [Finset.not_nonempty_iff_eq_empty.mp hne, Finset.card_empty]
      omega
  omega

/-- **Dimension-free full-recall budget bound**: `k + 1 ≤ 2 p`. -/
theorem antipodal_fullRecall_budget_ge {k : ℕ} (hk : 1 ≤ k) (u : Fin k → E) (r : Fin k → E)
    (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] {p : ℕ}
    (hall : ∀ q : Fin k × Bool, tieRank (affScore r b (antiQuery u q)) lt q.1 ≤ p) :
    k + 1 ≤ 2 * p := by
  let i0 : Fin k := ⟨0, by omega⟩
  have hp : 1 ≤ p := by
    have h0 : tieRank (affScore r b (antiQuery u (i0, true))) lt i0 ≤ p := hall (i0, true)
    have hpos := tieRank_pos (affScore r b (antiQuery u (i0, true))) lt i0
    omega
  have hle := antipodal_answered_le u r b lt hp
  have hfull : (Finset.univ.filter (fun q : Fin k × Bool =>
      tieRank (affScore r b (antiQuery u q)) lt q.1 ≤ p)).card = 2 * k := by
    rw [Finset.filter_true_of_mem (fun q _ => hall q)]
    simp [Fintype.card_prod, mul_comm]
  omega

/-- **Dimension-free average-rank bound**: `k (k + 3) ≤ 2 ∑ q, tieRank q`. -/
theorem antipodal_sum_tieRank_ge {k : ℕ} (u : Fin k → E) (r : Fin k → E) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] :
    k * (k + 3) ≤ 2 * ∑ q : Fin k × Bool, tieRank (affScore r b (antiQuery u q)) lt q.1 := by
  let ind (i j : Fin k) (s : Bool) : ℕ :=
    if beats (affScore r b (antiQuery u (i, s))) lt j i then 1 else 0
  let pairCount (i j : Fin k) : ℕ := ind i j false + ind i j true
  have hrewrite :
      (∑ q : Fin k × Bool, tieRank (affScore r b (antiQuery u q)) lt q.1) =
      2 * k + ∑ i : Fin k, ∑ j : Fin k, pairCount i j := by
    rw [Fintype.sum_prod_type]
    have h1 :
        (∑ i : Fin k, ∑ s : Bool, tieRank (affScore r b (antiQuery u (i, s))) lt i) =
        ∑ i : Fin k, (2 + ∑ j : Fin k, pairCount i j) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      simp only [Fintype.sum_bool, tieRank, Finset.card_filter, pairCount, ind]
      rw [Finset.sum_add_distrib]
      omega
    rw [h1, Finset.sum_add_distrib]
    simp [mul_comm]
  rw [hrewrite]
  have hsym :
      2 * (∑ i : Fin k, ∑ j : Fin k, pairCount i j) =
      ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
    rw [two_mul]
    conv_lhs => rhs; rw [Finset.sum_comm]
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
  have hoff :
      k * (k - 1) ≤ ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
    have hpt :
        (∑ i : Fin k, ∑ j : Fin k, if i = j then (0 : ℕ) else 1) ≤
        ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
      refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
      by_cases hij : i = j
      · simp [hij]
      · rw [if_neg hij]
        rcases biasOrder_total b lt hij with h | h
        · rcases antipodal_beats_of_biasOrder u r b lt h with hb | hb <;>
            · simp only [pairCount, ind, hb, ↓reduceIte]; omega
        · rcases antipodal_beats_of_biasOrder u r b lt h with hb | hb <;>
            · simp only [pairCount, ind, hb, ↓reduceIte]; omega
    have hcount :
        (∑ i : Fin k, ∑ j : Fin k, if i = j then (0 : ℕ) else 1) = k * (k - 1) := by
      have hinner (i : Fin k) :
          (∑ j : Fin k, if i = j then (0 : ℕ) else 1) = k - 1 := by
        have hfilt : Finset.univ.filter (fun j : Fin k => ¬ i = j) = Finset.univ.erase i := by
          ext j
          simp [ne_comm]
        rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, smul_eq_mul,
          mul_one, hfilt, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
          Fintype.card_fin]
      simp [hinner]
    omega
  have hmul_eq : 4 * k + k * (k - 1) = k * (k + 3) := by
    cases k with
    | zero => simp
    | succ n =>
      simp only [Nat.add_sub_cancel]
      ring
  omega

/-- `minProbedCount` version of `antipodal_sum_tieRank_ge` (via `Fin.lt`). -/
theorem antipodal_sum_minProbedCount_ge {k : ℕ} (u : Fin k → E) (r : Fin k → E)
    (b : Fin k → ℝ) :
    k * (k + 3) ≤ 2 * ∑ q : Fin k × Bool, minProbedCount (affScore r b (antiQuery u q)) q.1 := by
  have h1 := antipodal_sum_tieRank_ge u r b (· < ·)
  have h2 :
      (∑ q : Fin k × Bool, tieRank (affScore r b (antiQuery u q)) (· < ·) q.1) ≤
      ∑ q : Fin k × Bool, minProbedCount (affScore r b (antiQuery u q)) q.1 :=
    Finset.sum_le_sum fun q _ => tieRank_le_minProbedCount _ _ q.1
  omega

end Generic

/-- If a flat scan returns an answer in list `a.1`, it scored all `k` lists and
scanned at least `minProbedCount` lists of size `m`. -/
theorem work_ge_of_returns' {k m : ℕ} {pointScore : Fin k × Fin m → ℝ} {score : Fin k → ℝ}
    {budget : ℕ} {a : Fin k × Fin m} (hret : FlatScan.Returns pointScore score budget a) :
    k + m * minProbedCount score a.1 ≤ FlatScan.work (Fin m) score budget := by
  have hi : FlatScan.rank score a.1 ≤ budget := by
    have h := hret.1
    simp only [FlatScan.candidates, FlatScan.probed, Finset.mem_product, Finset.mem_filter,
      Finset.mem_univ, true_and, and_true] at h
    exact h
  have hc := minProbedCount_le_probed_card hi
  simp only [FlatScan.work, FlatScan.candidates, Finset.card_product, Finset.card_univ,
    Fintype.card_fin]
  have hmul : m * minProbedCount score a.1 ≤ (FlatScan.probed score budget).card * m := by
    rw [mul_comm]
    exact Nat.mul_le_mul_right m hc
  omega

/-! ### Part 2: rational points on the unit circle -/

abbrev Plane := EuclideanSpace ℝ (Fin 2)

/-- First basis vector of `ℝ²`. -/
def ex : Plane := EuclideanSpace.single 0 1

/-- Second basis vector of `ℝ²`. -/
def ey : Plane := EuclideanSpace.single 1 1

theorem ex_inner_ex : ⟪ex, ex⟫ = 1 := by simp [ex, EuclideanSpace.inner_single_left]
theorem ey_inner_ey : ⟪ey, ey⟫ = 1 := by simp [ey, EuclideanSpace.inner_single_left]
theorem ex_inner_ey : ⟪ex, ey⟫ = 0 := by simp [ex, ey, EuclideanSpace.inner_single_left]
theorem ey_inner_ex : ⟪ey, ex⟫ = 0 := by simp [ex, ey, EuclideanSpace.inner_single_left]

/-- `x`-coordinate of the rational circle point `u(t)`. -/
def cx (t : ℝ) : ℝ := (1 - t ^ 2) / (1 + t ^ 2)

/-- `y`-coordinate of the rational circle point `u(t)`. -/
def cy (t : ℝ) : ℝ := 2 * t / (1 + t ^ 2)

/-- Rational unit-circle point `u(t) = ((1 - t²)/(1 + t²), 2t/(1 + t²))`. -/
def circ (t : ℝ) : Plane := cx t • ex + cy t • ey

/-- The rotation of `u(t)` by a quarter turn. -/
def perp (t : ℝ) : Plane := (-cy t) • ex + cx t • ey

theorem inner_circ_circ (t t' : ℝ) : ⟪circ t, circ t'⟫ = cx t * cx t' + cy t * cy t' := by
  simp only [circ, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
    ex_inner_ex, ey_inner_ey, ex_inner_ey, ey_inner_ex]
  ring

theorem inner_circ_perp (t t' : ℝ) : ⟪circ t, perp t'⟫ = cy t * cx t' - cx t * cy t' := by
  simp only [circ, perp, inner_add_left, inner_add_right, real_inner_smul_left,
    real_inner_smul_right, ex_inner_ex, ey_inner_ey, ex_inner_ey, ey_inner_ex]
  ring

private theorem one_add_sq_pos (t : ℝ) : 0 < 1 + t ^ 2 := by positivity

theorem cx_sq_add_cy_sq (t : ℝ) : cx t ^ 2 + cy t ^ 2 = 1 := by
  have h := one_add_sq_pos t
  unfold cx cy
  field_simp
  ring

/-- Chord identity: `1 - ⟪u(t), u(t')⟫ = 2 (t - t')² / ((1 + t²)(1 + t'²))`. -/
theorem one_sub_inner_circ (t t' : ℝ) :
    1 - ⟪circ t, circ t'⟫ = 2 * (t - t') ^ 2 / ((1 + t ^ 2) * (1 + t' ^ 2)) := by
  have h := one_add_sq_pos t
  have h' := one_add_sq_pos t'
  rw [inner_circ_circ]
  unfold cx cy
  field_simp
  ring

/-- `⟪u(t), u(t')^⊥⟫ = 2 (t - t')(1 + t t') / ((1 + t²)(1 + t'²))`. -/
theorem inner_circ_perp_eq (t t' : ℝ) :
    ⟪circ t, perp t'⟫ = 2 * (t - t') * (1 + t * t') / ((1 + t ^ 2) * (1 + t' ^ 2)) := by
  have h := one_add_sq_pos t
  have h' := one_add_sq_pos t'
  rw [inner_circ_perp]
  unfold cx cy
  field_simp
  ring

theorem inner_circ_self (t : ℝ) : ⟪circ t, circ t⟫ = 1 := by
  rw [inner_circ_circ, ← cx_sq_add_cy_sq t]
  ring

theorem circ_unit (t : ℝ) : ‖circ t‖ = 1 := by
  have hsq : ‖circ t‖ ^ 2 = 1 := by
    rw [← real_inner_self_eq_norm_sq, inner_circ_self]
  nlinarith [norm_nonneg (circ t)]

theorem abs_inner_circ_perp_le (t t' : ℝ) : |⟪circ t, perp t'⟫| ≤ 1 := by
  rw [inner_circ_perp]
  have h1 := cx_sq_add_cy_sq t
  have h2 := cx_sq_add_cy_sq t'
  have hid : (cy t * cx t' - cx t * cy t') ^ 2 + (cx t * cx t' + cy t * cy t') ^ 2 =
      (cx t ^ 2 + cy t ^ 2) * (cx t' ^ 2 + cy t' ^ 2) := by ring
  rw [h1, h2, one_mul] at hid
  rw [abs_le]
  constructor <;> nlinarith [sq_nonneg (cx t * cx t' + cy t * cy t')]

theorem inner_circ_nonneg {t t' : ℝ} (ht : 0 ≤ t) (ht1 : t ≤ 1) (ht' : 0 ≤ t') (ht1' : t' ≤ 1) :
    0 ≤ ⟪circ t, circ t'⟫ := by
  rw [inner_circ_circ]
  have hx : 0 ≤ cx t := div_nonneg (by nlinarith) (one_add_sq_pos t).le
  have hx' : 0 ≤ cx t' := div_nonneg (by nlinarith) (one_add_sq_pos t').le
  have hy : 0 ≤ cy t := div_nonneg (by linarith) (one_add_sq_pos t).le
  have hy' : 0 ≤ cy t' := div_nonneg (by linarith) (one_add_sq_pos t').le
  positivity

/-- On `[0, 1]`, parameters at distance `δ` give cosine at most `1 - δ²/2`. -/
theorem inner_circ_le_of_sep {t t' δ : ℝ} (ht : 0 ≤ t) (ht1 : t ≤ 1) (ht' : 0 ≤ t')
    (ht1' : t' ≤ 1) (hδ : 0 ≤ δ) (hsep : δ ≤ |t - t'|) :
    ⟪circ t, circ t'⟫ + δ ^ 2 / 2 ≤ 1 := by
  have h := one_sub_inner_circ t t'
  have hD : 0 < (1 + t ^ 2) * (1 + t' ^ 2) := by positivity
  have ht2 : t ^ 2 ≤ 1 := by nlinarith
  have ht2' : t' ^ 2 ≤ 1 := by nlinarith
  have hD4 : (1 + t ^ 2) * (1 + t' ^ 2) ≤ 4 := by
    nlinarith [mul_le_mul ht2 ht2' (sq_nonneg t') zero_le_one]
  have hsq : δ ^ 2 ≤ (t - t') ^ 2 := by
    rw [← sq_abs (t - t')]
    exact pow_le_pow_left₀ hδ hsep 2
  have hfrac : δ ^ 2 / 2 ≤ 2 * (t - t') ^ 2 / ((1 + t ^ 2) * (1 + t' ^ 2)) := by
    rw [le_div_iff₀ hD]
    nlinarith
  linarith

/-! ### Part 3: a balanced planar database -/

/-- Slots `0, 1` sit on the list's own direction; slot `t ≥ 2` is offset by
`1/4 + 1/(2t) ∈ (1/4, 1/2]` toward the next direction. -/
def slotOff {m : ℕ} (t : Fin m) : ℝ := if t.val ≤ 1 then 0 else 1 / 4 + 1 / (2 * (t.val : ℝ))

/-- Slot `1` is the antipode of the list direction; every other slot has sign `+1`. -/
def slotSign {m : ℕ} (t : Fin m) : ℝ := if t.val = 1 then -1 else 1

/-- Circle parameter `i / k` of list `i`'s direction. -/
def tipParam {k : ℕ} (i : Fin k) : ℝ := (i.val : ℝ) / k

/-- Circle parameter of document `x`. -/
def docParam {k m : ℕ} (x : Fin k × Fin m) : ℝ := ((x.1.val : ℝ) + slotOff x.2) / k

/-- Planar document `x = ± u(docParam x)`. -/
def cdoc {k m : ℕ} (x : Fin k × Fin m) : Plane := slotSign x.2 • circ (docParam x)

/-- Direction `u(i/k)` of list `i`. -/
def tipDir {k : ℕ} (i : Fin k) : Plane := circ (tipParam i)

/-- The `2 k` planar queries `± u(i/k)`. -/
def cquery {k : ℕ} (q : Fin k × Bool) : Plane := antiQuery tipDir q

theorem slotOff_of_le_one {m : ℕ} {t : Fin m} (ht : t.val ≤ 1) : slotOff t = 0 := by
  simp [slotOff, ht]

theorem slotOff_bounds {m : ℕ} {t : Fin m} (ht : 2 ≤ t.val) :
    1 / 4 < slotOff t ∧ slotOff t ≤ 1 / 2 := by
  have hn : ¬ t.val ≤ 1 := by omega
  have h2 : (2 : ℝ) ≤ t.val := by exact_mod_cast ht
  have hpos : (0 : ℝ) < 2 * (t.val : ℝ) := by linarith
  have hle : 1 / (2 * (t.val : ℝ)) ≤ 1 / 4 := by
    rw [div_le_div_iff₀ hpos (by norm_num)]; linarith
  have hgt : 0 < 1 / (2 * (t.val : ℝ)) := by positivity
  simp only [slotOff, hn, ↓reduceIte]
  constructor <;> linarith

theorem slotSign_cases {m : ℕ} (t : Fin m) : slotSign t = 1 ∨ slotSign t = -1 := by
  unfold slotSign; split_ifs <;> simp

private theorem kpos_of {k : ℕ} (i : Fin k) : (0 : ℝ) < k := by
  have : 0 < k := Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  exact_mod_cast this

theorem tipParam_mem {k : ℕ} (i : Fin k) : 0 ≤ tipParam i ∧ tipParam i ≤ 1 := by
  have hk := kpos_of i
  have hi : (i.val : ℝ) + 1 ≤ k := by exact_mod_cast i.isLt
  unfold tipParam
  refine ⟨by positivity, ?_⟩
  rw [div_le_one hk]; linarith

theorem docParam_mem {k m : ℕ} (x : Fin k × Fin m) : 0 ≤ docParam x ∧ docParam x ≤ 1 := by
  have hk := kpos_of x.1
  have hi : (x.1.val : ℝ) + 1 ≤ k := by exact_mod_cast x.1.isLt
  have hoff : 0 ≤ slotOff x.2 ∧ slotOff x.2 ≤ 1 / 2 := by
    by_cases ht : x.2.val ≤ 1
    · rw [slotOff_of_le_one ht]; norm_num
    · have := slotOff_bounds (t := x.2) (by omega)
      constructor <;> linarith
  have hi0 : (0 : ℝ) ≤ x.1.val := Nat.cast_nonneg _
  unfold docParam
  refine ⟨div_nonneg (by linarith [hoff.1]) hk.le, ?_⟩
  rw [div_le_one hk]; linarith

theorem cdoc_slot_zero {k m : ℕ} (hm : 2 ≤ m) (i : Fin k) :
    cdoc (i, (⟨0, by omega⟩ : Fin m)) = tipDir i := by
  simp [cdoc, slotSign, docParam, slotOff, tipDir, tipParam]

theorem cdoc_slot_one {k m : ℕ} (hm : 2 ≤ m) (i : Fin k) :
    cdoc (i, (⟨1, by omega⟩ : Fin m)) = -tipDir i := by
  simp [cdoc, slotSign, docParam, slotOff, tipDir, tipParam]

theorem cdoc_unit {k m : ℕ} (x : Fin k × Fin m) : ‖cdoc x‖ = 1 := by
  rcases slotSign_cases x.2 with h | h <;> simp [cdoc, h, circ_unit]

theorem cquery_unit {k : ℕ} (q : Fin k × Bool) : ‖cquery q‖ = 1 := by
  rcases q with ⟨i, s⟩
  cases s <;> simp [cquery, antiQuery, querySign, tipDir, circ_unit]

theorem inner_cquery_cdoc {k m : ℕ} (q : Fin k × Bool) (x : Fin k × Fin m) :
    ⟪cquery q, cdoc x⟫ =
      querySign q.2 * slotSign x.2 * ⟪circ (tipParam q.1), circ (docParam x)⟫ := by
  simp only [cquery, antiQuery, cdoc, tipDir, real_inner_smul_left, real_inner_smul_right]
  ring

theorem cquery_target_inner {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    ⟪cquery q, cdoc (targetDoc hm q)⟫ = 1 := by
  rcases q with ⟨i, s⟩
  cases s
  · simp only [targetDoc, Bool.false_eq_true, ↓reduceIte, cdoc_slot_one hm, cquery, antiQuery,
      querySign, neg_smul, one_smul, inner_neg_neg, tipDir, inner_circ_self]
  · simp only [targetDoc, ↓reduceIte, cdoc_slot_zero hm, cquery, antiQuery, querySign, one_smul,
      tipDir, inner_circ_self]

private theorem nat_sep {a b : ℕ} (h : a ≠ b) : 1 ≤ |(a : ℝ) - b| := by
  rcases Nat.lt_or_gt_of_ne h with h | h
  · have : (a : ℝ) + 1 ≤ b := by exact_mod_cast h
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]; linarith
  · have : (b : ℝ) + 1 ≤ a := by exact_mod_cast h
    rw [abs_of_nonneg (by linarith)]; linarith

private theorem nat_off_sep (a b : ℕ) {o : ℝ} (ho : 1 / 4 < o) (ho' : o ≤ 1 / 2) :
    1 / 4 ≤ |(a : ℝ) - b - o| := by
  rcases le_or_gt a b with h | h
  · have : (a : ℝ) ≤ b := by exact_mod_cast h
    rw [abs_of_neg (by linarith)]; linarith
  · have : (b : ℝ) + 1 ≤ a := by exact_mod_cast h
    rw [abs_of_nonneg (by linarith)]; linarith

/-- Every non-target document either points into the opposite half-plane of the
query or has circle parameter at least `1/(4k)` away from it. -/
theorem nonTarget_sep {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) (x : Fin k × Fin m)
    (hx : x ≠ targetDoc hm q) :
    querySign q.2 * slotSign x.2 = -1 ∨ 1 / (4 * k) ≤ |tipParam q.1 - docParam x| := by
  rcases q with ⟨i, s⟩
  rcases x with ⟨j, t⟩
  have hk := kpos_of i
  have hdiff : tipParam i - docParam (j, t) = ((i.val : ℝ) - j.val - slotOff t) / k := by
    unfold tipParam docParam; field_simp; ring
  have hscale : ∀ c : ℝ, 1 / 4 ≤ |c| → 1 / (4 * k) ≤ |c / k| := by
    intro c hc
    rw [abs_div, abs_of_pos hk, div_le_div_iff₀ (by positivity) hk]
    nlinarith
  by_cases ht : 2 ≤ t.val
  · right
    rw [hdiff]
    exact hscale _ (nat_off_sep _ _ (slotOff_bounds ht).1 (slotOff_bounds ht).2)
  · rw [hdiff, slotOff_of_le_one (by omega), sub_zero]
    have hji_sep : j ≠ i → 1 / (4 * k) ≤ |((i.val : ℝ) - j.val) / k| := by
      intro hne
      refine hscale _ ?_
      have := nat_sep (fun h : i.val = j.val => hne (Fin.ext h.symm))
      linarith
    have hne_of : t.val = (if s then 0 else 1) → j ≠ i := by
      intro htv hji
      apply hx
      subst hji
      refine Prod.ext rfl (Fin.ext ?_)
      cases s <;> simpa [targetDoc] using htv
    rcases (show t.val = 0 ∨ t.val = 1 by omega) with h0 | h1 <;> cases s
    · left; simp [querySign, slotSign, h0]
    · right; exact hji_sep (hne_of (by simp [h0]))
    · right; exact hji_sep (hne_of (by simp [h1]))
    · left; simp [querySign, slotSign, h1]

private theorem one_le_k {k : ℕ} (i : Fin k) : (1 : ℝ) ≤ k := by
  have : 1 ≤ k := Nat.one_le_iff_ne_zero.mpr (by intro h; have := i.isLt; omega)
  exact_mod_cast this

/-- **Unique nearest neighbor with margin `1/(32 k²)`** for every planar query. -/
theorem circle_query_margin {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) (x : Fin k × Fin m)
    (hx : x ≠ targetDoc hm q) :
    ⟪cquery q, cdoc x⟫ + 1 / (32 * (k : ℝ) ^ 2) ≤ ⟪cquery q, cdoc (targetDoc hm q)⟫ := by
  rw [cquery_target_inner hm q, inner_cquery_cdoc]
  have hk1 := one_le_k q.1
  obtain ⟨h0, h1⟩ := tipParam_mem q.1
  obtain ⟨h0', h1'⟩ := docParam_mem x
  have hg := inner_circ_nonneg h0 h1 h0' h1'
  have hsmall : 1 / (32 * (k : ℝ) ^ 2) ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith
  rcases nonTarget_sep hm q x hx with hneg | hsep
  · rw [hneg]
    linarith
  · have hδ : (0 : ℝ) ≤ 1 / (4 * k) := by positivity
    have hb := inner_circ_le_of_sep h0 h1 h0' h1' hδ hsep
    have hsq : (1 / (4 * (k : ℝ))) ^ 2 / 2 = 1 / (32 * (k : ℝ) ^ 2) := by
      field_simp; ring
    rw [hsq] at hb
    have hprod : querySign q.2 * slotSign x.2 = 1 ∨ querySign q.2 * slotSign x.2 = -1 := by
      rcases slotSign_cases x.2 with h | h <;> cases q.2 <;> simp [querySign, h]
    rcases hprod with h | h <;> rw [h] <;> linarith

/-- All `k m` planar documents are pairwise distinct. -/
theorem circle_doc_injective {k m : ℕ} : Function.Injective (cdoc (k := k) (m := m)) := by
  intro x y hxy
  have hin : ⟪cdoc x, cdoc y⟫ = 1 := by
    rw [hxy, real_inner_self_eq_norm_sq, cdoc_unit]; norm_num
  simp only [cdoc, real_inner_smul_left, real_inner_smul_right] at hin
  obtain ⟨h0, h1⟩ := docParam_mem x
  obtain ⟨h0', h1'⟩ := docParam_mem y
  have hg := inner_circ_nonneg h0 h1 h0' h1'
  have hsign : slotSign x.2 = slotSign y.2 ∧ ⟪circ (docParam x), circ (docParam y)⟫ = 1 := by
    rcases slotSign_cases x.2 with hx | hx <;> rcases slotSign_cases y.2 with hy | hy <;>
      rw [hx, hy] at hin
    · exact ⟨hx.trans hy.symm, by linarith⟩
    · exfalso; linarith
    · exfalso; linarith
    · exact ⟨hx.trans hy.symm, by linarith⟩
  have hpar : docParam x = docParam y := by
    have h := one_sub_inner_circ (docParam x) (docParam y)
    rw [hsign.2, sub_self] at h
    have hD : 0 < (1 + docParam x ^ 2) * (1 + docParam y ^ 2) := by positivity
    have hz : (docParam x - docParam y) ^ 2 = 0 := by
      have h2 := (div_eq_zero_iff.mp h.symm).resolve_right hD.ne'
      linarith
    exact sub_eq_zero.mp (pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hz)
  have hsg := hsign.1
  rcases x with ⟨i, t⟩
  rcases y with ⟨j, u⟩
  have hk := kpos_of i
  have hsum : (i.val : ℝ) - j.val - slotOff u + slotOff t = 0 := by
    unfold docParam at hpar
    simp only at hpar
    rw [div_eq_div_iff hk.ne' hk.ne'] at hpar
    have := mul_right_cancel₀ hk.ne' hpar
    linarith
  simp only at hsg
  by_cases ht : 2 ≤ t.val <;> by_cases hu : 2 ≤ u.val
  · obtain ⟨a1, a2⟩ := slotOff_bounds ht
    obtain ⟨b1, b2⟩ := slotOff_bounds hu
    have hij : i = j := by
      by_contra hne
      have h := nat_sep (fun h : i.val = j.val => hne (Fin.ext h))
      rw [show (i.val : ℝ) - j.val = slotOff u - slotOff t by linarith] at h
      have habs : |slotOff u - slotOff t| < 1 := abs_sub_lt_iff.mpr ⟨by linarith, by linarith⟩
      linarith
    subst hij
    have hoff : slotOff t = slotOff u := by linarith
    have htn : ¬ t.val ≤ 1 := by omega
    have hun : ¬ u.val ≤ 1 := by omega
    simp only [slotOff, htn, hun, ↓reduceIte, add_right_inj] at hoff
    have h2t : (0 : ℝ) < 2 * t.val := by
      have : (2 : ℝ) ≤ t.val := by exact_mod_cast ht
      linarith
    have h2u : (0 : ℝ) < 2 * u.val := by
      have : (2 : ℝ) ≤ u.val := by exact_mod_cast hu
      linarith
    rw [div_eq_div_iff h2t.ne' h2u.ne'] at hoff
    have : (t.val : ℝ) = u.val := by linarith
    exact Prod.ext rfl (Fin.ext (Nat.cast_injective this))
  · rw [slotOff_of_le_one (show u.val ≤ 1 by omega)] at hsum
    have h := nat_off_sep j i (slotOff_bounds ht).1 (slotOff_bounds ht).2
    rw [show (j.val : ℝ) - i.val - slotOff t = 0 by linarith, abs_zero] at h
    norm_num at h
  · rw [slotOff_of_le_one (show t.val ≤ 1 by omega)] at hsum
    have h := nat_off_sep i j (slotOff_bounds hu).1 (slotOff_bounds hu).2
    rw [show (i.val : ℝ) - j.val - slotOff u = 0 by linarith, abs_zero] at h
    norm_num at h
  · rw [slotOff_of_le_one (show t.val ≤ 1 by omega),
      slotOff_of_le_one (show u.val ≤ 1 by omega)] at hsum
    have hij : i = j := Fin.ext (Nat.cast_injective (by linarith : (i.val : ℝ) = j.val))
    subst hij
    have htu : t.val = u.val := by
      unfold slotSign at hsg
      by_cases h1 : t.val = 1 <;> by_cases h2 : u.val = 1 <;>
        simp only [h1, h2, ↓reduceIte] at hsg <;> first | omega | norm_num at hsg
    exact Prod.ext rfl (Fin.ext htu)

theorem circle_database_size (k m : ℕ) : Fintype.card (Fin k × Fin m) = k * m := by
  simp [Fintype.card_prod]

/-! ### Part 4: two tips per list still recover every answer in `ℝ²` -/

/-- Two-tip score: list `j` stores its own `± u(j/k)`. -/
def ctipScore {k m : ℕ} (hm : 2 ≤ m) (x : Plane) (j : Fin k) : ℝ :=
  max ⟪x, cdoc (j, (⟨0, by omega⟩ : Fin m))⟫ ⟪x, cdoc (j, (⟨1, by omega⟩ : Fin m))⟫

/-- Negated two-tip score (smaller is better). -/
def negCTip {k m : ℕ} (hm : 2 ≤ m) (x : Plane) (j : Fin k) : ℝ := -ctipScore hm x j

theorem ctipScore_query {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) (j : Fin k) :
    ctipScore hm (cquery q) j = ⟪circ (tipParam q.1), circ (tipParam j)⟫ := by
  unfold ctipScore
  rw [cdoc_slot_zero hm, cdoc_slot_one hm]
  obtain ⟨h0, h1⟩ := tipParam_mem q.1
  obtain ⟨h0', h1'⟩ := tipParam_mem j
  have hg := inner_circ_nonneg h0 h1 h0' h1'
  rcases q with ⟨i, s⟩
  cases s
  · simp only [cquery, antiQuery, querySign, Bool.false_eq_true, ↓reduceIte, neg_smul, one_smul,
      inner_neg_left, inner_neg_right, neg_neg, tipDir]
    exact max_eq_right (by linarith)
  · simp only [cquery, antiQuery, querySign, ↓reduceIte, one_smul, inner_neg_right, tipDir]
    exact max_eq_left (by linarith)

theorem tip_inner_lt_one {k : ℕ} {i j : Fin k} (hij : j ≠ i) :
    ⟪circ (tipParam i), circ (tipParam j)⟫ < 1 := by
  have hk := kpos_of i
  obtain ⟨h0, h1⟩ := tipParam_mem i
  obtain ⟨h0', h1'⟩ := tipParam_mem j
  have hsep : 1 / (k : ℝ) ≤ |tipParam i - tipParam j| := by
    have h := nat_sep (fun h : i.val = j.val => hij (Fin.ext h.symm))
    unfold tipParam
    rw [← sub_div, abs_div, abs_of_pos hk]
    exact div_le_div_of_nonneg_right h hk.le
  have hb := inner_circ_le_of_sep h0 h1 h0' h1' (by positivity) hsep
  have : 0 < (1 / (k : ℝ)) ^ 2 / 2 := by positivity
  linarith

theorem circle_twoTip_rank_eq_one {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    FlatScan.rank (negCTip hm (cquery q)) q.1 = 1 := by
  apply FlatScan.rank_first
  intro j hj
  simp only [negCTip, ctipScore_query, inner_circ_self]
  linarith [tip_inner_lt_one (i := q.1) hj]

/-- **Two-tip exact recovery in `ℝ²`**: probe budget `1` returns every answer. -/
theorem circle_twoTip_returns_all {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    FlatScan.Returns (fun x : Fin k × Fin m => -⟪cquery q, cdoc x⟫)
      (negCTip hm (cquery q)) 1 (targetDoc hm q) := by
  refine (FlatScan.returns_iff _ 1 (targetDoc hm q) (fun y hy => ?_)
    (circle_twoTip_rank_eq_one hm q)).mpr le_rfl
  have h := circle_query_margin hm q y hy
  have : 0 < 1 / (32 * (k : ℝ) ^ 2) := by
    have := one_le_k q.1
    positivity
  linarith

theorem circle_twoTip_probed_card {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    (FlatScan.probed (negCTip (k := k) hm (cquery q)) 1).card = 1 := by
  have hfilt : FlatScan.probed (negCTip hm (cquery q)) 1 = {q.1} := by
    ext j
    simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_singleton]
    constructor
    · intro hr
      by_contra hne
      have hlt : negCTip hm (cquery q) q.1 < negCTip hm (cquery q) j := by
        simp only [negCTip, ctipScore_query, inner_circ_self]
        linarith [tip_inner_lt_one (i := q.1) hne]
      have hpos : 1 ≤ (Finset.univ.filter (fun l =>
          negCTip hm (cquery q) l < negCTip hm (cquery q) j)).card :=
        Finset.card_pos.mpr ⟨q.1, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hlt⟩⟩
      unfold FlatScan.rank at hr
      omega
    · rintro rfl
      rw [circle_twoTip_rank_eq_one hm q]
  rw [hfilt, Finset.card_singleton]

/-- Two-tip total work in `ℝ²`, summed over all `2 k` queries: `2 k (2 k + m)`. -/
theorem circle_twoTip_sum_work {k m : ℕ} (hm : 2 ≤ m) :
    ∑ q : Fin k × Bool, twoTipWork (Fin m) (negCTip (k := k) hm (cquery q)) 1 = 2 * k * (2 * k + m) := by
  have h (q : Fin k × Bool) : twoTipWork (Fin m) (negCTip (k := k) hm (cquery q)) 1 = 2 * k + m := by
    simp [twoTipWork, FlatScan.candidates, circle_twoTip_probed_card hm q]
  simp [h, mul_comm]

/-! ### Part 5: every one-vector router still pays `Ω(k)` in `ℝ²` -/

/-- **Average-rank lower bound in `ℝ²`**: `k (k + 3) ≤ 2 ∑ rank`. -/
theorem circle_affine_sum_tieRank_ge {k : ℕ} (r : Fin k → Plane) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] :
    k * (k + 3) ≤ 2 * ∑ q : Fin k × Bool, tieRank (affScore r b (cquery q)) lt q.1 :=
  antipodal_sum_tieRank_ge tipDir r b lt

/-- **Full-recall budget bound in `ℝ²`**: `k + 1 ≤ 2 p`. -/
theorem circle_affine_fullRecall_budget_ge {k : ℕ} (hk : 1 ≤ k) (r : Fin k → Plane)
    (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] {p : ℕ}
    (hall : ∀ q : Fin k × Bool, tieRank (affScore r b (cquery q)) lt q.1 ≤ p) :
    k + 1 ≤ 2 * p :=
  antipodal_fullRecall_budget_ge hk tipDir r b lt hall

/-- **Work lower bound in `ℝ²`**, even with an oracle per-query budget. -/
theorem circle_affine_sum_work_ge {k m : ℕ} (hm : 2 ≤ m)
    (r : Fin k → Plane) (b : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪cquery q, cdoc x⟫)
        (affScore r b (cquery q)) (budget q) (targetDoc hm q)) :
    4 * k ^ 2 + m * (k * (k + 3)) ≤
      2 * ∑ q : Fin k × Bool, FlatScan.work (Fin m) (affScore r b (cquery q)) (budget q) := by
  have hsum :
      (∑ q : Fin k × Bool, (k + m * minProbedCount (affScore r b (cquery q)) q.1)) ≤
      ∑ q : Fin k × Bool, FlatScan.work (Fin m) (affScore r b (cquery q)) (budget q) :=
    Finset.sum_le_sum fun q _ => by
      have h := work_ge_of_returns' (hret q)
      rw [show (targetDoc hm q).1 = q.1 from rfl] at h
      exact h
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have hk_sum : (∑ _q : Fin k × Bool, k) = 2 * k ^ 2 := by
    simp [Fintype.card_prod]
    ring
  rw [hk_sum] at hsum
  have hmp := antipodal_sum_minProbedCount_ge tipDir r b
  have hmul : m * (k * (k + 3)) ≤
      m * (2 * ∑ q : Fin k × Bool, minProbedCount (affScore r b (cquery q)) q.1) :=
    Nat.mul_le_mul_left m hmp
  change m * (k * (k + 3)) ≤
      m * (2 * ∑ q : Fin k × Bool, minProbedCount (affScore r b (antiQuery tipDir q)) q.1)
    at hmul
  change 2 * k ^ 2 + m * ∑ q : Fin k × Bool,
      minProbedCount (affScore r b (antiQuery tipDir q)) q.1 ≤ _ at hsum
  linarith

/-- Power-diagram corollary of `circle_affine_sum_work_ge`. -/
theorem circle_powerDist_sum_work_ge {k m : ℕ} (hm : 2 ≤ m)
    (c : Fin k → Plane) (w : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪cquery q, cdoc x⟫)
        (powScore c w (cquery q)) (budget q) (targetDoc hm q)) :
    4 * k ^ 2 + m * (k * (k + 3)) ≤
      2 * ∑ q : Fin k × Bool, FlatScan.work (Fin m) (powScore c w (cquery q)) (budget q) := by
  have heq (q : Fin k × Bool) :
      powScore c w (cquery q) =
        affScore (fun l => (2 : ℝ) • c l) (fun l => w l - ‖c l‖ ^ 2 - 1) (cquery q) := by
    funext j
    exact powScore_eq_affScore c w (cquery_unit q) j
  have h := circle_affine_sum_work_ge hm (fun l => (2 : ℝ) • c l)
    (fun l => w l - ‖c l‖ ^ 2 - 1) budget (fun q => by rw [← heq q]; exact hret q)
  simpa [heq] using h

/-- **Planar separation**: in `ℝ²`, two tips cost `2 k (2 k + m)` in total
while any one-vector affine router that returns every answer costs at least
`(4 k² + m k (k + 3)) / 2`. -/
theorem circle_affine_vs_twoTip {k m : ℕ} (hm : 2 ≤ m)
    (r : Fin k → Plane) (b : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪cquery q, cdoc x⟫)
        (affScore r b (cquery q)) (budget q) (targetDoc hm q)) :
    (∀ q : Fin k × Bool, FlatScan.Returns (fun x : Fin k × Fin m => -⟪cquery q, cdoc x⟫)
      (negCTip hm (cquery q)) 1 (targetDoc hm q)) ∧
    (∑ q : Fin k × Bool, twoTipWork (Fin m) (negCTip (k := k) hm (cquery q)) 1 =
      2 * k * (2 * k + m)) ∧
    (4 * k ^ 2 + m * (k * (k + 3)) ≤
      2 * ∑ q : Fin k × Bool, FlatScan.work (Fin m) (affScore r b (cquery q)) (budget q)) :=
  ⟨circle_twoTip_returns_all hm, circle_twoTip_sum_work hm,
    circle_affine_sum_work_ge hm r b budget hret⟩

/-! ### Part 6: the bound is attained in `ℝ²` -/

/-- Growth base `T = 4k + 2` of the planar witness. -/
def wBase (k : ℕ) : ℝ := 4 * k + 2

/-- Planar witness vector `r_j = 4k T^j u(j/k)^⊥`. -/
def circWVec {k : ℕ} (j : Fin k) : Plane := (4 * (k : ℝ) * wBase k ^ j.val) • perp (tipParam j)

/-- Planar witness bias `b_j = T^j`. -/
def circWBias {k : ℕ} (j : Fin k) : ℝ := wBase k ^ j.val

/-- Negated planar witness score at query `q`. -/
def circWScore {k : ℕ} (q : Fin k × Bool) : Fin k → ℝ := affScore circWVec circWBias (cquery q)

theorem inner_circ_perp_self (t : ℝ) : ⟪circ t, perp t⟫ = 0 := by
  rw [inner_circ_perp]; ring

theorem inner_circ_perp_le {t t' : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (h0' : 0 ≤ t') (h1' : t' ≤ 1)
    (hlt : t ≤ t') : ⟪circ t, perp t'⟫ ≤ -(t' - t) / 2 := by
  rw [inner_circ_perp_eq]
  have hD : 0 < (1 + t ^ 2) * (1 + t' ^ 2) := by positivity
  have ht2 : t ^ 2 ≤ 1 := by nlinarith
  have ht2' : t' ^ 2 ≤ 1 := by nlinarith
  have hD4 : (1 + t ^ 2) * (1 + t' ^ 2) ≤ 4 := by
    nlinarith [mul_le_mul ht2 ht2' (sq_nonneg t') zero_le_one]
  rw [div_le_iff₀ hD]
  have hd : 0 ≤ t' - t := by linarith
  have hgap : 0 ≤ 4 + 4 * (t * t') - (1 + t ^ 2) * (1 + t' ^ 2) := by
    nlinarith [mul_nonneg h0 h0']
  nlinarith [mul_nonneg hd hgap]

theorem circWScore_apply {k : ℕ} (q : Fin k × Bool) (l : Fin k) :
    circWScore q l = -(querySign q.2 * (4 * (k : ℝ) * wBase k ^ l.val) *
      ⟪circ (tipParam q.1), perp (tipParam l)⟫ + wBase k ^ l.val) := by
  simp only [circWScore, affScore, cquery, antiQuery, tipDir, circWVec, circWBias,
    real_inner_smul_left, real_inner_smul_right]
  ring

/-- On `±u(i/k)`, a list `l ≠ i` beats list `i` strictly when `i < l` and the
query is `-u(i/k)`, and loses strictly otherwise. -/
theorem circW_cmp {k : ℕ} (q : Fin k × Bool) {l : Fin k} (hl : l ≠ q.1) :
    (q.1 < l ∧ false = q.2 → circWScore q l < circWScore q q.1) ∧
    (¬ (q.1 < l ∧ false = q.2) → circWScore q q.1 < circWScore q l) := by
  rcases q with ⟨i, s⟩
  simp only at hl ⊢
  rw [circWScore_apply, circWScore_apply, inner_circ_perp_self, mul_zero, zero_add]
  have hk1 := one_le_k i
  have hT1 : (1 : ℝ) ≤ wBase k := by unfold wBase; linarith
  have hPl : 0 < wBase k ^ l.val := by positivity
  set σ := ⟪circ (tipParam i), perp (tipParam l)⟫ with hσ
  rcases lt_or_gt_of_ne hl with hli | hil
  · -- `l < i`: list `l` loses on both queries.
    have hcond : ¬ (i < l ∧ false = s) := fun h => absurd h.1 (lt_asymm hli)
    refine ⟨fun h => absurd h hcond, fun _ => ?_⟩
    have hPi : wBase k * wBase k ^ l.val ≤ wBase k ^ i.val := by
      rw [← pow_succ']
      exact pow_le_pow_right₀ hT1 (by exact_mod_cast hli)
    have hσ1 : querySign s * σ ≤ 1 := by
      have := abs_inner_circ_perp_le (tipParam i) (tipParam l)
      rw [← hσ] at this
      cases s <;> simp only [querySign, Bool.false_eq_true, ↓reduceIte] <;>
        linarith [abs_le.mp this]
    have hA : 0 ≤ 4 * (k : ℝ) * wBase k ^ l.val := by positivity
    have : querySign s * (4 * (k : ℝ) * wBase k ^ l.val) * σ ≤ 4 * (k : ℝ) * wBase k ^ l.val := by
      have := mul_le_mul_of_nonneg_left hσ1 hA
      linarith
    have hexp : wBase k * wBase k ^ l.val =
        4 * (k : ℝ) * wBase k ^ l.val + 2 * wBase k ^ l.val := by
      unfold wBase; ring
    linarith
  · -- `i < l`: `4 k σ ≤ -2`.
    obtain ⟨h0, h1⟩ := tipParam_mem i
    obtain ⟨h0', h1'⟩ := tipParam_mem l
    have hk := kpos_of i
    have hτ : tipParam i ≤ tipParam l := by
      unfold tipParam
      exact div_le_div_of_nonneg_right (by exact_mod_cast hil.le) hk.le
    have hσle := inner_circ_perp_le h0 h1 h0' h1' hτ
    rw [← hσ] at hσle
    have hgap : 1 ≤ (k : ℝ) * (tipParam l - tipParam i) := by
      unfold tipParam
      rw [← sub_div, mul_div_cancel₀ _ hk.ne']
      have : (i.val : ℝ) + 1 ≤ l.val := by exact_mod_cast hil
      linarith
    have h4 : 4 * (k : ℝ) * σ ≤ -2 := by nlinarith
    have hPi : wBase k ^ i.val ≤ wBase k ^ l.val :=
      pow_le_pow_right₀ hT1 (by exact_mod_cast hil.le)
    cases s
    · refine ⟨fun _ => ?_, fun h => absurd ⟨hil, rfl⟩ h⟩
      simp only [querySign, Bool.false_eq_true, ↓reduceIte]
      nlinarith
    · have hcond : ¬ (i < l ∧ false = true) := fun h => Bool.false_ne_true h.2
      refine ⟨fun h => absurd h hcond, fun _ => ?_⟩
      simp only [querySign, ↓reduceIte]
      have hPi0 : 0 < wBase k ^ i.val := by positivity
      nlinarith

/-- Exact `tieRank` of the planar witness for every tie-breaking order:
list `l > i` beats list `i` exactly on `-u(i/k)`. -/
theorem circW_tieRank {k : ℕ} (lt : Fin k → Fin k → Prop) [IsIrrefl (Fin k) lt]
    (q : Fin k × Bool) :
    tieRank (circWScore q) lt q.1 = 1 + witnessBeatCount (fun _ _ => false) q := by
  unfold tieRank witnessBeatCount
  congr 2
  ext l
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, beats]
  by_cases hl : l = q.1
  · subst hl
    simp only [lt_irrefl, false_or, true_and, false_and, iff_false]
    exact irrefl_of lt _
  · obtain ⟨hyes, hno⟩ := circW_cmp q hl
    by_cases hc : q.1 < l ∧ false = q.2
    · exact ⟨fun _ => hc, fun _ => Or.inl (hyes hc)⟩
    · have h := hno hc
      refine ⟨fun hb => ?_, fun h' => absurd h' hc⟩
      rcases hb with hb | ⟨hb, _⟩
      · exact absurd (lt_trans hb h) (lt_irrefl _)
      · exact absurd hb (ne_of_gt h)

/-- **The `(k + 3)/4` bound is tight in `ℝ²`**: the planar witness router has
`2 ∑ q, tieRank = k (k + 3)` for every tie-breaking order. -/
theorem circle_witness_sum_tieRank_eq {k : ℕ} (lt : Fin k → Fin k → Prop)
    [IsIrrefl (Fin k) lt] :
    2 * ∑ q : Fin k × Bool,
      tieRank (affScore circWVec circWBias (cquery q)) lt q.1 = k * (k + 3) := by
  have h := witness_sum_one_add (k := k) (fun _ _ => false)
  rw [← h]
  congr 2
  funext q
  exact circW_tieRank lt q

end IndexRecall.OneVectorCircle
