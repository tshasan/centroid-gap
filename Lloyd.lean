import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Sharp approximation ratio for weighted Lloyd's algorithm on three real points

For weights in `[1, R]`, the greatest fixed-point cost divided by the global
two-center optimum is `B R = (1 + R) ^ 2` (`sharp_ratio`). The same factor can
be approached by a strict run starting at two input points (`reachable_sharpness`).

`LloydStep` checks ordinary distances and updates both weighted means.
`optCost_isLeast` minimizes the squared-distance objective over arbitrary real
centers. The fixed-point relation allows coincident centers and either choice at a tie.
-/

noncomputable section

namespace Lloyd

structure Point where
  x : ℝ
  w : ℝ

def InRange (R : ℝ) (p : Point) : Prop := 1 ≤ p.w ∧ p.w ≤ R

theorem InRange.pos {R : ℝ} {p : Point} (h : InRange R p) : 0 < p.w :=
  lt_of_lt_of_le zero_lt_one h.1

def B (R : ℝ) : ℝ := (1 + R) ^ 2

def mean (p q : Point) : ℝ := (p.w * p.x + q.w * q.x) / (p.w + q.w)

def pairCost (p q : Point) : ℝ := p.w * q.w / (p.w + q.w) * (p.x - q.x) ^ 2

theorem mean_comm (p q : Point) : mean p q = mean q p := by
  unfold mean
  rw [add_comm (p.w * p.x), add_comm p.w]

theorem pairCost_comm (p q : Point) : pairCost p q = pairCost q p := by
  unfold pairCost
  rw [add_comm p.w q.w]
  ring

def loss (p : Point) (s t : ℝ) : ℝ := p.w * min ((p.x - s) ^ 2) ((p.x - t) ^ 2)

def objective (p q r : Point) (s t : ℝ) : ℝ :=
  loss p s t + loss q s t + loss r s t

/-- The three ways to share one center. Its global optimality is proved in
`optCost_isLeast`. -/
def optCost (p q r : Point) : ℝ :=
  min (pairCost p q) (min (pairCost p r) (pairCost q r))

/-- Stability of the pair `p,q` with the singleton `r`. The singleton is
already at distance zero from its center. -/
structure PairFixed (p q r : Point) : Prop where
  left : (p.x - mean p q) ^ 2 ≤ (p.x - r.x) ^ 2
  right : (q.x - mean p q) ^ 2 ≤ (q.x - r.x) ^ 2

/-- All three nearest-center decisions are strict. -/
structure StrictPair (p q r : Point) (s t : ℝ) : Prop where
  left : (p.x - s) ^ 2 < (p.x - t) ^ 2
  right : (q.x - s) ^ 2 < (q.x - t) ^ 2
  single : (r.x - t) ^ 2 < (r.x - s) ^ 2

/-! ## Assignments and Lloyd updates

Each Boolean selects one of the two centers. A step computes both means from
the same assignment that passed the three nearest-center comparisons.
-/

structure Assignment where
  first : Bool
  second : Bool
  third : Bool

/-- `true` selects the first center and `false` selects the second. -/
def Assignment.usesBoth (g : Assignment) : Prop :=
  (g.first = true ∨ g.second = true ∨ g.third = true) ∧
  (g.first = false ∨ g.second = false ∨ g.third = false)

def chooses (cmp : ℝ → ℝ → Prop) (side : Bool) (p : Point) (s t : ℝ) : Prop :=
  if side then cmp |p.x - s| |p.x - t| else cmp |p.x - t| |p.x - s|

def Assignment.follows (g : Assignment) (cmp : ℝ → ℝ → Prop)
    (p q r : Point) (s t : ℝ) : Prop :=
  chooses cmp g.first p s t ∧ chooses cmp g.second q s t ∧ chooses cmp g.third r s t

/-- Weighted mean of exactly the points assigned to `side`. -/
def Assignment.center (g : Assignment) (side : Bool) (p q r : Point) : ℝ :=
  let sum (f : Point → ℝ) :=
    (if g.first = side then f p else 0) +
    (if g.second = side then f q else 0) +
    (if g.third = side then f r else 0)
  sum (fun u => u.w * u.x) / sum Point.w

def StepWith (cmp : ℝ → ℝ → Prop) (p q r : Point) (s t s' t' : ℝ) : Prop :=
  ∃ g : Assignment, g.usesBoth ∧ g.follows cmp p q r s t ∧
    s' = g.center true p q r ∧ t' = g.center false p q r

/-- Either choice at a tie is a legal assignment. -/
abbrev LloydStep := StepWith (· ≤ ·)

/-- Every point is strictly closer to the center it selects. -/
abbrev StrictStep := StepWith (· < ·)

def FixedPoint (p q r : Point) (s t : ℝ) : Prop := LloydStep p q r s t s t

theorem StrictStep.toLloydStep {p q r : Point} {s t s' t' : ℝ}
    (h : StrictStep p q r s t s' t') : LloydStep p q r s t s' t' := by
  obtain ⟨g, used, near, hs, ht⟩ := h
  have weaken (side : Bool) (u : Point) (hu : chooses (· < ·) side u s t) :
      chooses (· ≤ ·) side u s t := by cases side <;> exact le_of_lt hu
  exact ⟨g, used, ⟨weaken _ _ near.1, weaken _ _ near.2.1, weaken _ _ near.2.2⟩, hs, ht⟩

theorem singleton_mean {p : Point} (h : 0 < p.w) : p.w * p.x / p.w = p.x := by
  field_simp [h.ne']

theorem StrictPair.step {p q r : Point} {s t : ℝ}
    (h : StrictPair p q r s t) (hr : 0 < r.w) :
    StrictStep p q r s t (mean p q) r.x := by
  refine ⟨⟨true, true, false⟩, by simp [Assignment.usesBoth],
    ⟨sq_lt_sq.mp h.left, sq_lt_sq.mp h.right, sq_lt_sq.mp h.single⟩, ?_, ?_⟩
  · simp [Assignment.center, mean]
  · simp [Assignment.center, singleton_mean hr]

/-! ## The global optimum

Completing the square gives the minimum cost of fitting two points with one
center. For any two real centers, at least two points share a nearest center.
Conversely, the mean of any pair and the remaining point give candidate centers.
-/

/-- Weighted completion of the square, valid for every candidate center. -/
theorem pair_error_decomposition {p q : Point} (hp : 0 < p.w) (hq : 0 < q.w) (c : ℝ) :
    p.w * (p.x - c) ^ 2 + q.w * (q.x - c) ^ 2 =
      pairCost p q + (p.w + q.w) * (mean p q - c) ^ 2 := by
  unfold pairCost mean
  field_simp [ne_of_gt (add_pos hp hq)]
  ring

theorem pair_error_at_mean {p q : Point} (hp : 0 < p.w) (hq : 0 < q.w) :
    p.w * (p.x - mean p q) ^ 2 + q.w * (q.x - mean p q) ^ 2 = pairCost p q := by
  simpa using pair_error_decomposition hp hq (mean p q)

theorem pairCost_le_error {p q : Point} (hp : 0 < p.w) (hq : 0 < q.w) (c : ℝ) :
    pairCost p q ≤ p.w * (p.x - c) ^ 2 + q.w * (q.x - c) ^ 2 := by
  rw [pair_error_decomposition hp hq c]
  exact le_add_of_nonneg_right (mul_nonneg (add_pos hp hq).le (sq_nonneg _))

private theorem loss_nonneg {p : Point} (hp : 0 ≤ p.w) (s t : ℝ) : 0 ≤ loss p s t :=
  mul_nonneg hp (le_min (sq_nonneg _) (sq_nonneg _))

private theorem loss_cases (p : Point) (s t : ℝ) :
    loss p s t = p.w * (p.x - s) ^ 2 ∨ loss p s t = p.w * (p.x - t) ^ 2 := by
  rcases le_total ((p.x - s) ^ 2) ((p.x - t) ^ 2) with h | h
  · exact Or.inl (by simp [loss, min_eq_left h])
  · exact Or.inr (by simp [loss, min_eq_right h])

/-- Lower bound for arbitrary centers, including choices with an empty cell. -/
theorem optCost_le_objective {p q r : Point}
    (hp : 0 < p.w) (hq : 0 < q.w) (hr : 0 < r.w) (s t : ℝ) :
    optCost p q r ≤ objective p q r s t := by
  have pq (c : ℝ) : optCost p q r ≤ p.w * (p.x - c) ^ 2 + q.w * (q.x - c) ^ 2 :=
    (min_le_left _ _).trans (pairCost_le_error hp hq c)
  have pr (c : ℝ) : optCost p q r ≤ p.w * (p.x - c) ^ 2 + r.w * (r.x - c) ^ 2 :=
    ((min_le_right _ _).trans (min_le_left _ _)).trans (pairCost_le_error hp hr c)
  have qr (c : ℝ) : optCost p q r ≤ q.w * (q.x - c) ^ 2 + r.w * (r.x - c) ^ 2 :=
    ((min_le_right _ _).trans (min_le_right _ _)).trans (pairCost_le_error hq hr c)
  have hp₀ := loss_nonneg hp.le s t
  have hq₀ := loss_nonneg hq.le s t
  have hr₀ := loss_nonneg hr.le s t
  unfold objective
  rcases loss_cases p s t with hps | hpt <;> rcases loss_cases q s t with hqs | hqt
  · linarith only [hps, hqs, pq s, hr₀]
  · rcases loss_cases r s t with hrs | hrt
    · linarith only [hps, hrs, pr s, hq₀]
    · linarith only [hqt, hrt, qr t, hp₀]
  · rcases loss_cases r s t with hrs | hrt
    · linarith only [hqs, hrs, qr s, hp₀]
    · linarith only [hpt, hrt, pr t, hq₀]
  · linarith only [hpt, hqt, pq t, hr₀]

/-- The mean of a selected pair and the singleton are candidate centers. -/
theorem objective_at_pair_le {p q : Point} (hp : 0 < p.w) (hq : 0 < q.w) (r : Point) :
    objective p q r (mean p q) r.x ≤ pairCost p q := by
  have single : loss r (mean p q) r.x = 0 := by
    simp [loss, min_eq_right (sq_nonneg (r.x - mean p q))]
  unfold objective
  rw [single, add_zero, ← pair_error_at_mean hp hq]
  exact add_le_add (mul_le_mul_of_nonneg_left (min_le_left _ _) hp.le)
    (mul_le_mul_of_nonneg_left (min_le_left _ _) hq.le)

/-- A pair attaining the finite minimum supplies globally optimal centers. -/
theorem optCost_attained {p q r : Point} (hp : 0 < p.w) (hq : 0 < q.w) (hr : 0 < r.w) :
    ∃ s t, objective p q r s t = optCost p q r := by
  obtain ⟨s, t, hst⟩ : ∃ s t, objective p q r s t ≤ optCost p q r := by
    unfold optCost
    rcases le_total (pairCost p q) (min (pairCost p r) (pairCost q r)) with h | h
    · rw [min_eq_left h]
      exact ⟨_, _, objective_at_pair_le hp hq r⟩
    · rw [min_eq_right h]
      rcases le_total (pairCost p r) (pairCost q r) with hpr | hqr
      · rw [min_eq_left hpr]
        refine ⟨mean p r, q.x, ?_⟩
        simpa only [objective, add_assoc, add_left_comm, add_comm] using
          objective_at_pair_le hp hr q
      · rw [min_eq_right hqr]
        refine ⟨mean q r, p.x, ?_⟩
        simpa only [objective, add_assoc, add_left_comm, add_comm] using
          objective_at_pair_le hq hr p
  exact ⟨s, t, le_antisymm hst (optCost_le_objective hp hq hr s t)⟩

/-- The minimum is taken over exactly the objective values of real centers. -/
theorem optCost_isLeast {p q r : Point} (hp : 0 < p.w) (hq : 0 < q.w) (hr : 0 < r.w) :
    IsLeast {z : ℝ | ∃ s t, objective p q r s t = z} (optCost p q r) := by
  refine ⟨optCost_attained hp hq hr, ?_⟩
  rintro z ⟨s, t, rfl⟩
  exact optCost_le_objective hp hq hr s t

private theorem pairCost_pos_of_lt {p q : Point} (hp : 0 < p.w) (hq : 0 < q.w)
    (h : p.x < q.x) : 0 < pairCost p q := by
  exact mul_pos (div_pos (mul_pos hp hq) (add_pos hp hq)) (sq_pos_of_neg (sub_neg.mpr h))

/-- Distinct sorted coordinates make every candidate pair cost positive. -/
theorem optCost_pos {p q r : Point} (hp : 0 < p.w) (hq : 0 < q.w) (hr : 0 < r.w)
    (hpq : p.x < q.x) (hqr : q.x < r.x) : 0 < optCost p q r :=
  lt_min (pairCost_pos_of_lt hp hq hpq)
    (lt_min (pairCost_pos_of_lt hp hr (hpq.trans hqr)) (pairCost_pos_of_lt hq hr hqr))

/-! ## Upper bound for all fixed points

Two different pairs share a point. If `p` is shared, compare its distance to
the current mean of `p,q` with its distance to the singleton `r`. The weight
factor in the resulting cost comparison is `(1 + p.w/q.w) * (1 + p.w/r.w)`.
-/

/-- Compare the current pair with the competing pair that keeps `p`. -/
private theorem bound_at_shared_point {R : ℝ} {p q r : Point} (hR : 1 ≤ R)
    (hp : InRange R p) (hq : InRange R q) (hr : InRange R r)
    (hnear : (p.x - mean p q) ^ 2 ≤ (p.x - r.x) ^ 2) :
    pairCost p q ≤ B R * pairCost p r := by
  have hp₀ := hp.pos
  have hq₀ := hq.pos
  have hr₀ := hr.pos
  have weight_ratio_le (s : Point) (hs : InRange R s) : p.w / s.w ≤ R := by
    apply (div_le_iff₀ hs.pos).2
    exact hp.2.trans (by simpa only [mul_one] using
      mul_le_mul_of_nonneg_left hs.1 (zero_le_one.trans hR))
  calc
    pairCost p q = (p.w * (p.w + q.w) / q.w) * (p.x - mean p q) ^ 2 := by
      unfold pairCost mean
      field_simp [hq₀.ne', (add_pos hp₀ hq₀).ne']
      ring
    _ ≤ (p.w * (p.w + q.w) / q.w) * (p.x - r.x) ^ 2 :=
      mul_le_mul_of_nonneg_left hnear (by positivity)
    _ = ((1 + p.w / q.w) * (1 + p.w / r.w)) * pairCost p r := by
      unfold pairCost
      field_simp [hq₀.ne', hr₀.ne', (add_pos hp₀ hr₀).ne']
      ring
    _ ≤ B R * pairCost p r := by
      rw [B, pow_two]
      apply mul_le_mul_of_nonneg_right _ (by unfold pairCost; positivity)
      exact mul_le_mul (add_le_add (le_refl 1) (weight_ratio_le q hq))
        (add_le_add (le_refl 1) (weight_ratio_le r hr)) (by positivity) (by linarith)

/-- Every weakly stable pair costs at most `B R` times the minimum of the
three pair-plus-singleton costs, for any labeling of the input. -/
theorem pairFixed_bound {R : ℝ} {p q r : Point} (hR : 1 ≤ R)
    (hp : InRange R p) (hq : InRange R q) (hr : InRange R r)
    (hfix : PairFixed p q r) : pairCost p q ≤ B R * optCost p q r := by
  have hp₀ := hp.pos
  have hq₀ := hq.pos
  have hB : 1 ≤ B R := by unfold B; nlinarith [sq_nonneg R]
  have hself : pairCost p q ≤ B R * pairCost p q := by
    simpa only [one_mul] using
      mul_le_mul_of_nonneg_right hB (show 0 ≤ pairCost p q by unfold pairCost; positivity)
  have hleft := bound_at_shared_point hR hp hq hr hfix.left
  have hright := bound_at_shared_point hR hq hp hr
    (by rw [mean_comm q p]; exact hfix.right)
  rw [pairCost_comm q p] at hright
  have hB₀ := zero_le_one.trans hB
  rw [optCost, mul_min_of_nonneg _ _ hB₀, mul_min_of_nonneg _ _ hB₀]
  exact le_min hself (le_min hleft hright)

theorem PairFixed.objective_eq {p q r : Point} (h : PairFixed p q r)
    (hp : 0 < p.w) (hq : 0 < q.w) :
    objective p q r (mean p q) r.x = pairCost p q := by
  simpa [objective, loss, min_eq_left h.left, min_eq_left h.right,
    min_eq_right (sq_nonneg (r.x - mean p q))] using pair_error_at_mean hp hq

private theorem canonical_bound {R : ℝ} {p q r : Point} (hR : 1 ≤ R)
    (hp : InRange R p) (hq : InRange R q) (hr : InRange R r) (h : PairFixed p q r) :
    objective p q r (mean p q) r.x ≤ B R * optCost p q r := by
  rw [h.objective_eq hp.pos hq.pos]
  exact pairFixed_bound hR hp hq hr h

private theorem objective_swap (p q r : Point) (s t : ℝ) :
    objective p q r s t = objective p q r t s := by
  simp only [objective, loss, min_comm]

/-- Every fixed point of the assignment-and-update relation satisfies the
cost bound. This includes all nonempty tie assignments and coincident centers. -/
theorem fixedPoint_cost_le {R : ℝ} {p q r : Point} {s t : ℝ} (hR : 1 ≤ R)
    (hp : InRange R p) (hq : InRange R q) (hr : InRange R r)
    (h : FixedPoint p q r s t) : objective p q r s t ≤ B R * optCost p q r := by
  -- Transfer the pair bound to each of the three labelings.
  have pq := canonical_bound hR hp hq hr
  have pr (f : PairFixed p r q) :
      objective p q r (mean p r) q.x ≤ B R * optCost p q r := by
    simpa only [objective, optCost, pairCost_comm, min_comm, min_left_comm, min_assoc,
      add_comm, add_left_comm, add_assoc] using canonical_bound hR hp hr hq f
  have qr (f : PairFixed q r p) :
      objective p q r (mean q r) p.x ≤ B R * optCost p q r := by
    simpa only [objective, optCost, pairCost_comm, min_comm, min_left_comm, min_assoc,
      add_comm, add_left_comm, add_assoc] using canonical_bound hR hq hr hp f
  -- Of the eight assignments, six use both centers. Their updates compute
  -- the pair mean and the singleton coordinate, in either center order.
  obtain ⟨⟨i, j, k⟩, used, near, hs, ht⟩ := h
  cases i <;> cases j <;> cases k <;> simp [Assignment.usesBoth] at used
  all_goals
    simp [Assignment.center, singleton_mean hp.pos, singleton_mean hq.pos,
      singleton_mean hr.pos] at hs ht
    subst s
    subst t
  · rw [objective_swap]
    exact pq ⟨sq_le_sq.mpr near.1, sq_le_sq.mpr near.2.1⟩
  · rw [objective_swap]
    exact pr ⟨sq_le_sq.mpr near.1, sq_le_sq.mpr near.2.2⟩
  · exact qr ⟨sq_le_sq.mpr near.2.1, sq_le_sq.mpr near.2.2⟩
  · rw [objective_swap]
    exact qr ⟨sq_le_sq.mpr near.2.1, sq_le_sq.mpr near.2.2⟩
  · exact pr ⟨sq_le_sq.mpr near.1, sq_le_sq.mpr near.2.2⟩
  · exact pq ⟨sq_le_sq.mpr near.1, sq_le_sq.mpr near.2.1⟩

/-- The approximation ratio is well-defined on strictly ordered inputs. -/
theorem fixedPoint_ratio_le {R : ℝ} {p q r : Point} {s t : ℝ} (hR : 1 ≤ R)
    (hpq : p.x < q.x) (hqr : q.x < r.x)
    (hp : InRange R p) (hq : InRange R q) (hr : InRange R r)
    (h : FixedPoint p q r s t) : objective p q r s t / optCost p q r ≤ B R :=
  (div_le_iff₀ (optCost_pos hp.pos hq.pos hr.pos hpq hqr)).2
    (fixedPoint_cost_le hR hp hq hr h)

/-! ## Strict runs approaching the bound

Place the points at `-a, 0, 1` with weights `1, R, 1`. Starting at `0, 1`
groups the two left points together. Their updated center is `-a / (1 + R)`.
For `1 < a < 1 + R`, the assignments stay strict and the exact ratio is `a ^ 2`.
-/

/-- Every center pair `(-t, 1)` in this range gives the same strict assignments. -/
private theorem example_assignments {a w t : ℝ} (ha : 1 < a)
    (ht : 0 ≤ t) (ht' : t < 1) :
    StrictPair ⟨-a, 1⟩ ⟨0, w⟩ ⟨1, 1⟩ (-t) 1 := by
  have hsum : 0 < 1 + t := by linarith
  constructor
  · change (-a - -t) ^ 2 < (-a - 1) ^ 2
    nlinarith only [mul_pos hsum (show 0 < 2 * a + 1 - t by linarith)]
  · change (0 - -t) ^ 2 < (0 - 1) ^ 2
    nlinarith only [mul_pos (sub_pos.mpr ht') hsum]
  · change (1 - 1) ^ 2 < (1 - -t) ^ 2
    nlinarith only [sq_pos_of_pos hsum]

/-- Ordered inputs witnessing sharpness. The first strict assignment starts at
the two right input points. The second starts at the centers after the update,
and therefore repeats at every subsequent step. -/
theorem strict_sharpness {R ε : ℝ} (hR : 1 ≤ R) (hε : 0 < ε) :
    ∃ p q r : Point, p.x < q.x ∧ q.x < r.x ∧
      InRange R p ∧ InRange R q ∧ InRange R r ∧
      StrictPair p q r q.x r.x ∧ StrictPair p q r (mean p q) r.x ∧
      B R - ε < pairCost p q / optCost p q r := by
  -- Choose a positive offset d keeping a above 1 and its square within ε.
  let d := min (1 / 2 : ℝ) (ε / (2 * (1 + R)))
  have hden : 0 < 2 * (1 + R) := by linarith
  have hd : 0 < d := lt_min (by norm_num) (div_pos hε hden)
  have hdhalf : d ≤ 1 / 2 := min_le_left _ _
  have hdε : d * (2 * (1 + R)) ≤ ε :=
    (le_div_iff₀ hden).mp (min_le_right _ _)
  let a := 1 + R - d
  have ha : 1 < a := by dsimp only [a]; linarith
  have halim : a < 1 + R := by dsimp only [a]; linarith
  have hnear : B R - ε < a ^ 2 := by
    dsimp only [B, a]
    nlinarith only [hdε, sq_pos_of_pos hd]

  let p : Point := ⟨-a, 1⟩
  let q : Point := ⟨0, R⟩
  let r : Point := ⟨1, 1⟩
  have hsum : 0 < 1 + R := by linarith
  have hmean : mean p q = -(a / (1 + R)) := by
    simp [mean, p, q, neg_div]
  have hstart : StrictPair p q r q.x r.x := by
    simpa [p, q, r] using
      example_assignments (w := R) ha (t := 0) le_rfl zero_lt_one
  have hfixed : StrictPair p q r (mean p q) r.x := by
    rw [hmean]
    apply example_assignments ha
    · exact div_nonneg (by linarith) hsum.le
    · exact (div_lt_one hsum).mpr halim

  -- The common factor R / (1 + R) cancels from the cost ratio.
  let f := R / (1 + R)
  have hf : 0 < f := div_pos (by linarith) hsum
  have hf1 : f ≤ 1 := (div_le_iff₀ hsum).mpr (by linarith)
  have hleft : pairCost p q = f * a ^ 2 := by
    simp [p, q, f, pairCost]
  have hright : pairCost q r = f := by
    simp [q, r, f, pairCost, add_comm]
  have houter : pairCost p r = (a + 1) ^ 2 / 2 := by
    norm_num [p, r, pairCost]
    ring
  have hfleft : f ≤ f * a ^ 2 := by
    have ha2 : 1 ≤ a ^ 2 := by nlinarith only [ha, sq_nonneg (a - 1)]
    simpa using mul_le_mul_of_nonneg_left ha2 hf.le
  have hfouter : f ≤ (a + 1) ^ 2 / 2 := by
    apply hf1.trans
    nlinarith only [ha, sq_nonneg a]
  have hopt : optCost p q r = f := by
    rw [optCost, hleft, houter, hright, min_eq_right hfouter, min_eq_right hfleft]
  have hratio : pairCost p q / optCost p q r = a ^ 2 := by
    rw [hleft, hopt]
    exact (div_eq_iff hf.ne').mpr (mul_comm _ _)

  refine ⟨p, q, r, ?_, ?_, ⟨le_rfl, hR⟩, ⟨hR, le_rfl⟩, ⟨le_rfl, hR⟩,
    hstart, hfixed, ?_⟩
  · change -a < 0
    linarith
  · exact zero_lt_one
  · rwa [hratio]

/-- Starting at the two right input points gives a strict first step followed
by a strict self-step. Repeating that self-step yields the entire trajectory. -/
theorem reachable_sharpness {R ε : ℝ} (hR : 1 ≤ R) (hε : 0 < ε) :
    ∃ p q r : Point, p.x < q.x ∧ q.x < r.x ∧ InRange R p ∧ InRange R q ∧ InRange R r ∧
      StrictStep p q r q.x r.x (mean p q) r.x ∧
      StrictStep p q r (mean p q) r.x (mean p q) r.x ∧ FixedPoint p q r (mean p q) r.x ∧
      B R - ε < objective p q r (mean p q) r.x / optCost p q r := by
  obtain ⟨p, q, r, hpq, hqr, hp, hq, hr, start, finish, ratio⟩ := strict_sharpness hR hε
  have stable : PairFixed p q r := ⟨finish.left.le, finish.right.le⟩
  have step := finish.step hr.pos
  refine ⟨p, q, r, hpq, hqr, hp, hq, hr, start.step hr.pos, step, step.toLloydStep, ?_⟩
  rwa [stable.objective_eq hp.pos hq.pos]

/-! ## The exact maximum when ties are allowed -/

/-- Cost ratios for all nonempty fixed points on admissible inputs. -/
def fixedPointRatios (R : ℝ) : Set ℝ :=
  {z | ∃ (p q r : Point) (s t : ℝ), p.x < q.x ∧ q.x < r.x ∧
    InRange R p ∧ InRange R q ∧ InRange R r ∧ FixedPoint p q r s t ∧
    objective p q r s t / optCost p q r = z}

/-- With ties, the factor is attained exactly. At coordinates `-1, 0, R`
with weights `R, 1, 1`, the two outer points have mean zero. Assign them
to one center and the middle point to the other, both at zero. -/
theorem coincident_attains {R : ℝ} (hR : 1 ≤ R) :
    ∃ p q r : Point, p.x < q.x ∧ q.x < r.x ∧ InRange R p ∧ InRange R q ∧ InRange R r ∧
      FixedPoint p q r 0 0 ∧ objective p q r 0 0 / optCost p q r = B R := by
  let p : Point := ⟨-1, R⟩
  let q : Point := ⟨0, 1⟩
  let r : Point := ⟨R, 1⟩
  have hp : InRange R p := ⟨hR, le_rfl⟩
  have hq : InRange R q := ⟨le_rfl, hR⟩
  have hr : InRange R r := ⟨le_rfl, hR⟩
  have hpq : p.x < q.x := by norm_num [p, q]
  have hqr : q.x < r.x := lt_of_lt_of_le zero_lt_one hR
  have fixed : FixedPoint p q r 0 0 := by
    exact ⟨⟨true, false, true⟩, by simp [Assignment.usesBoth], ⟨le_rfl, le_rfl, le_rfl⟩,
      by simp [Assignment.center, p, r], by simp [Assignment.center, q]⟩
  have pair_eq : B R * pairCost p q = objective p q r 0 0 := by
    simp [B, pairCost, objective, loss, p, q, r]
    field_simp [show R + 1 ≠ 0 by linarith]
    ring
  -- This competing pair already gives the reverse of the universal bound.
  have reverse : B R * optCost p q r ≤ objective p q r 0 0 :=
    (mul_le_mul_of_nonneg_left (min_le_left _ _) (sq_nonneg _)).trans_eq pair_eq
  exact ⟨p, q, r, hpq, hqr, hp, hq, hr, fixed, (div_eq_iff
    (optCost_pos hp.pos hq.pos hr.pos hpq hqr).ne').2
      (le_antisymm (fixedPoint_cost_le hR hp hq hr fixed) reverse)⟩

/-- The greatest cost ratio of all admissible nonempty fixed points. -/
theorem sharp_ratio {R : ℝ} (hR : 1 ≤ R) : IsGreatest (fixedPointRatios R) (B R) := by
  obtain ⟨p, q, r, hpq, hqr, hp, hq, hr, fixed, ratio⟩ := coincident_attains hR
  refine ⟨⟨p, q, r, 0, 0, hpq, hqr, hp, hq, hr, fixed, ratio⟩, ?_⟩
  rintro z ⟨a, b, c, s, t, hab, hbc, ha, hb, hc, stable, rfl⟩
  exact fixedPoint_ratio_le hR hab hbc ha hb hc stable

end Lloyd

