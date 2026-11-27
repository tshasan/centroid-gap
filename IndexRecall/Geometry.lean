import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Tauto

/-!
# Nearest-neighbor geometry for four ordered sites

At each midpoint the smaller site wins. With the same tie rule for routing,
the queries whose nearest neighbor lies outside the probed list form an
interval open on the left and closed on the right.
-/

noncomputable section

namespace IndexRecall

def mid (x y : ℝ) : ℝ := (x + y) / 2

/-- For ordered centers, the left center wins precisely up to their midpoint. -/
theorem dist_le_iff (x y q : ℝ) (h : x < y) :
    (q - x) ^ 2 ≤ (q - y) ^ 2 ↔ q ≤ mid x y := by
  have factor : (q - y) ^ 2 - (q - x) ^ 2 = (y - x) * (x + y - 2 * q) := by ring
  rw [← sub_nonneg, factor, mul_nonneg_iff_of_pos_left (sub_pos.mpr h)]
  unfold mid
  constructor <;> intro hq <;> linarith

structure Four where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  ab : a < b
  bc : b < c
  cd : c < d

/-- The number of sites in the left list of a nonempty two-list partition. -/
inductive Cut | one | two | three deriving DecidableEq

def Four.cut (D : Four) (b : ℝ) : Cut :=
  if b < D.b then .one else if b < D.c then .two else .three

/-- The true-neighbor boundary between the two adjacent lists. -/
def Four.edge (D : Four) : Cut → ℝ
  | .one => mid D.a D.b
  | .two => mid D.b D.c
  | .three => mid D.c D.d

def Four.IsSite (D : Four) (x : ℝ) : Prop :=
  x = D.a ∨ x = D.b ∨ x = D.c ∨ x = D.d

/-- The globally nearest site, choosing the smaller coordinate at a tie. -/
def Four.nearest (D : Four) (q : ℝ) : ℝ :=
  if q ≤ mid D.a D.b then D.a else
  if q ≤ mid D.b D.c then D.b else
  if q ≤ mid D.c D.d then D.c else D.d

theorem Four.nearest_isSite (D : Four) (q : ℝ) : D.IsSite (D.nearest q) := by
  unfold Four.nearest
  split_ifs <;> simp [Four.IsSite]

theorem prefer_left {x y q : ℝ} (hxy : x < y) (hq : q ≤ mid x y) :
    (q - x) ^ 2 ≤ (q - y) ^ 2 ∧ ((q - x) ^ 2 = (q - y) ^ 2 → x ≤ y) :=
  ⟨(dist_le_iff x y q hxy).2 hq, fun _ => hxy.le⟩

theorem prefer_right {x y q : ℝ} (hxy : x < y) (hq : mid x y < q) :
    (q - y) ^ 2 ≤ (q - x) ^ 2 ∧ ((q - y) ^ 2 = (q - x) ^ 2 → y ≤ x) := by
  have hlt : (q - y) ^ 2 < (q - x) ^ 2 :=
    lt_of_not_ge (fun h => hq.not_ge ((dist_le_iff x y q hxy).1 h))
  exact ⟨hlt.le, fun he => (hlt.ne he).elim⟩

/-- Global squared-distance optimality with the leftmost choice at equality. -/
theorem Four.nearest_min (D : Four) (q x : ℝ) (hx : D.IsSite x) :
    (q - D.nearest q) ^ 2 ≤ (q - x) ^ 2 ∧
      ((q - D.nearest q) ^ 2 = (q - x) ^ 2 → D.nearest q ≤ x) := by
  have hab := D.ab
  have hbc := D.bc
  have hcd := D.cd
  unfold Four.nearest
  split_ifs
  all_goals
    rcases hx with rfl | rfl | rfl | rfl
    all_goals first
      | exact ⟨le_rfl, fun _ => le_rfl⟩
      | apply prefer_left <;> simp only [mid] at * <;> linarith
      | apply prefer_right <;> simp only [mid] at * <;> linarith

/-- Both sites and queries on the boundary route to the left list. A probe
hits when the query and its global nearest site route to the same list. -/
def Hit (D : Four) (b q : ℝ) : Prop := q ≤ b ↔ D.nearest q ≤ b

def InGap (b h q : ℝ) : Prop := min b h < q ∧ q ≤ max b h

/-- At any split with two nonempty lists, the true nearest site belongs to
the left list exactly up to the midpoint separating the adjacent sites. -/
theorem Four.nearest_le_iff (D : Four) (b q : ℝ) (ha : D.a ≤ b) (hd : b < D.d) :
    D.nearest q ≤ b ↔ q ≤ D.edge (D.cut b) := by
  unfold Four.nearest Four.cut
  split_ifs <;> simp only [Four.edge, mid] at *
  all_goals constructor <;> intro h <;> linarith [D.ab, D.bc, D.cd]

/-- Exact misses, with the endpoint conventions induced by left ties. -/
theorem miss_iff (D : Four) (b q : ℝ) (ha : D.a ≤ b) (hd : b < D.d) :
    ¬ Hit D b q ↔ InGap b (D.edge (D.cut b)) q := by
  rw [Hit, D.nearest_le_iff b q ha hd, InGap, min_lt_iff, le_max_iff]
  simp only [← not_le]
  tauto

end IndexRecall
