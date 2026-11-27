import IndexRecall.Geometry
import Mathlib.Data.Finset.Card

/-!
# Exact Recall@2 for two posting lists of size two

The true top two change at the midpoints of the nonadjacent pairs `a,c` and
`b,d`. A single probe scans two candidates, so recall is the cardinality of
the intersection with the true top two, divided by two.
-/

noncomputable section

namespace IndexRecall

/-- The two nearest sites, with smaller coordinates winning distance ties. -/
def Four.trueTopTwo (D : Four) (q : ℝ) : Finset ℝ :=
  if q ≤ mid D.a D.c then {D.a, D.b} else
  if q ≤ mid D.b D.d then {D.b, D.c} else {D.c, D.d}

/-- There are two selected sites, and each precedes every excluded site in
squared-distance order, breaking ties by coordinate. -/
theorem Four.trueTopTwo_correct (D : Four) (q : ℝ) :
    (D.trueTopTwo q).card = 2 ∧ (∀ x ∈ D.trueTopTwo q, D.IsSite x) ∧
      ∀ x ∈ D.trueTopTwo q, ∀ y, D.IsSite y → y ∉ D.trueTopTwo q →
        (q - x) ^ 2 ≤ (q - y) ^ 2 ∧ ((q - x) ^ 2 = (q - y) ^ 2 → x ≤ y) := by
  classical
  refine ⟨?_, ?_, ?_⟩
  · unfold Four.trueTopTwo
    split_ifs <;> simp [D.ab.ne, D.bc.ne, D.cd.ne]
  · unfold Four.trueTopTwo
    split_ifs <;> simp [Four.IsSite]
  · intro x hx y hy hout
    have hab := D.ab
    have hbc := D.bc
    have hcd := D.cd
    unfold Four.trueTopTwo at hx hout
    split_ifs at hx hout
    all_goals
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx hout
      rcases hx with rfl | rfl <;> rcases hy with rfl | rfl | rfl | rfl
      all_goals first
        | exact (hout (Or.inl rfl)).elim
        | exact (hout (Or.inr rfl)).elim
        | apply prefer_left <;> simp only [mid] at * <;> linarith
        | apply prefer_right <;> simp only [mid] at * <;> linarith

/-- Probing a single balanced posting list returns its two sites. -/
def Four.balancedProbe (D : Four) (boundary q : ℝ) : Finset ℝ :=
  if q ≤ boundary then {D.a, D.b} else {D.c, D.d}

theorem Four.balancedProbe_card (D : Four) (boundary q : ℝ) :
    (D.balancedProbe boundary q).card = 2 := by
  unfold Four.balancedProbe
  split_ifs <;> simp [D.ab.ne, D.cd.ne]

def Four.balancedRecallAtTwo (D : Four) (boundary q : ℝ) : ℝ :=
  ((D.trueTopTwo q ∩ D.balancedProbe boundary q).card : ℝ) / 2

/-- On the middle interval either probe recovers one true neighbor. On each
outer interval the probe recovers both true neighbors or neither. -/
theorem Four.balanced_recallAtTwo (D : Four) (boundary q : ℝ) :
    D.balancedRecallAtTwo boundary q =
      if q ≤ mid D.a D.c then (if q ≤ boundary then 1 else 0) else
      if q ≤ mid D.b D.d then 1 / 2 else (if q ≤ boundary then 0 else 1) := by
  classical
  have hab := D.ab
  have hbc := D.bc
  have hcd := D.cd
  unfold Four.balancedRecallAtTwo Four.trueTopTwo Four.balancedProbe
  split_ifs <;> simp (disch := grind)

/-- A switch improves recall precisely when it sends an outer-interval query
to the posting list containing both of its true neighbors. -/
theorem Four.balanced_recallAtTwo_improves_iff (D : Four) (old new q : ℝ) :
    D.balancedRecallAtTwo old q < D.balancedRecallAtTwo new q ↔
      (q ≤ mid D.a D.c ∧ old < q ∧ q ≤ new) ∨
      (mid D.b D.d < q ∧ q ≤ old ∧ new < q) := by
  have hcuts : mid D.a D.c < mid D.b D.d := by
    unfold mid
    linarith [D.ab, D.cd]
  rw [D.balanced_recallAtTwo old q, D.balanced_recallAtTwo new q]
  split_ifs <;> simp_all <;> grind

/-- If the centers lie in the convex hulls of their respective balanced
lists, changing their positions leaves Recall@2 unchanged at every query. -/
theorem Four.balanced_recallAtTwo_centers (D : Four) (s t q : ℝ)
    (has : D.a ≤ s) (hsb : s ≤ D.b) (hct : D.c ≤ t) (htd : t ≤ D.d) :
    D.balancedRecallAtTwo (mid s t) q =
      if q ≤ mid D.a D.c then 1 else if q ≤ mid D.b D.d then 1 / 2 else 1 := by
  rw [D.balanced_recallAtTwo]
  split_ifs <;> first | rfl | unfold mid at *; linarith

end IndexRecall
