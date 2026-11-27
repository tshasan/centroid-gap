import IndexRecall.Geometry
import Mathlib.Tactic.Linarith

/-!
# One weighted Lloyd round on four ordered points

The boundary is the midpoint of the two centers. A tie goes to the left center,
so the left cell includes its boundary. The endpoint conditions on `Centers`
ensure that both cells contain data. The next centers are the weighted means
of the two clusters, whose sizes are determined by `Four.cut`.
-/

namespace IndexRecall

noncomputable section

structure Weights where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  apos : 0 < a
  bpos : 0 < b
  cpos : 0 < c
  dpos : 0 < d

def Weights.unit : Weights := ⟨1, 1, 1, 1, zero_lt_one, zero_lt_one,
  zero_lt_one, zero_lt_one⟩

structure Centers (D : Four) where
  left : ℝ
  right : ℝ
  ordered : left < right
  first : D.a ≤ mid left right
  last : mid left right < D.d

def Centers.boundary {D : Four} (I : Centers D) : ℝ := mid I.left I.right

def mean2 (x y wx wy : ℝ) : ℝ := (wx * x + wy * y) / (wx + wy)

def mean3 (x y z wx wy wz : ℝ) : ℝ :=
  (wx * x + wy * y + wz * z) / (wx + wy + wz)

private theorem mean2_between {x y wx wy : ℝ} (hxy : x ≤ y)
    (hx : 0 < wx) (hy : 0 < wy) :
    x ≤ mean2 x y wx wy ∧ mean2 x y wx wy ≤ y := by
  have hsum := add_pos hx hy
  constructor
  · apply (le_div_iff₀ hsum).mpr
    nlinarith only [mul_nonneg hy.le (sub_nonneg.mpr hxy)]
  · apply (div_le_iff₀ hsum).mpr
    nlinarith only [mul_nonneg hx.le (sub_nonneg.mpr hxy)]

private theorem mean3_between {x y z wx wy wz : ℝ} (hxy : x ≤ y) (hyz : y ≤ z)
    (hx : 0 < wx) (hy : 0 < wy) (hz : 0 < wz) :
    x ≤ mean3 x y z wx wy wz ∧ mean3 x y z wx wy wz ≤ z := by
  have hsum := add_pos (add_pos hx hy) hz
  have hxz := hxy.trans hyz
  constructor
  · apply (le_div_iff₀ hsum).mpr
    nlinarith only [mul_nonneg hy.le (sub_nonneg.mpr hxy),
      mul_nonneg hz.le (sub_nonneg.mpr hxz)]
  · apply (div_le_iff₀ hsum).mpr
    nlinarith only [mul_nonneg hx.le (sub_nonneg.mpr hxz),
      mul_nonneg hy.le (sub_nonneg.mpr hyz)]

/-- Two ordered centers in the data interval always assign the endpoints to
different cells. -/
def Centers.ofInterval {D : Four} (x y : ℝ) (hx : D.a ≤ x)
    (hxy : x < y) (hy : y ≤ D.d) : Centers D where
  left := x
  right := y
  ordered := hxy
  first := by dsimp [mid]; linarith
  last := by dsimp [mid]; linarith

/-- Assign by the midpoint, then replace each center with its cluster's
weighted mean. A singleton moves its center to its unique point. -/
def Centers.next {D : Four} (I : Centers D) (w : Weights) : Centers D :=
  match D.cut I.boundary with
  | .one => by
    have hr := mean3_between D.bc.le D.cd.le w.bpos w.cpos w.dpos
    exact ofInterval D.a (mean3 D.b D.c D.d w.b w.c w.d)
      le_rfl (D.ab.trans_le hr.1) hr.2
  | .two => by
    have hl := mean2_between D.ab.le w.apos w.bpos
    have hr := mean2_between D.cd.le w.cpos w.dpos
    exact ofInterval (mean2 D.a D.b w.a w.b) (mean2 D.c D.d w.c w.d)
      hl.1 ((hl.2.trans_lt D.bc).trans_le hr.1) hr.2
  | .three => by
    have hl := mean3_between D.ab.le D.bc.le w.apos w.bpos w.cpos
    exact ofInterval (mean3 D.a D.b D.c w.a w.b w.c) D.d
      hl.1 (hl.2.trans_lt D.cd) le_rfl

theorem Centers.next_in_balanced_hulls {D : Four} (I : Centers D) (w : Weights)
    (hc : D.cut I.boundary = .two) :
    (D.a ≤ (I.next w).left ∧ (I.next w).left ≤ D.b) ∧
      (D.c ≤ (I.next w).right ∧ (I.next w).right ≤ D.d) := by
  simp only [Centers.next, hc, Centers.ofInterval]
  exact ⟨mean2_between D.ab.le w.apos w.bpos, mean2_between D.cd.le w.cpos w.dpos⟩

/-- Sum of weighted squared distances to the nearer of the two centers. -/
def Centers.sse {D : Four} (I : Centers D) (w : Weights) : ℝ :=
  let err (x : ℝ) := min ((x - I.left) ^ 2) ((x - I.right) ^ 2)
  w.a * err D.a + w.b * err D.b + w.c * err D.c + w.d * err D.d

end

end IndexRecall
