import IndexRecall.OptimalBuckets
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.FieldSimp

/-! # Paired rows in the plane

There are `s` pairs of rows, each with `2*s` points. The first point of
each lower row is its tip. The second layout exchanges the two first
points in every pair. Coordinates and router scores are exact reals.
-/

noncomputable section

namespace IndexRecall.PlanarGap

abbrev Point := ℝ × ℝ
abbrev Label (s : ℕ) := Fin s × Bool
def size (s : ℕ) := 2 * s
abbrev Slot (s : ℕ) := Fin (size s)
abbrev Layout (s : ℕ) := Label s → Slot s → Point

def pitch (s : ℕ) : ℝ := 10 * (size s : ℝ)
def height (s : ℕ) : ℝ := (4 * pitch s * (size s : ℝ))^2
def delta {s : ℕ} (t : Slot s) : ℝ := (t.val : ℝ) / (size s : ℝ)^2
def drift (s : ℕ) : ℝ := (∑ t : Slot s, delta t) / (size s : ℝ)
def origin {s : ℕ} (i : Fin s) : ℝ := 3 * (i.val : ℝ) * pitch s
def baseX {s : ℕ} (l : Label s) : ℝ := origin l.1 + if l.2 then pitch s else 0

def site {s : ℕ} (l : Label s) (t : Slot s) : Point :=
  (baseX l + delta t, if l.2 then 1 else if t.val = 0 then 2 else 0)

def exchange {s : ℕ} : Equiv.Perm (Label s × Slot s) where
  toFun := fun ((i, d), t) => ((i, if t.val = 0 then !d else d), t)
  invFun := fun ((i, d), t) => ((i, if t.val = 0 then !d else d), t)
  left_inv := by rintro ⟨⟨i, d⟩, t⟩; by_cases h : t.val = 0 <;> simp [h]
  right_inv := by rintro ⟨⟨i, d⟩, t⟩; by_cases h : t.val = 0 <;> simp [h]

def swapped {s : ℕ} : Layout s := fun l t => site (exchange (l, t)).1 (exchange (l, t)).2

def mean {s : ℕ} (x : Layout s) (l : Label s) : Point :=
  ((∑ t, (x l t).1) / (size s : ℝ), (∑ t, (x l t).2) / (size s : ℝ))

def tip {s : ℕ} (i : Fin s) : Point := (origin i, 2)

/-- The replacement height lies strictly between the decoy and tip heights. -/
def rerouted {s : ℕ} (l : Label s) : Point :=
  if l.2 then mean site l else (origin l.1, 3 / 2)

def query {s : ℕ} (i : Fin s) : Point := (origin i + drift s + pitch s / 3, height s)

def sqDist (p q : Point) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem size_four {s : ℕ} (hs : 2 ≤ s) : (4 : ℝ) ≤ size s := by
  exact_mod_cast (show 4 ≤ size s by unfold size; omega)

theorem size_pos {s : ℕ} (hs : 2 ≤ s) : 0 < size s := by
  unfold size
  omega

theorem size_cast_pos {s : ℕ} (hs : 2 ≤ s) : (0 : ℝ) < size s := by
  exact_mod_cast size_pos hs

theorem delta_bounds {s : ℕ} (hs : 2 ≤ s) (t : Slot s) :
    0 ≤ delta t ∧ delta t ≤ 1 / (size s : ℝ) := by
  have hsize := size_cast_pos hs
  have hslot : (t.val : ℝ) ≤ size s := by exact_mod_cast t.isLt.le
  constructor
  · unfold delta
    positivity
  · unfold delta
    apply (div_le_iff₀ (pow_pos hsize 2)).2
    calc
      (t.val : ℝ) ≤ size s := hslot
      _ = (1 / (size s : ℝ)) * (size s : ℝ)^2 := by field_simp [hsize.ne']

theorem drift_bounds {s : ℕ} (hs : 2 ≤ s) :
    0 ≤ drift s ∧ drift s ≤ 1 / (size s : ℝ) := by
  have hsize := size_cast_pos hs
  unfold drift
  refine ⟨div_nonneg (Finset.sum_nonneg fun t _ => (delta_bounds hs t).1) hsize.le,
    (div_le_iff₀ hsize).2 ?_⟩
  calc
    _ ≤ ∑ _t : Slot s, (1 / (size s : ℝ)) :=
      Finset.sum_le_sum fun t _ => (delta_bounds hs t).2
    _ = _ := by simp [mul_comm]

theorem label_card (s : ℕ) : Fintype.card (Label s) = size s := by
  simp [size, Nat.mul_comm]

end IndexRecall.PlanarGap
