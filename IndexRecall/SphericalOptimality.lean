import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Tactic.Linarith

/-! # Strict spherical k-means optimality in the repeated-anchor configuration -/

noncomputable section

open scoped BigOperators InnerProductSpace

namespace IndexRecall.SphericalGap

variable {k : ℕ} {E : Type*}

/-- Vector sum of the documents assigned to a label. -/
def clusterSum [AddCommMonoid E] (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k)
    (j : Fin k) : E :=
  ∑ p ∈ Finset.univ.filter (fun p => f p = j), x p

/-- Spherical k-means score after optimizing the unit representative of each list. -/
def quality [NormedAddCommGroup E] (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k) : ℝ :=
  ∑ j, ‖clusterSum x f j‖

/-- An assignment obtained from the original rows by permuting labels. -/
def Relabeled (f : Fin k × Fin k → Fin k) : Prop :=
  ∃ e : Equiv.Perm (Fin k), ∀ p, f p = e p.1

theorem clusterSum_row [AddCommMonoid E] (x : Fin k × Fin k → E) (j : Fin k) :
    clusterSum x Prod.fst j = ∑ t, x (j, t) := by
  classical
  simp only [clusterSum, Finset.sum_filter, Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  simp

theorem clusterSum_relabel [AddCommMonoid E] (x : Fin k × Fin k → E)
    (f : Fin k × Fin k → Fin k) (e : Equiv.Perm (Fin k)) (he : ∀ p, f p = e p.1) (j : Fin k) :
    clusterSum x f (e j) = clusterSum x Prod.fst j := by
  simp [clusterSum, he]

theorem quality_relabel [NormedAddCommGroup E] (x : Fin k × Fin k → E)
    (f : Fin k × Fin k → Fin k) (hf : Relabeled f) : quality x f = quality x Prod.fst := by
  obtain ⟨e, he⟩ := hf
  calc
    _ = ∑ j, ‖clusterSum x f (e j)‖ := (Equiv.sum_comp e fun j => ‖clusterSum x f j‖).symm
    _ = _ := by simp_rw [clusterSum_relabel x f e he]; rfl

/-- Any partition which is not a relabeling must put two different rows in one list.
This pigeonhole argument also allows unused labels. -/
theorem mixed_pair (z : Fin k) (f : Fin k × Fin k → Fin k) (hf : ¬Relabeled f) :
    ∃ p b : Fin k × Fin k, p.1 ≠ b.1 ∧ f p = f b := by
  by_contra! hm
  let r (i : Fin k) := f (i, z)
  have hi : Function.Injective r := by
    intro i j hij
    by_contra hne
    exact hm (i,z) (j,z) hne hij
  let e : Equiv.Perm (Fin k) := Equiv.ofBijective r hi.bijective_of_finite
  apply hf
  refine ⟨e, fun p => ?_⟩
  obtain ⟨j, hj⟩ := e.surjective (f p)
  have hp : p.1 = j := by
    by_contra hne
    exact hm p (j,z) hne hj.symm
  rw [hp, hj]

variable [NormedAddCommGroup E]

/-- Each assignment's optimized score moves by at most the total document
displacement, regardless of the sizes of its lists. -/
theorem quality_movement (x y : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k) :
    |quality x f - quality y f| ≤ ∑ p, ‖x p - y p‖ := by
  classical
  let rows (j : Fin k) := Finset.univ.filter (fun p => f p = j)
  unfold quality
  rw [← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ j, |‖clusterSum x f j‖ - ‖clusterSum y f j‖| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, ∑ p ∈ rows j, ‖x p - y p‖ := by
      apply Finset.sum_le_sum
      intro j _
      apply (abs_norm_sub_norm_le _ _).trans
      simpa only [rows, clusterSum, Finset.sum_sub_distrib] using
        norm_sum_le (rows j) (fun p => x p - y p)
    _ = _ := Finset.sum_fiberwise Finset.univ f _

private theorem sum_split_pair {A M : Type*} [DecidableEq A] [AddCommMonoid M]
    (s : Finset A) (g : A → M) {p b : A} (hp : p ∈ s) (hb : b ∈ s) (hne : p ≠ b) :
    ∑ t ∈ s, g t = g p + g b + ∑ t ∈ (s.erase p).erase b, g t := by
  rw [add_assoc, Finset.add_sum_erase (s.erase p) g (Finset.mem_erase.mpr ⟨hne.symm, hb⟩),
    Finset.add_sum_erase s g hp]

/-- Removing a pair from a list isolates the least possible penalty incurred by
keeping the pair together. Adding the remaining documents cannot cancel that penalty. -/
private theorem pair_deficit {A : Type*} (s : Finset A) (x : A → E) {p b : A}
    (hp : p ∈ s) (hb : b ∈ s) (hne : p ≠ b) :
    ‖x p‖ + ‖x b‖ - ‖x p + x b‖ ≤ (∑ t ∈ s, ‖x t‖) - ‖∑ t ∈ s, x t‖ := by
  classical
  let rest := (s.erase p).erase b
  rw [sum_split_pair s x hp hb hne, sum_split_pair s (fun t => ‖x t‖) hp hb hne]
  linarith [norm_add_le (x p + x b) (∑ t ∈ rest, x t), norm_sum_le rest x]

/-- If two unit documents share a list then the total spherical loss is at least
their pair cost `2 - ‖x p + x b‖`. -/
theorem pair_cost_le (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k)
    (hu : ∀ t, ‖x t‖ = 1) {p b : Fin k × Fin k} (hne : p ≠ b) (he : f p = f b) :
    2 - ‖x p + x b‖ ≤ (∑ _t : Fin k × Fin k, (1 : ℝ)) - quality x f := by
  classical
  let s (j : Fin k) := Finset.univ.filter (fun t => f t = j)
  let loss (j : Fin k) := (∑ t ∈ s j, ‖x t‖) - ‖clusterSum x f j‖
  have hnon (j : Fin k) : 0 ≤ loss j := sub_nonneg.mpr (norm_sum_le (s j) x)
  have hp : p ∈ s (f p) := by simp [s]
  have hb : b ∈ s (f p) := by simp [s, he]
  have hpair := pair_deficit (s (f p)) x hp hb hne
  rw [hu p, hu b, one_add_one_eq_two] at hpair
  calc
    _ ≤ loss (f p) := hpair
    _ ≤ ∑ j, loss j := Finset.single_le_sum (fun j _ => hnon j) (Finset.mem_univ (f p))
    _ = _ := by
      simp only [loss, s, Finset.sum_sub_distrib, Finset.sum_fiberwise, hu, quality]

variable [InnerProductSpace ℝ E]

/-- Group the document scores by their assigned center. -/
theorem score_by_list (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k) (c : Fin k → E) :
    (∑ p, ⟪x p, c (f p)⟫_ℝ) = ∑ j, ⟪clusterSum x f j, c j⟫_ℝ := by
  classical
  rw [← Finset.sum_fiberwise Finset.univ f (fun p => ⟪x p, c (f p)⟫_ℝ)]
  apply Finset.sum_congr rfl
  intro j _
  rw [clusterSum, sum_inner]
  exact Finset.sum_congr rfl fun p hp => by rw [(Finset.mem_filter.mp hp).2]

/-- For arbitrary unit centers, the assignment's sum of document cosines is
bounded by its sum of list-sum norms. -/
theorem center_score_le_quality (x : Fin k × Fin k → E) (f : Fin k × Fin k → Fin k)
    (c : Fin k → E) (hc : ∀ j, ‖c j‖ = 1) :
    (∑ p, ⟪x p, c (f p)⟫_ℝ) ≤ quality x f := by
  rw [score_by_list]
  exact Finset.sum_le_sum fun j _ => by
    simpa only [hc, mul_one] using real_inner_le_norm (clusterSum x f j) (c j)

/-- The anchor centers lose only `1 - ⟪tip, a z⟫` while a mixed pair loses more
than `1/5`. The difference gives a uniform margin over all competing assignments. -/
theorem repeated_quality_gap (z : Fin k) (a : Fin k → E) (tip : E) (ha : ∀ i, ‖a i‖ = 1)
    (ht : ‖tip‖ = 1)
    (hpair : ∀ p b : Fin k × Fin k, p.1 ≠ b.1 →
      ‖(if p = (z,z) then tip else a p.1) + (if b = (z,z) then tip else a b.1)‖ < 9 / 5)
    (f : Fin k × Fin k → Fin k) (hf : ¬Relabeled f) :
    quality (fun p => if p = (z,z) then tip else a p.1) f + (⟪tip, a z⟫_ℝ - 4 / 5) <
      quality (fun p => if p = (z,z) then tip else a p.1) Prod.fst := by
  classical
  let x (p : Fin k × Fin k) := if p = (z,z) then tip else a p.1
  change quality x f + (⟪tip, a z⟫_ℝ - 4 / 5) < quality x Prod.fst
  have hu (p) : ‖x p‖ = 1 := by
    by_cases hp : p = (z,z) <;> simp [x, hp, ht, ha]
  obtain ⟨p, b, hrows, hsame⟩ := mixed_pair z f hf
  have hbad := pair_cost_le x f hu (fun h => hrows (congrArg Prod.fst h)) hsame
  have hbound : ‖x p + x b‖ < 9 / 5 := hpair p b hrows
  have hbase : (∑ p : Fin k × Fin k, (1 - ⟪x p, a p.1⟫_ℝ)) = 1 - ⟪tip, a z⟫_ℝ := by
    rw [Finset.sum_eq_single (z,z)]
    · simp [x]
    · intro b _ hne
      simp [x, hne, real_inner_self_eq_norm_sq, ha]
    · simp
  rw [Finset.sum_sub_distrib] at hbase
  have hcenter := center_score_le_quality x Prod.fst a ha
  linarith

/-- All but one document lie at their row anchors. Assigning the tip to its own row
costs less than `1/5`, while every cross-row pair costs more than `1/5`. Thus every
other partition, balanced or unbalanced, has strictly smaller optimized score. -/
theorem repeated_strict (z : Fin k) (a : Fin k → E) (tip : E) (ha : ∀ i, ‖a i‖ = 1)
    (ht : ‖tip‖ = 1) (halign : 4 / 5 < ⟪tip, a z⟫_ℝ)
    (hpair : ∀ p b : Fin k × Fin k, p.1 ≠ b.1 →
      ‖(if p = (z,z) then tip else a p.1) + (if b = (z,z) then tip else a b.1)‖ < 9 / 5)
    (f : Fin k × Fin k → Fin k) (hf : ¬Relabeled f) :
    quality (fun p => if p = (z,z) then tip else a p.1) f <
      quality (fun p => if p = (z,z) then tip else a p.1) Prod.fst := by
  linarith [repeated_quality_gap z a tip ha ht hpair f hf]

end IndexRecall.SphericalGap
