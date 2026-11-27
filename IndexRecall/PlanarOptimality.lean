import IndexRecall.PlanarFamily
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-! # Global squared-error optimality of the original planar rows -/

noncomputable section

namespace IndexRecall.PlanarGap

variable {s : ℕ}

/-- Squared error for any assignment of the points to the labeled centers. -/
def assignmentLoss (x : Layout s) (f : Label s → Slot s → Label s) (c : Label s → Point) : ℝ :=
  ∑ l, ∑ t, sqDist (x l t) (c (f l t))

/-- Squared error when every point uses the center of its own row. -/
def rowLoss (x : Layout s) (c : Label s → Point) : ℝ :=
  assignmentLoss x (fun l _ => l) c

private theorem rowLoss_mean_le (hs : 2 ≤ s) (x : Layout s) (c : Label s → Point) :
    rowLoss x (mean x) ≤ rowLoss x c := by
  unfold rowLoss assignmentLoss
  apply Finset.sum_le_sum
  intro l _
  simp only [sqDist, mean, Finset.sum_add_distrib]
  exact add_le_add (OptimalBuckets.scalar_mean_le (size_pos hs) (fun t => (x l t).1) (c l).1)
    (OptimalBuckets.scalar_mean_le (size_pos hs) (fun t => (x l t).2) (c l).2)

private def anchor (l : Label s) : Point := (baseX l, if l.2 then 1 else 0)

private theorem anchor_loss_le (hs : 2 ≤ s) :
    rowLoss (@site s) anchor ≤ 4 * (s : ℝ) + 1 := by
  classical
  have hsize : 0 < (size s : ℝ) := by linarith [size_four hs]
  let z : Slot s := ⟨0, by unfold size; omega⟩
  have hdist (l : Label s) (t : Slot s) :
      sqDist (site l t) (anchor l) =
        delta t ^ 2 + if l.2 then 0 else if t = z then 4 else 0 := by
    have hz : t.val = 0 ↔ t = z := ⟨fun ht => Fin.ext ht, by rintro rfl; rfl⟩
    simp only [site, anchor, sqDist, add_sub_cancel_left, hz]
    split_ifs <;> norm_num
  have hy (l : Label s) :
      (∑ t : Slot s, if l.2 then (0 : ℝ) else if t = z then 4 else 0) =
        if l.2 then 0 else 4 := by
    split_ifs with h
    · simp
    · simp [Finset.sum_ite_eq']
  have hx : (∑ _l : Label s, ∑ t : Slot s, delta t ^ 2) ≤ 1 := calc
    _ ≤ ∑ _l : Label s, ∑ _t : Slot s, (1 / (size s : ℝ)) ^ 2 := by
      apply Finset.sum_le_sum
      intro l _
      apply Finset.sum_le_sum
      intro t _
      have ht := delta_bounds hs t
      exact (sq_le_sq₀ ht.1 (by positivity)).mpr ht.2
    _ = 1 := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, label_card, nsmul_eq_mul]
      field_simp
  have htotalY : (∑ l : Label s, if l.2 then (0 : ℝ) else 4) = 4 * (s : ℝ) := by
    rw [Fintype.sum_prod_type]
    simp [nsmul_eq_mul, mul_comm]
  unfold rowLoss assignmentLoss
  simp_rw [hdist, Finset.sum_add_distrib, hy]
  rw [htotalY]
  linarith

private def rowIndex (l : Label s) : ℕ := 3 * l.1.val + if l.2 then 1 else 0

private theorem rowIndex_injective : Function.Injective (@rowIndex s) := by
  rintro ⟨i, a⟩ ⟨j, b⟩ h
  cases a <;> cases b <;> simp [rowIndex] at h ⊢ <;>
    first | exact Fin.ext (by omega) | omega

private theorem baseX_separated {l j : Label s} (hne : l ≠ j) :
    pitch s ≤ |baseX l - baseX j| := by
  have h (a : Label s) : baseX a = (rowIndex a : ℝ) * pitch s := by
    cases a with
    | mk i b =>
      cases b <;> simp [baseX, origin, rowIndex]
      ring
  have hp : 0 ≤ pitch s := by unfold pitch; positivity
  rw [h, h, ← sub_mul, abs_mul, abs_of_nonneg hp]
  have hidx : 1 ≤ |(rowIndex l : ℝ) - (rowIndex j : ℝ)| := by
    rcases lt_or_gt_of_ne (fun h => hne (rowIndex_injective h)) with hlt | hlt
    · have hc : (rowIndex l : ℝ) + 1 ≤ (rowIndex j : ℝ) := by
        exact_mod_cast Nat.succ_le_of_lt hlt
      rw [abs_sub_comm, abs_of_nonneg (by linarith)]
      linarith
    · have hc : (rowIndex j : ℝ) + 1 ≤ (rowIndex l : ℝ) := by
        exact_mod_cast Nat.succ_le_of_lt hlt
      rw [abs_of_nonneg (by linarith)]
      linarith
  simpa only [one_mul] using mul_le_mul_of_nonneg_right hidx hp

private theorem site_separated (hs : 2 ≤ s) {l j : Label s} (hne : l ≠ j)
    (t u : Slot s) :
    pitch s - 1 ≤ |(site l t).1 - (site j u).1| := by
  have hs' : (1 : ℝ) ≤ (size s : ℝ) := by linarith [size_four hs]
  have hcap : 1 / (size s : ℝ) ≤ 1 := (div_le_one (by positivity)).2 hs'
  have ht := delta_bounds hs t
  have hu := delta_bounds hs u
  have hdiff : |delta t - delta u| ≤ 1 := abs_le.mpr ⟨by linarith, by linarith⟩
  have h := abs_sub ((site l t).1 - (site j u).1) (delta t - delta u)
  rw [show (site l t).1 - (site j u).1 - (delta t - delta u) = baseX l - baseX j by
    simp only [site]; ring] at h
  linarith [baseX_separated hne]

theorem distinct_sites (hs : 2 ≤ s) :
    Function.Injective (fun p : Label s × Slot s => site p.1 p.2) := by
  rintro ⟨l,t⟩ ⟨j,u⟩ he
  change site l t = site j u at he
  have hrow : l = j := by
    by_contra hn
    have hgap := site_separated hs hn t u
    rw [he, sub_self, abs_zero] at hgap
    unfold pitch at hgap
    linarith [size_four hs]
  subst j
  have hx : delta t = delta u := add_left_cancel (congrArg Prod.fst he)
  have ht : (t.val : ℝ) = u.val := (div_left_inj' (pow_ne_zero 2 (size_cast_pos hs).ne')).mp hx
  rw [Fin.ext (show t.val = u.val by exact_mod_cast ht)]

private theorem dist_nonneg (p q : Point) : 0 ≤ sqDist p q := by
  unfold sqDist
  positivity

private theorem mixed_loss_gt (hs : 2 ≤ s)
    (f : Label s → Slot s → Label s) (c : Label s → Point)
    {l j : Label s} (hne : l ≠ j) (t u : Slot s) (heq : f l t = f j u) :
    4 * (s : ℝ) + 1 < assignmentLoss site f c := by
  classical
  have hp : 40 ≤ pitch s := by unfold pitch; linarith [size_four hs]
  let p := site l t
  let q := site j u
  let a := c (f l t)
  have hgap : (pitch s - 1)^2 ≤ (p.1 - q.1)^2 := by
    apply sq_le_sq.mpr
    rw [abs_of_nonneg (by linarith)]
    exact site_separated hs hne t u
  have hpair : (p.1 - q.1)^2 ≤ 2 * (sqDist p a + sqDist q a) := by
    unfold sqDist
    nlinarith [sq_nonneg (p.1 + q.1 - 2 * a.1), sq_nonneg (p.2 - a.2), sq_nonneg (q.2 - a.2)]
  let cost (l : Label s) := ∑ v, sqDist (site l v) (c (f l v))
  have hrow (l : Label s) (v : Slot s) : sqDist (site l v) (c (f l v)) ≤ cost l := by
    apply Finset.single_le_sum _ (Finset.mem_univ v)
    exact fun _ _ => dist_nonneg _ _
  have htot : cost l + cost j ≤ assignmentLoss site f c := by
    rw [← Finset.sum_pair (f := cost) hne]
    unfold assignmentLoss
    apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    exact fun _ _ _ => Finset.sum_nonneg fun _ _ => dist_nonneg _ _
  have hsum : sqDist p a + sqDist q a ≤ assignmentLoss site f c := by
    dsimp [p, q, a]
    rw [heq]
    have hrowl := hrow l t
    rw [heq] at hrowl
    exact (add_le_add hrowl (hrow j u)).trans htot
  have hslope : 4 * (s : ℝ) + 1 < (pitch s - 1)^2 / 2 := by
    have hsize : pitch s = 20 * (s : ℝ) := by simp [pitch, size]; ring
    nlinarith [sq_nonneg (pitch s - 40)]
  linarith

/-- The original planar rows globally minimize squared error at their means.
Every minimizing assignment uses these same rows up to a permutation. -/
theorem strict_optimal (hs : 2 ≤ s) (f : Label s → Slot s → Label s) (c : Label s → Point) :
    rowLoss (@site s) (mean site) ≤ assignmentLoss site f c ∧
      (assignmentLoss site f c ≤ rowLoss (@site s) (mean site) →
        ∃ e : Equiv.Perm (Label s), ∀ l t, f l t = e l) := by
  classical
  by_cases hm : ∃ (l j : Label s) (t u : Slot s), l ≠ j ∧ f l t = f j u
  · obtain ⟨l, j, t, u, hne, heq⟩ := hm
    have h := ((rowLoss_mean_le hs site anchor).trans (anchor_loss_le hs)).trans_lt
      (mixed_loss_gt hs f c hne t u heq)
    exact ⟨h.le, fun hle => (h.not_ge hle).elim⟩
  · let z : Slot s := ⟨0, by unfold size; omega⟩
    have hinj : Function.Injective (fun l => f l z) := by
      intro l j he
      by_contra hne
      exact hm ⟨l, j, z, z, hne, he⟩
    let e : Equiv.Perm (Label s) := Equiv.ofBijective (fun l => f l z) hinj.bijective_of_finite
    have he (l : Label s) (t : Slot s) : f l t = e l := by
      obtain ⟨j, hj⟩ := e.surjective (f l t)
      have hl : l = j := by
        by_contra hne
        exact hm ⟨l, j, t, z, hne, hj.symm⟩
      simpa only [hl] using hj.symm
    have hcost : assignmentLoss site f c = rowLoss site (fun l => c (e l)) := by
      simp only [assignmentLoss, rowLoss, he]
    exact ⟨hcost ▸ rowLoss_mean_le hs site _, fun _ => ⟨e, he⟩⟩

end IndexRecall.PlanarGap
