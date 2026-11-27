import IndexRecall.ManyQueryVectors

/-! # Splitting rows and combining orthogonal pieces in spherical k-means -/

noncomputable section
open scoped InnerProductSpace Classical

namespace IndexRecall.SphericalGap.ManyQuery

variable {k : ℕ}

private abbrev Assignment (k : ℕ) := Fin k × Fin k → Fin k
private def columns (a : Assignment k) (i j : Fin k) :=
  Finset.univ.filter fun l => a (i,l) = j
private def piece (a : Assignment k) (i j : Fin k) : Space k :=
  ∑ l ∈ columns a i j, base (i,l)
private def rowLabels (a : Assignment k) (i : Fin k) :=
  Finset.univ.filter fun j => (columns a i j).Nonempty
private def listRows (a : Assignment k) (j : Fin k) :=
  Finset.univ.filter fun i => (columns a i j).Nonempty

private theorem sum_pieces_row (a : Assignment k) (i : Fin k) :
    ∑ j, piece a i j = rowSum base i :=
  Finset.sum_fiberwise Finset.univ (fun l => a (i,l)) (fun l => base (i,l))

private theorem sum_pieces_list (a : Assignment k) (j : Fin k) :
    clusterSum base a j = ∑ i, piece a i j := by
  simp only [clusterSum, piece, columns, Finset.sum_filter, Fintype.sum_prod_type]

private theorem piece_empty {a : Assignment k} {i j : Fin k} (h : ¬(columns a i j).Nonempty) :
    piece a i j = 0 := by simp [piece, Finset.not_nonempty_iff_eq_empty.mp h]

private theorem piece_norm_lower {a : Assignment k} {i j : Fin k} (h : (columns a i j).Nonempty) :
    1 ≤ ‖piece a i j‖ := by
  obtain ⟨l, hl⟩ := h
  calc
    1 = ⟪base (i,l), base (i,l)⟫_ℝ := by rw [real_inner_self_eq_norm_sq, base_unit, one_pow]
    _ ≤ ∑ b ∈ columns a i j, ⟪base (i,l), base (i,b)⟫_ℝ :=
      Finset.single_le_sum (fun b _ => base_same_nonneg i l b) hl
    _ = ⟪base (i,l), piece a i j⟫_ℝ := (inner_sum ..).symm
    _ ≤ _ := by simpa only [base_unit, one_mul] using real_inner_le_norm (base (i,l)) (piece a i j)

private theorem piece_orthogonal (a : Assignment k) {i r : Fin k} (hir : i ≠ r) (j : Fin k) :
    ⟪piece a i j, piece a r j⟫_ℝ = 0 := by
  simp [piece, sum_inner, inner_sum, base_cross hir]

private abbrev splitGain : ℝ := 1 - Real.sqrt 3 / 2
private abbrev mergeLoss : ℝ := 2 - Real.sqrt 2

private theorem gain_nonneg : 0 ≤ splitGain := by
  dsimp [splitGain]
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]

/-- An unsplit row gains nothing. Even splitting off every document can gain at
most the difference between k and the row sum's projection onto its anchor. -/
private theorem split_bound (a : Assignment k) (i : Fin k) :
    ∑ j, ‖piece a i j‖ ≤ ‖rowSum base i‖ + splitGain * ((rowLabels a i).card - 1 : ℝ) := by
  by_cases hone : (rowLabels a i).card = 1
  · obtain ⟨j, hj⟩ := Finset.card_eq_one.mp hone
    have hz (r) (hr : r ≠ j) : piece a i r = 0 := by
      apply piece_empty
      intro h
      have hm : r ∈ rowLabels a i := Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩
      exact hr (Finset.mem_singleton.mp (hj ▸ hm))
    have hs : ∑ r, piece a i r = piece a i j :=
      Finset.sum_eq_single j (fun r _ hr => hz r hr) (by simp)
    have hn : ∑ r, ‖piece a i r‖ = ‖piece a i j‖ :=
      Finset.sum_eq_single j (fun r _ hr => by rw [hz r hr, norm_zero]) (by simp)
    simp [← sum_pieces_row a i, hs, hn, hone]
  · have hmem : a (i,i) ∈ rowLabels a i :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, i, by simp [columns]⟩
    have hpos := Finset.card_pos.mpr ⟨_, hmem⟩
    have hn : (2 : ℝ) ≤ (rowLabels a i).card := by exact_mod_cast (show 2 ≤ _ by omega)
    have hunit : ∑ j, ‖piece a i j‖ ≤ k := calc
      _ ≤ ∑ j, ∑ l ∈ columns a i j, ‖base (i,l)‖ :=
        Finset.sum_le_sum fun j _ => norm_sum_le _ _
      _ = ∑ l, ‖base (i,l)‖ := Finset.sum_fiberwise Finset.univ (fun l => a (i,l)) _
      _ = _ := by simp [base_unit]
    calc
      _ ≤ (k : ℝ) := hunit
      _ ≤ ‖rowSum base i‖ + splitGain := sub_le_iff_le_add.mp (base_row_lower i)
      _ ≤ _ := add_le_add_left (le_mul_of_one_le_right gain_nonneg (by linarith)) _

section Orthogonal
variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

private theorem merge_two (u v : E) (h : ⟪u,v⟫_ℝ = 0) (hu : 1 ≤ ‖u‖) (hv : 1 ≤ ‖v‖) :
    ‖u + v‖ + mergeLoss ≤ ‖u‖ + ‖v‖ := by
  have hr : (1 : ℝ) ≤ Real.sqrt 2 := by
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_nonneg 2]
  have hp := norm_add_sq_real u v
  rw [h] at hp
  apply le_sub_iff_add_le.mp
  apply (sq_le_sq₀ (norm_nonneg _) (by dsimp [mergeLoss]; linarith)).mp
  dsimp [mergeLoss]
  nlinarith only [hp, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2),
    mul_nonneg (sub_nonneg.mpr hu) (sub_nonneg.mpr hv),
    mul_nonneg (sub_nonneg.mpr hr) (show 0 ≤ ‖u‖ + ‖v‖ - 2 by linarith)]

/-- Adding another orthogonal piece costs at least 2 − √2 once a list has a piece. -/
private theorem merge_pieces (v : ι → E) (s : Finset ι)
    (ho : ∀ i j, i ≠ j → ⟪v i,v j⟫_ℝ = 0)
    (hu : ∀ i ∈ s, 1 ≤ ‖v i‖) (hs : s.Nonempty) :
    1 ≤ ‖∑ i ∈ s, v i‖ ∧
      ‖∑ i ∈ s, v i‖ + mergeLoss * (s.card - 1 : ℝ) ≤ ∑ i ∈ s, ‖v i‖ := by
  induction s using Finset.induction_on with
  | empty => simp at hs
  | @insert i s hnot ih =>
    have hi := hu i (Finset.mem_insert_self _ _)
    by_cases hn : s.Nonempty
    · obtain ⟨hold, hloss⟩ := ih (fun j hj => hu j (Finset.mem_insert_of_mem hj)) hn
      have hdot : ⟪v i, ∑ j ∈ s, v j⟫_ℝ = 0 := by
        rw [inner_sum]
        exact Finset.sum_eq_zero fun j hj => ho i j (fun he => hnot (he ▸ hj))
      rw [Finset.sum_insert hnot, Finset.sum_insert hnot, Finset.card_insert_of_notMem hnot,
        Nat.cast_add, Nat.cast_one]
      refine ⟨hi.trans ((sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp ?_), ?_⟩
      · rw [norm_add_sq_real, hdot]
        linarith [sq_nonneg ‖∑ j ∈ s, v j‖]
      · linarith only [merge_two _ _ hdot hi hold, hloss]
    · rw [Finset.not_nonempty_iff_eq_empty.mp hn]
      simp [hi]
end Orthogonal

private def extra (n : ℕ) : ℝ := max ((n : ℝ) - 1) 0

private theorem merge_bound (a : Assignment k) (j : Fin k) :
    ‖clusterSum base a j‖ + mergeLoss * extra (listRows a j).card ≤ ∑ i, ‖piece a i j‖ := by
  let s := listRows a j
  let v := fun i => piece a i j
  have hz (i) (hi : i ∉ s) : v i = 0 :=
    piece_empty (fun h => hi (Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩))
  have hsum : ∑ i ∈ s, v i = ∑ i, v i :=
    Finset.sum_subset (Finset.subset_univ _) fun i _ hi => hz i hi
  have hnorm : ∑ i ∈ s, ‖v i‖ = ∑ i, ‖v i‖ :=
    Finset.sum_subset (Finset.subset_univ _) fun i _ hi => by rw [hz i hi, norm_zero]
  rw [sum_pieces_list]
  change ‖∑ i, v i‖ + mergeLoss * extra s.card ≤ ∑ i, ‖v i‖
  rw [← hsum, ← hnorm]
  by_cases hs : s.Nonempty
  · have hn : (1 : ℝ) ≤ s.card := by exact_mod_cast hs.card_pos
    rw [extra, max_eq_left (sub_nonneg.mpr hn)]
    exact (merge_pieces v s (fun _ _ h => piece_orthogonal a h j)
      (fun _ hi => piece_norm_lower (Finset.mem_filter.mp hi).2) hs).2
  · simp [Finset.not_nonempty_iff_eq_empty.mp hs, extra]

/-- Count the same nonempty (row, list) intersections in both directions. An unused
label can only increase the number of mergers relative to the number of splits. -/
private theorem split_count_le_merges (a : Assignment k) :
    (∑ i, ((rowLabels a i).card - 1 : ℝ)) ≤ ∑ j, extra (listRows a j).card := by
  have he : ∑ i, (rowLabels a i).card = ∑ j, (listRows a j).card := by
    simp only [rowLabels, listRows, Finset.card_filter]
    exact Finset.sum_comm
  calc
    _ = ∑ j, ((listRows a j).card - 1 : ℝ) := by
      simp only [Finset.sum_sub_distrib, ← Nat.cast_sum, he]
    _ ≤ _ := Finset.sum_le_sum fun _ _ => le_max_left _ _

private theorem one_merge {a : Assignment k} (i : Fin k) (ha : ¬Relabeled a) :
    1 ≤ ∑ j, extra (listRows a j).card := by
  obtain ⟨p,b,hrows,hsame⟩ := mixed_pair i a ha
  have hm (p : Fin k × Fin k) : p.1 ∈ listRows a (a p) :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, p.2, by simp [columns]⟩
  have hc := Finset.one_lt_card.mpr ⟨p.1, hm p, b.1, hsame ▸ hm b, hrows⟩
  have hn : (2 : ℝ) ≤ (listRows a (a p)).card := by exact_mod_cast hc
  calc
    (1 : ℝ) ≤ (listRows a (a p)).card - 1 := by linarith
    _ ≤ extra (listRows a (a p)).card := le_max_left _ _
    _ ≤ _ := Finset.single_le_sum (fun j _ => le_max_right ((listRows a j).card - 1 : ℝ) 0)
      (Finset.mem_univ (a p))

/-- A uniform objective gap for every competing assignment, even with empty labels. -/
theorem repeated_quality_gap (hk : 2 ≤ k) (a : Fin k × Fin k → Fin k) (ha : ¬Relabeled a) :
    quality base a + 1 / 4 ≤ quality (base (k := k)) Prod.fst := by
  have hm := Finset.sum_le_sum (s := Finset.univ) (fun j _ => merge_bound a j)
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ => split_bound a i)
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_comm (f := fun j i => ‖piece a i j‖)] at hm
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hs
  have htotal := hm.trans hs
  have hcount := mul_le_mul_of_nonneg_left (split_count_le_merges a) gain_nonneg
  have hgap : (1 : ℝ) / 4 ≤ mergeLoss - splitGain := by
    have h2 : Real.sqrt 2 ≤ 3 / 2 := by nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    have h3 : 3 / 2 ≤ Real.sqrt 3 := by
      nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]
    dsimp [mergeLoss, splitGain]; linarith
  have hpenalty := hgap.trans (le_mul_of_one_le_right (by linarith : 0 ≤ mergeLoss - splitGain)
    (one_merge (⟨0, by omega⟩ : Fin k) ha))
  change quality base a + _ ≤ (∑ i, ‖rowSum base i‖) + _ at htotal
  rw [sub_mul] at hpenalty
  calc
    _ ≤ ∑ i : Fin k, ‖rowSum base i‖ := by linarith only [htotal, hcount, hpenalty]
    _ = _ := by simp only [quality, clusterSum_row, rowSum]

/-- The radius controls the largest tilt in any one row. -/
def perturbationRadius (k : ℕ) : ℝ := 1 / (100 * (k : ℝ)^2)

theorem radius_bounds (hk : 2 ≤ k) : 0 < perturbationRadius k ∧ perturbationRadius k ≤ 1 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hp : (0 : ℝ) < k := by linarith
  unfold perturbationRadius
  refine ⟨by positivity, (div_le_one (by positivity)).mpr ?_⟩
  nlinarith [sq_nonneg ((k : ℝ) - 1)]

/-- Perturbation changes each assignment score by at most 1/50. The strict gap
persists for the entire interval, including the repeated instance at t = 0. -/
theorem unique_optimum (hk : 2 ≤ k) {t : ℝ} (h0 : 0 ≤ t) (ht : t ≤ perturbationRadius k) :
    UniqueOptimum (site (k := k) t) := by
  have h1 := ht.trans (radius_bounds hk).2
  have hsum : (∑ p : Fin k × Fin k, ‖site t p - base p‖) ≤ 1 / 50 := calc
    _ ≤ ∑ _p : Fin k × Fin k, 2 * perturbationRadius k :=
      Finset.sum_le_sum fun p _ => (site_close h0 h1 p).trans (by linarith)
    _ = _ := by
      have hn : (k : ℝ) ≠ 0 := by exact_mod_cast (show k ≠ 0 by omega)
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
        nsmul_eq_mul, Nat.cast_mul, perturbationRadius]
      field_simp [hn]
      ring
  have hm (a : Fin k × Fin k → Fin k) := abs_le.mp ((quality_movement (site t) base a).trans hsum)
  apply uniqueOptimum_of_quality _ (rowSum_ne_zero hk h0 h1)
  intro a ha
  linarith only [(hm a).2, (hm Prod.fst).1, repeated_quality_gap hk a ha]

end IndexRecall.SphericalGap.ManyQuery
