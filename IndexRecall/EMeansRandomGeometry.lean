import IndexRecall.EMeansWitness
import Mathlib.Analysis.Convex.Combination
import Mathlib.Analysis.Normed.Module.Convex

/-! # Geometric separation for a single ordinary target-row seed -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

open Vectors

variable {k : ℕ}

/-- The explicit cosine-gap documents shared by the theorem and the replay. -/
abbrev replaySites (z : Fin k) := site z (replayParam k)

/-- The raw target center ranges from its original document to the full target-row mean. -/
def targetSegment (z s : Fin k) : Set (Space k) :=
  let x := site z (replayParam k)
  segment ℝ (x (z,s)) ((k : ℝ)⁻¹ • rowSum x z)

/-- Other centers can mix documents from any of the foreign rows. -/
def foreignHull (z : Fin k) : Set (Space k) :=
  convexHull ℝ ((site z (replayParam k)) '' {p | p.1 ≠ z})

theorem site_mem_foreignHull (z : Fin k) (p : Fin k × Fin k) (hp : p.1 ≠ z) :
    site z (replayParam k) p ∈ foreignHull z :=
  subset_convexHull ℝ _ ⟨p, hp, rfl⟩

private theorem foreignHull_subset (z : Fin k) {C : Set (Space k)} (hc : Convex ℝ C)
    (h : ∀ p : Fin k × Fin k, p.1 ≠ z → replaySites z p ∈ C) : foreignHull z ⊆ C :=
  convexHull_min (by rintro _ ⟨p, hp, rfl⟩; exact h p hp) hc

private theorem targetSegment_subset_cell (hk : 2 ≤ k) (z s : Fin k) :
    targetSegment z s ⊆ cell (replaySites z) z := by
  have hd (j : Fin k) := (replay_geometry hk z).1 (z,j)
  apply (cell_convex _ z).segment_subset (hd s)
  have hm := (cell_convex (replaySites z) z).centerMass_mem (t := Finset.univ)
    (z := fun j : Fin k => replaySites z (z,j)) (w := fun _ => (1 : ℝ))
    (fun _ _ => zero_le_one) (by simpa using (show (0 : ℝ) < k by exact_mod_cast Fin.pos z))
    (fun j _ => hd j)
  simpa [Finset.centerMass, rowSum] using hm

private theorem foreign_query_bounds (hk : 2 ≤ k) (z : Fin k) (p : Fin k × Fin k)
    (hp : p.1 ≠ z) :
    39 / 100 ≤ ⟪query k, replaySites z p⟫_ℝ ∧ ⟪query k, replaySites z p⟫_ℝ ≤ 2 / 5 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hclose : ‖replaySites z p - base z p‖ ≤ 1 / 100 := (replay_close hk z p).trans
    (one_div_le_one_div_of_le (by norm_num) (by nlinarith [sq_nonneg ((k : ℝ) - 1)]))
  have hmove : |⟪query k, replaySites z p⟫_ℝ - ⟪query k, base z p⟫_ℝ| ≤ 1 / 100 := by
    apply le_trans _ hclose
    simpa [inner_sub_right, query_unit] using abs_real_inner_le_norm (query k) (replaySites z p - base z p)
  have hne : p ≠ (z,z) := fun e => hp (congrArg Prod.fst e)
  simp only [base, if_neg hne, query_anchor, if_neg hp] at hmove
  have hmargin := site_margin z (replayParam k) p hne
  rw [site_tip, query_tip] at hmargin
  exact ⟨by linarith [(abs_le.mp hmove).1], by linarith⟩

private theorem hull_bounds (hk : 2 ≤ k) {z : Fin k} {v : Space k} (hv : v ∈ foreignHull z) :
    ‖v‖ ≤ 1 ∧ 39 / 100 ≤ ⟪query k, v⟫_ℝ ∧ 39 / 100 ≤ ‖v‖ := by
  have hball : v ∈ Metric.closedBall (0 : Space k) 1 :=
    foreignHull_subset z (convex_closedBall _ _) (fun p _ => by
      simp [Metric.mem_closedBall, replaySites, site_unit]) hv
  have hq : 39 / 100 ≤ ⟪query k, v⟫_ℝ :=
    foreignHull_subset z (convex_halfSpace_ge ((innerₗ _ (query k)).isLinear) _)
      (fun p hp => (foreign_query_bounds hk z p hp).1) hv
  exact ⟨by simpa using hball, hq, hq.trans (by
    simpa [query_unit] using real_inner_le_norm (query k) v)⟩

/-- A linear score that vanishes on the ordinary target documents has both segment
endpoints bounded by half the tip's score, since the full mean gives the tip weight 1/k. -/
private theorem segment_projection_le (hk : 2 ≤ k) {z s : Fin k} (hs : s ≠ z)
    (w : Space k) {r : ℝ} (hr : 0 ≤ r)
    (hrow : ∀ j : Fin k, ⟪w, replaySites z (z,j)⟫_ℝ = if j = z then r else 0)
    {u : Space k} (hu : u ∈ targetSegment z s) : ⟪w,u⟫_ℝ ≤ r / 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  apply (convex_halfSpace_le ((innerₗ _ w).isLinear) (r / 2)).segment_subset _ _ hu
  · change ⟪w, replaySites z (z,s)⟫_ℝ ≤ r / 2
    rw [hrow, if_neg hs]
    positivity
  · change ⟪w, (k : ℝ)⁻¹ • rowSum (replaySites z) z⟫_ℝ ≤ r / 2
    simp only [real_inner_smul_right, rowSum, inner_sum, hrow, Finset.sum_ite_eq',
      Finset.mem_univ, if_true]
    rw [mul_comm, ← div_eq_mul_inv]
    exact div_le_div_of_nonneg_left hr (by norm_num) hk'

private theorem score_unitDir (w v : Space k) :
    ⟪w,unitDir v⟫_ℝ = ⟪w,v⟫_ℝ / ‖v‖ := by
  simp [unitDir, real_inner_smul_right, div_eq_mul_inv, mul_comm]

/-- Every target document strictly prefers the target segment to every foreign center. -/
theorem target_prefers_segment (hk : 2 ≤ k) {z s : Fin k} {u v : Space k}
    (hu : u ∈ targetSegment z s) (hv : v ∈ foreignHull z) (p : Fin k × Fin k) (hp : p.1 = z) :
    ⟪site z (replayParam k) p, unitDir v⟫_ℝ < ⟪site z (replayParam k) p, unitDir u⟫_ℝ := by
  obtain ⟨i,j⟩ := p
  change i = z at hp
  subst i
  have hc := targetSegment_subset_cell hk z s hu
  have hn := cell_norm_lower _ (site_unit z _) hc
  have hown := hc.2.1 (z,j) rfl
  obtain ⟨_, _, hvn⟩ := hull_bounds hk hv
  have hcross : ⟪replaySites z (z,j), v⟫_ℝ ≤ 1 / 5 := by
    apply foreignHull_subset z (convex_halfSpace_le ((innerₗ _ (replaySites z (z,j))).isLinear) _) _ hv
    intro b hb
    change ⟪replaySites z (z,j), replaySites z b⟫_ℝ ≤ 1 / 5
    rw [site_cross_inner z _ (p := (z,j)) hb.symm, query_target]
    split_ifs <;> linarith [(foreign_query_bounds hk z b hb).2]
  rw [score_unitDir, score_unitDir]
  apply lt_of_lt_of_le (b := (3 / 4 : ℝ))
  · apply (div_lt_iff₀ (by linarith : 0 < ‖v‖)).2
    linarith
  · apply (le_div_iff₀ (by linarith : 0 < ‖u‖)).2
    linarith [hc.1]

/-- Even a foreign row without its own seed prefers each foreign center to the target.
Same-row document cosines are at least 3/4. Different foreign rows share the query coordinate. -/
theorem foreign_prefers_hull (hk : 2 ≤ k) {z s : Fin k} (hs : s ≠ z) {u v : Space k}
    (hu : u ∈ targetSegment z s) (hv : v ∈ foreignHull z) (p : Fin k × Fin k) (hp : p.1 ≠ z) :
    ⟪site z (replayParam k) p, unitDir u⟫_ℝ < ⟪site z (replayParam k) p, unitDir v⟫_ℝ := by
  have hq := foreign_query_bounds hk z p hp
  have hn := cell_norm_lower _ (site_unit z _) (targetSegment_subset_cell hk z s hu)
  obtain ⟨hv1, _, hvn⟩ := hull_bounds hk hv
  have htarget := segment_projection_le hk hs (replaySites z p) (r := ⟪query k, replaySites z p⟫_ℝ / 2)
    (by linarith [hq.1]) (fun j => by
      rw [site_cross_inner z _ (b := (z,j)) hp, query_target]
      split_ifs <;> ring) hu
  have hother : 3 / 20 ≤ ⟪replaySites z p, v⟫_ℝ := by
    apply foreignHull_subset z (convex_halfSpace_ge ((innerₗ _ (replaySites z p)).isLinear) _) _ hv
    intro b hb
    change 3 / 20 ≤ ⟪replaySites z p, replaySites z b⟫_ℝ
    by_cases h : p.1 = b.1
    · linarith [((replay_geometry hk z).1 b).2.1 p h]
    · rw [site_cross_inner z _ h]
      have hbq := (foreign_query_bounds hk z b hb).1
      have hprod := mul_le_mul hq.1 hbq (by norm_num) (by linarith [hq.1])
      linarith
  rw [score_unitDir, score_unitDir]
  apply lt_of_lt_of_le (b := (3 / 20 : ℝ))
  · apply (div_lt_iff₀ (by linarith : 0 < ‖u‖)).2
    linarith [hq.2]
  · apply (le_div_iff₀ (by linarith : 0 < ‖v‖)).2
    linarith

/-- The winning list is strictly behind every foreign direction, including directions
of centers whose closing list is empty. -/
theorem query_prefers_hull (hk : 2 ≤ k) {z s : Fin k} (hs : s ≠ z) {u v : Space k}
    (hu : u ∈ targetSegment z s) (hv : v ∈ foreignHull z) :
    ⟪query k, unitDir u⟫_ℝ < ⟪query k, unitDir v⟫_ℝ := by
  have hn := cell_norm_lower _ (site_unit z _) (targetSegment_subset_cell hk z s hu)
  obtain ⟨hv1, hq, hvn⟩ := hull_bounds hk hv
  have huq := segment_projection_le hk hs (query k) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (query_target z _) hu
  rw [score_unitDir, score_unitDir]
  apply lt_of_le_of_lt (b := (1 / 3 : ℝ))
  · apply (div_le_iff₀ (by linarith : 0 < ‖u‖)).2
    linarith
  · apply (lt_div_iff₀ (by linarith : 0 < ‖v‖)).2
    linarith

end IndexRecall.SphericalGap.EMeans
