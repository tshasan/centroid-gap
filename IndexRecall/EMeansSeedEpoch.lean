import IndexRecall.EMeansRandomGeometry
import IndexRecall.EMeansUpdate

/-! # Full-batch training with one ordinary seed in the target row -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

open Vectors

variable {k : ℕ}

/-- One logical batch visits every document ID exactly once, in any order. -/
def FullBatch (b : Batch k) : Prop := b.rows.Perm Finset.univ.toList

theorem full_row_perm (rows : List (Address k)) (hfull : rows.Perm Finset.univ.toList) (i : Fin k) :
    (assigned rows Prod.fst i).Perm ((Finset.univ.filter (fun p : Address k => p.1 = i)).toList) := by
  apply (hfull.filter _).trans
  apply Multiset.coe_eq_coe.mp
  rw [← Multiset.filter_coe (fun p : Fin k × Fin k => p.1 = i), Finset.coe_toList, Finset.coe_toList]
  rfl

/-- Only the preimage of the target label is needed to recover the full row statistics. -/
theorem full_row_stats {E : Type*} [NormedAddCommGroup E] (x : Address k → E) {rows : List (Address k)}
    (hfull : rows.Perm Finset.univ.toList) {f : Address k → Fin k} {label z : Fin k}
    (hf : ∀ p, f p = label ↔ p.1 = z) :
    (assigned rows f label).length = k ∧ ((assigned rows f label).map x).sum = rowSum x z := by
  have he : assigned rows f label = assigned rows Prod.fst z :=
    List.filter_congr fun p _ => by simp [hf p]
  rw [he]
  have hp := full_row_perm rows hfull z
  refine ⟨by simpa only [Finset.length_toList, row_card] using hp.length_eq, ?_⟩
  have hs := (hp.map x).sum_eq
  rw [Finset.sum_map_toList] at hs
  exact hs.trans (clusterSum_row x z)

/-- The target label and the ordinary seed column can both differ from the row index. -/
structure IsolatedPositions (z col label : Fin k) (pos : Fin k → Space k) : Prop where
  target : pos label ∈ targetSegment z col
  foreign : ∀ i ≠ label, pos i ∈ foreignHull z

/-- A spherical readout of these raw positions assigns exactly the target row to its label. -/
theorem IsolatedPositions.labels {z col label : Fin k} {pos : Fin k → Space k}
    (h : IsolatedPositions z col label pos) (hk : 2 ≤ k) (hcol : col ≠ z) (p : Address k) :
    assignDirection (replaySites z) (fun i => unitDir (pos i)) p = label ↔ p.1 = z := by
  let c := fun i => unitDir (pos i)
  by_cases hp : p.1 = z
  · refine iff_of_true ?_ hp
    by_contra hne
    exact (target_prefers_segment hk h.target (h.foreign _ hne) p hp).not_ge
      (assignDirection_max (replaySites z) c p label)
  · refine iff_of_false (fun he => ?_) hp
    haveI : Nontrivial (Fin k) := Fin.nontrivial_iff_two_le.mpr hk
    obtain ⟨i, hi⟩ := exists_ne label
    have hm := assignDirection_max (replaySites z) c p i
    change assignDirection (replaySites z) c p = label at he
    rw [he] at hm
    exact (foreign_prefers_hull hk hcol h.target (h.foreign i hi) p hp).not_ge hm

/-- Every occupied competing list has an unweighted mean of foreign documents. -/
theorem assigned_foreign_mean {z label i : Fin k} {f : Address k → Fin k}
    (hf : ∀ p, f p = label ↔ p.1 = z) (hi : i ≠ label) (rows : List (Address k))
    (hn : (assigned rows f i).length ≠ 0) :
    ((assigned rows f i).length : ℝ)⁻¹ • ((assigned rows f i).map (replaySites z)).sum ∈
      foreignHull z := by
  let vs := (assigned rows f i).map (replaySites z)
  have hmem (v) (hv : v ∈ vs) : v ∈ foreignHull z := by
    obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hv
    have he : f p = i := of_decide_eq_true (List.mem_filter.mp hp).2
    exact site_mem_foreignHull z p (fun hz => hi (he.symm.trans ((hf p).mpr hz)))
  simpa only [vs, List.length_map] using list_mean_mem (foreignHull z) (convex_convexHull ℝ _)
    vs (by simpa [vs] using Nat.pos_of_ne_zero hn) hmem

structure IsolatedState (z col label : Fin k) (s : State k (Space k)) : Prop where
  positions : IsolatedPositions z col label s.pos
  mass_pos : ∀ i, 0 < s.mass i

theorem begin_isolated {z col label : Fin k} (seed : Fin k → Address k)
    (hseed : seed label = (z, col)) (hforeign : ∀ i ≠ label, (seed i).1 ≠ z) :
    IsolatedState z col label (beginState (fun i => replaySites z (seed i))) := by
  have hunit (p : Address k) : unitDir (replaySites z p) = replaySites z p := by
    simp [unitDir, replaySites, site_unit]
  refine ⟨⟨?_, ?_⟩, fun i => by dsimp [beginState]; positivity⟩
  · simpa only [beginState, hseed, hunit] using
      (left_mem_segment ℝ (replaySites z (z, col)) ((k : ℝ)⁻¹ • rowSum (replaySites z) z))
  · intro i hi
    simpa only [beginState, hunit] using site_mem_foreignHull z (seed i) (hforeign i hi)

/-- The target mean is always the same endpoint of the segment, regardless of the batch
order. The remaining labels can change their membership at every epoch. -/
theorem IsolatedState.update {z col label : Fin k} {s : State k (Space k)}
    (h : IsolatedState z col label s) (hk : 2 ≤ k) (hcol : col ≠ z)
    (b : Batch k) (hb : FullBatch b) : IsolatedState z col label (update (replaySites z) s b) := by
  let x := replaySites z
  have hf (p) : predict x s p = label ↔ p.1 = z := h.positions.labels hk hcol p
  obtain ⟨hn, hsum⟩ := full_row_stats x hb hf
  refine ⟨⟨?_, ?_⟩, fun i => updateMass_pos _ b.alpha_nonneg b.alpha_le_one (h.mass_pos i)⟩
  · apply update_pos_mem x s b label (convex_segment _ _) (h.mass_pos label) h.positions.target
    dsimp only
    rw [hn, hsum]
    exact fun _ => right_mem_segment ℝ _ _
  · intro i hi
    exact update_pos_mem x s b i (convex_convexHull ℝ _) (h.mass_pos i) (h.positions.foreign i hi)
      (assigned_foreign_mean hf hi b.rows)

theorem IsolatedState.run {z col label : Fin k} {s : State k (Space k)}
    (h : IsolatedState z col label s) (hk : 2 ≤ k) (hcol : col ≠ z)
    (bs : List (Batch k)) (hb : ∀ b ∈ bs, FullBatch b) :
    IsolatedState z col label (run (replaySites z) s bs) := by
  induction bs generalizing s with
  | nil => exact h
  | cons b bs ih =>
    exact ih (h.update hk hcol b (hb _ (by simp))) (fun b' hm => hb b' (List.mem_cons_of_mem _ hm))

end IndexRecall.SphericalGap.EMeans
