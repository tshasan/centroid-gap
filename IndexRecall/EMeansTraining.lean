import IndexRecall.EMeansSeedEpoch
import IndexRecall.EMeansSeedCounting
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-! # Closing a spherical e-means run on the cosine-gap construction -/

noncomputable section
open Filter
open scoped Topology InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The pinned cosine step schedule, indexed by completed logical batches across epochs. -/
def scheduledAlpha (initial : ℝ) (step total : ℕ) : ℝ :=
  let fraction : ℝ := (min step total : ℕ) / (max 1 total : ℕ)
  initial * (1 / 10 + (9 / 10) * ((1 + Real.cos (Real.pi * fraction)) / 2))

theorem scheduledAlpha_bounds {initial : ℝ} (hi : 0 < initial) (hu : initial ≤ 1)
    (step total : ℕ) : 0 < scheduledAlpha initial step total ∧ scheduledAlpha initial step total ≤ 1 := by
  let θ := Real.pi * ((min step total : ℕ) / (max 1 total : ℕ) : ℝ)
  have hc := Real.cos_mem_Icc θ
  have hlo : 0 < 1 / 10 + (9 / 10) * ((1 + Real.cos θ) / 2) := by linarith [hc.1]
  have hhi : 1 / 10 + (9 / 10) * ((1 + Real.cos θ) / 2) ≤ 1 := by linarith [hc.2]
  exact ⟨mul_pos hi hlo, (mul_le_mul_of_nonneg_left hhi hi.le).trans (by simpa using hu)⟩

/-- Addresses receiving a particular label in the moving assignment log of the last pass. -/
def passRows (log : List ((Fin k × Fin k) × Fin k)) (i : Fin k) : List (Fin k × Fin k) :=
  (log.filter (fun item => decide (item.2 = i))).map Prod.fst

/-- The raw closing position is the last-pass mean for an occupied list and the
terminal EMA position for an empty list. -/
def closingPosition (x : Fin k × Fin k → E) (pos : Fin k → E)
    (log : List ((Fin k × Fin k) × Fin k)) (i : Fin k) : E :=
  let rows := passRows log i
  if rows.length = 0 then pos i else (rows.length : ℝ)⁻¹ • (rows.map x).sum

/-- Normalize the closing position, including the fallback for an empty list. -/
def finalize (x : Fin k × Fin k → E) (pos : Fin k → E)
    (log : List ((Fin k × Fin k) × Fin k)) (i : Fin k) : E :=
  unitDir (closingPosition x pos log i)

private theorem passRows_map (rows : List (Address k)) (f : Address k → Fin k) (i : Fin k) :
    passRows (rows.map (fun p => (p, f p))) i = assigned rows f i := by
  simp [passRows, assigned, List.filter_map, List.map_map, Function.comp_def]

/-- A single batch containing the whole database meets the per-batch revival condition. -/
theorem full_batch_balanced {rows : List (Address k)} (hfull : rows.Perm Finset.univ.toList) :
    BalancedRows rows := by
  have hc (i : Fin k) : (assigned rows Prod.fst i).length = k := by
    simpa only [Finset.length_toList, row_card] using (full_row_perm rows hfull i).length_eq
  exact fun i j => (hc i).trans (hc j).symm

/-- A complete final pass in any order gives the constructed normalized means whenever
the moving snapshots gave the owner labels. This equality removes every EMA weight. -/
theorem finalize_full (hk : 0 < k) (x : Fin k × Fin k → E) (pos : Fin k → E)
    (rows : List (Fin k × Fin k)) (hfull : rows.Perm Finset.univ.toList) :
    finalize x pos (rows.map (fun p => (p, p.1))) = meanDirection x := by
  funext i
  obtain ⟨hn, hs⟩ := full_row_stats x hfull (f := Prod.fst) (label := i) (fun _ => Iff.rfl)
  simpa only [finalize, closingPosition, passRows_map, hn, if_neg hk.ne', hs, meanDirection]
    using normalized_mean_eq hk x i

/-- A full last epoch visits each address once, across any ordering and batch boundaries. -/
def FullPass (last : List (Batch k)) : Prop :=
  (last.flatMap Batch.rows).Perm Finset.univ.toList

/-- Follow the no-split branch of the pinned trainer. Revival checks continue across
epochs. Closing uses only the moving labels recorded in the last epoch. -/
def train (τ : ℝ) (x : Address k → E) (init : Fin k → E) (prior last : List (Batch k)) :
    Option (Fin k → E) :=
  let start := beginState init
  (checkedRun τ x start 0 (prior ++ last)).map fun terminal =>
    finalize x terminal.pos (captured x (run x start prior) last)

private theorem closed_assignment {x : Address k → E} (hk : 0 < k) (B : Basin x)
    (p : Address k) (j : Fin k) (hj : j ≠ p.1) :
    ⟪x p, meanDirection x j⟫_ℝ < ⟪x p, meanDirection x p.1⟫_ℝ := by
  let avg (i : Fin k) := (k : ℝ)⁻¹ • rowSum x i
  have havg (i : Fin k) : avg i ∈ B.cell i := by
    have hm := (B.convex i).centerMass_mem (t := Finset.univ)
      (z := fun j : Fin k => x (i, j)) (w := fun _ => (1 : ℝ))
      (fun _ _ => zero_le_one) (by simpa using (show (0 : ℝ) < k by exact_mod_cast hk))
      (fun j _ => B.documents (i, j))
    simpa only [Finset.centerMass, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, mul_one, one_smul, avg, rowSum] using hm
  simpa only [avg, normalized_mean_eq hk, meanDirection] using B.separates avg havg p j hj

/-- Properties of the actual closing representatives, including the fresh assignment
used to populate IVF after training and the one-based query rank. -/
structure FinalIndex (x : Address k → E) (q : E) (z : Fin k) (c : Fin k → E) : Prop where
  means : c = meanDirection x
  membership : ∀ (p : Address k) j, j ≠ p.1 → ⟪x p, c j⟫_ℝ < ⟪x p, c p.1⟫_ℝ
  last : ∀ i ≠ z, ⟪q, c z⟫_ℝ < ⟪q, c i⟫_ℝ
  rank : FlatScan.rank (fun i => -⟪q, c i⟫_ℝ) z = k

/-- Guarantees for a fixed initialization. The step bounds are part of `Batch`.
`ChecksClear` tests the evolving masses at every fifth logical update. -/
structure WarmStart (x : Address k → E) (q : E) (z : Fin k) (init : Fin k → E) : Prop where
  snapshots : ∀ (prior : List (Batch k)) (p : Address k) j, j ≠ p.1 →
    let c := (run x (beginState init) prior).pos
    ⟪x p, unitDir (c j)⟫_ℝ < ⟪x p, unitDir (c p.1)⟫_ℝ
  labels : ∀ bs, captured x (beginState init) bs = (bs.flatMap Batch.rows).map (fun p => (p, p.1))
  disabled : ∀ bs, ChecksClear 0 x (beginState init) 0 bs
  balanced : ∀ τ ≤ 1, ∀ bs, (∀ b ∈ bs, BalancedRows b.rows) → ChecksClear τ x (beginState init) 0 bs
  finalized : ∀ τ prior last, FullPass last → ChecksClear τ x (beginState init) 0 (prior ++ last) →
    ∃ c, train τ x init prior last = some c ∧ FinalIndex x q z c

/-- With revival disabled, the outcome is independent of the batch order and step schedule. -/
theorem WarmStart.without_revival {x : Address k → E} {q : E} {z : Fin k} {init : Fin k → E}
    (h : WarmStart x q z init) (prior last : List (Batch k)) (hp : FullPass last) :
    ∃ c, train 0 x init prior last = some c ∧ FinalIndex x q z c :=
  h.finalized 0 prior last hp (h.disabled _)

/-- Equal owner counts in each batch give the same outcome with a threshold up to one. -/
theorem WarmStart.with_balanced_batches {x : Address k → E} {q : E} {z : Fin k} {init : Fin k → E}
    (h : WarmStart x q z init) (τ : ℝ) (hτ : τ ≤ 1) (prior last : List (Batch k))
    (hb : ∀ b ∈ prior ++ last, BalancedRows b.rows) (hp : FullPass last) :
    ∃ c, train τ x init prior last = some c ∧ FinalIndex x q z c :=
  h.finalized τ prior last hp (h.balanced τ hτ _ hb)

theorem warmStart_of_cells {x : Address k → E} {q : E} {z : Fin k}
    (hg : CosineGap x q z) (hd : ∀ p, x p ∈ cell x p.1)
    (init : Fin k → E) (hi : ∀ i, unitDir (init i) ∈ cell x i) : WarmStart x q z init := by
  let B : Basin x := ⟨cell x, cell_convex x, hd, owner_strict x hg.documents_unit⟩
  have hs := begin_stable B init hi
  refine ⟨fun bs => B.separates _ (run_invariant B _ hs bs).1,
    captured_eq_owners B _ hs, checksClear_zero B _ hs 0, ?_, ?_⟩
  · intro τ hτ bs hb
    exact checksClear_balanced B _ hs (begin_equalMasses _) 0 bs hb hτ
  · intro τ prior last hp hc
    refine ⟨meanDirection x, ?_, ⟨rfl, closed_assignment (Fin.pos z) B, hg.mean_last, hg.scans.mean_rank⟩⟩
    rw [train, checkedRun_of_checksClear _ _ _ _ _ hc]
    simp only [Option.map_some, captured_eq_owners B _ (run_invariant B _ hs prior),
      finalize_full (Fin.pos z) x _ _ hp]

open Vectors

/-- For all sufficiently small data perturbations, the same radius of 0.01 per initial
center works uniformly over the allowed training schedules. -/
theorem emeans_gap_near (hk : 2 ≤ k) (z : Fin k) :
    ∀ᶠ t in 𝓝[>] (0 : ℝ), CosineGap (site z t) (query k) z ∧
      ∀ init : Fin k → Space k, (∀ i, ‖init i - anchor z i‖ < 1 / 100) →
        WarmStart (site z t) (query k) z init := by
  filter_upwards [cosine_gap_near hk z, cells_near hk z] with t hg hc
  exact ⟨hg, fun init hi => warmStart_of_cells hg hc.1 init (fun i => hc.2 i _ (hi i))⟩

/-- At the explicit replay parameter, every raw initialization in the anchor balls
preserves the globally optimal rows throughout training. Last-epoch finalization
places the winning document's list strictly last. `WarmStart` gives the batch,
step-size and revival hypotheses, as well as the labels of every moving snapshot. -/
theorem emeans_gap (hk : 2 ≤ k) (z : Fin k) :
    let x := site z (replayParam k)
    CosineGap x (query k) z ∧ ∀ init : Fin k → Space k,
      (∀ i, ‖init i - anchor z i‖ < 1 / 100) → WarmStart x (query k) z init :=
  ⟨replay_gap hk z, fun init hi => warmStart_of_cells (replay_gap hk z)
    (replay_geometry hk z).1 init (fun i => (replay_geometry hk z).2 i _ (hi i))⟩

/-- The warm starts contain a neighborhood in the topology of labeled center tables. -/
theorem warm_starts_nhds (hk : 2 ≤ k) (z : Fin k) :
    ∀ᶠ init in 𝓝 (anchor z), WarmStart (site z (replayParam k)) (query k) z init := by
  have h : ∀ᶠ init : Fin k → Space k in 𝓝 (anchor z), ∀ i, ‖init i - anchor z i‖ < 1 / 100 := by
    apply eventually_all.mpr
    intro i
    exact ((continuous_apply i).sub continuous_const).norm.continuousAt.eventually_lt
      continuousAt_const (by simp)
  exact h.mono fun init hi => (emeans_gap hk z).2 init hi

/-! ## Uniform document initialization -/

/-- The closing index guarantees for the target row alone. The rank uses the winning
document's actual list after a fresh assignment to the finalized representatives. -/
structure TargetRowIndex (x : Address k → E) (q : E) (z label : Fin k) (c : Fin k → E) : Prop where
  mean : c label = meanDirection x z
  membership : ∀ p, assignDirection x c p = label ↔ p.1 = z
  last : ∀ i ≠ label, ⟪q, c label⟫_ℝ < ⟪q, c i⟫_ℝ
  rank : FlatScan.rank (fun i => -⟪q, c i⟫_ℝ) (assignDirection x c (z,z)) = k

private theorem closing_isolated {z col label : Fin k} {s : State k (Space k)}
    (h : IsolatedState z col label s) (hk : 2 ≤ k) (hcol : col ≠ z)
    (b : Batch k) (hb : FullBatch b) :
    let x := replaySites z
    let raw := closingPosition x (update x s b).pos (captured x s [b])
    IsolatedPositions z col label raw ∧ raw label = (k : ℝ)⁻¹ • rowSum x z := by
  intro x raw
  let f := predict x s
  have hf (p) : f p = label ↔ p.1 = z := h.positions.labels hk hcol p
  have hraw (i) : raw i = if (assigned b.rows f i).length = 0 then (update x s b).pos i
      else ((assigned b.rows f i).length : ℝ)⁻¹ • ((assigned b.rows f i).map x).sum := by
    simp only [raw, closingPosition, captured, List.append_nil, passRows_map, f]
  obtain ⟨hn, hsum⟩ := full_row_stats x hb hf
  have htarget : raw label = (k : ℝ)⁻¹ • rowSum x z := by
    rw [hraw, hn, if_neg (Fin.pos z).ne', hsum]
  refine ⟨⟨?_, ?_⟩, htarget⟩
  · rw [htarget]
    exact right_mem_segment ℝ _ _
  · intro i hi
    rw [hraw]
    split_ifs with hempty
    · exact (h.update hk hcol b hb).positions.foreign i hi
    · exact assigned_foreign_mean hf hi b.rows hempty

private theorem train_isolated {z col label : Fin k} {init : Fin k → Space k}
    (h : IsolatedState z col label (beginState init)) (hk : 2 ≤ k) (hcol : col ≠ z)
    (prior : List (Batch k)) (last : Batch k) (hp : ∀ b ∈ prior, FullBatch b) (hl : FullBatch last) :
    ∃ c, train 0 (replaySites z) init prior [last] = some c ∧
      TargetRowIndex (replaySites z) (query k) z label c := by
  let x := replaySites z
  let s := run x (beginState init) prior
  let raw := closingPosition x (update x s last).pos (captured x s [last])
  let c := fun i => unitDir (raw i)
  have hs := h.run hk hcol prior hp
  obtain ⟨hraw, hmean⟩ := closing_isolated hs hk hcol last hl
  have hlabels (p) : assignDirection x c p = label ↔ p.1 = z := hraw.labels hk hcol p
  have hlast (i) (hi : i ≠ label) : ⟪query k, c label⟫_ℝ < ⟪query k, c i⟫_ℝ :=
    query_prefers_hull hk hcol hraw.target (hraw.foreign i hi)
  have hall (b) (hb : b ∈ prior ++ [last]) : FullBatch b := by
    rcases List.mem_append.mp hb with hm | hm
    · exact hp b hm
    · exact List.mem_singleton.mp hm ▸ hl
  have hc : ChecksClear 0 x (beginState init) 0 (prior ++ [last]) := by
    intro n _ _ _ i
    rw [zero_mul]
    exact ((h.run hk hcol _ (fun b hb => hall b (List.mem_of_mem_take hb))).mass_pos i).le
  refine ⟨c, ?_, ⟨?_, hlabels, hlast, ?_⟩⟩
  · rw [train, checkedRun_of_checksClear _ _ _ _ _ hc, run_append]
    rfl
  · change unitDir (raw label) = meanDirection x z
    rw [show raw label = _ from hmean]
    exact normalized_mean_eq (Fin.pos z) x z
  · rw [(hlabels (z,z)).mpr rfl]
    simpa using FlatScan.rank_last (score := fun i => -⟪query k, c i⟫_ℝ)
      (fun i hi => neg_lt_neg (hlast i hi))

/-- For any labeling of a sampled set with one ordinary target seed, that seed's label
receives exactly the target row at every full-batch snapshot and in the final index.
The last epoch is explicit, so there is at least one epoch. `Batch` permits any
step in `[0,1]`, including every positive EMA step at most one. -/
theorem random_start_training (hk : 2 ≤ k) (z : Fin k) (sample : Finset (Address k))
    (he : SeedEvent z sample) (labels : Fin k ≃ ↥sample) :
    let x := replaySites z
    let init := fun i => x (labels i).val
    ∃ label, (labels label).val.1 = z ∧
      (∀ bs, (∀ b ∈ bs, FullBatch b) → ∀ p,
        predict x (run x (beginState init) bs) p = label ↔ p.1 = z) ∧
      (∀ prior last, (∀ b ∈ prior, FullBatch b) → FullBatch last →
        ∃ c, train 0 x init prior [last] = some c ∧ TargetRowIndex x (query k) z label c) := by
  intro x init
  obtain ⟨col, hcol, hm, huniq⟩ := he
  let label := labels.symm ⟨(z,col), hm⟩
  have hseed : (labels label).val = (z,col) := congrArg Subtype.val (labels.apply_symm_apply _)
  have hforeign (i) (hi : i ≠ label) : (labels i).val.1 ≠ z := by
    intro hz
    apply hi
    exact labels.injective (Subtype.ext ((huniq _ (labels i).property hz).trans hseed.symm))
  have h := begin_isolated (fun i => (labels i).val) hseed hforeign
  exact ⟨label, congrArg Prod.fst hseed,
    fun bs hb p => (h.run hk hcol bs hb).positions.labels hk hcol p,
    train_isolated h hk hcol⟩

/-- A lower bound for strict last place under uniform sampling without replacement.
The event on the right requires success for *every* labeling of the sampled IDs. -/
theorem random_start_last_probability (hk : 2 ≤ k) (z : Fin k)
    (prior : List (Batch k)) (last : Batch k) (hp : ∀ b ∈ prior, FullBatch b) (hl : FullBatch last) :
    let x := replaySites z
    seedProbability k ≤ uniformSeedProbability k (fun sample => ∀ labels : Fin k ≃ ↥sample,
      ∃ c, train 0 x (fun i => x (labels i).val) prior [last] = some c ∧
        FlatScan.rank (fun i => -⟪query k, c i⟫_ℝ) (assignDirection x c (z,z)) = k) := by
  intro x
  apply seedProbability_le_of_event z
  intro sample _ he labels
  obtain ⟨_, _, _, hfinal⟩ := random_start_training hk z sample he labels
  obtain ⟨c, hc, hi⟩ := hfinal prior last hp hl
  exact ⟨c, hc, hi.rank⟩

end IndexRecall.SphericalGap.EMeans

#print axioms IndexRecall.SphericalGap.EMeans.emeans_gap
#print axioms IndexRecall.SphericalGap.EMeans.random_start_last_probability
