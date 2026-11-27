import IndexRecall.SphericalCenters
import IndexRecall.SphericalVectors
import Mathlib.Analysis.InnerProductSpace.Continuous

/-! # A fixed cosine margin with the best document's list ranked last

`cosine_gap` gives a unit-vector example for every number of lists at least two.
`cosine_gap_near` gives the witnesses along an explicit perturbation of the anchors.
-/

noncomputable section

open Filter
open scoped Topology InnerProductSpace

namespace IndexRecall.SphericalGap

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Use the tip to route to its own list, keeping the other normalized means. -/
def retunedDirection (x : Fin k × Fin k → E) (z i : Fin k) : E :=
  if i = z then x (z, z) else meanDirection x i

/-- The canonical partition has exactly one address for every slot in every row. -/
theorem row_card (i : Fin k) :
    (Finset.univ.filter (fun p : Fin k × Fin k => p.1 = i)).card = k := by
  rw [← Finset.univ_product_univ, Finset.filter_product_left (fun j : Fin k => j = i)]
  simp [Finset.filter_eq']

/-- Every globally optimal labeling has the same list sizes. -/
theorem UniqueOptimum.list_card {x : Fin k × Fin k → E} (ho : UniqueOptimum x)
    (f : Fin k × Fin k → Fin k) (c : Fin k → E) (hc : ∀ j, ‖c j‖ = 1)
    (heq : sphericalLoss x f c = sphericalLoss x Prod.fst (meanDirection x)) (i : Fin k) :
    (Finset.univ.filter (fun p => f p = i)).card = k := by
  obtain ⟨e, he, _⟩ := (ho.equality_iff f c hc).mp heq
  obtain ⟨j, rfl⟩ := e.surjective i
  simpa only [he, e.apply_eq_iff_eq] using row_card j

/-- All search claims concern the same documents, query, and assignment `Prod.fst`.
Negating cosine turns the router's decreasing order into `FlatScan`'s increasing order. -/
structure CosineGap (x : Fin k × Fin k → E) (q : E) (z : Fin k) : Prop where
  database_size : Fintype.card (Fin k × Fin k) = k ^ 2
  list_size : ∀ i : Fin k, (Finset.univ.filter (fun p : Fin k × Fin k => p.1 = i)).card = k
  distinct : Function.Injective x
  documents_unit : ∀ p, ‖x p‖ = 1
  query_unit : ‖q‖ = 1
  optimum : UniqueOptimum x
  tip_score : ⟪q, x (z, z)⟫_ℝ = 1 / 2
  margin : ∀ b ≠ (z, z), ⟪q, x b⟫_ℝ + 1 / 10 ≤ ⟪q, x (z, z)⟫_ℝ
  mean_last : ∀ i ≠ z, ⟪q, meanDirection x z⟫_ℝ < ⟪q, meanDirection x i⟫_ℝ
  retuned_first : ∀ i ≠ z, ⟪q, retunedDirection x z i⟫_ℝ < ⟪q, retunedDirection x z z⟫_ℝ
  scans : ScanGap (fun p => -⟪q, x p⟫_ℝ)
    (fun i => -⟪q, meanDirection x i⟫_ℝ) (fun i => -⟪q, retunedDirection x z i⟫_ℝ) (z, z)

/-- The best document beats each possible runner-up by strictly more than 0.09. -/
theorem CosineGap.margin_gt {x : Fin k × Fin k → E} {q : E} {z : Fin k}
    (h : CosineGap x q z) (b : Fin k × Fin k) (hb : b ≠ (z, z)) :
    ⟪q, x b⟫_ℝ + 9 / 100 < ⟪q, x (z, z)⟫_ℝ := by
  linarith [h.margin b hb]

theorem CosineGap.retuned_unit {x : Fin k × Fin k → E} {q : E} {z : Fin k}
    (h : CosineGap x q z) (i : Fin k) : ‖retunedDirection x z i‖ = 1 := by
  unfold retunedDirection
  split_ifs
  · exact h.documents_unit _
  · exact h.optimum.unit_centers _

private structure StrictConditions (x : Fin k × Fin k → E) (q : E) (z : Fin k) : Prop where
  optimal : ∀ f, ¬Relabeled f → quality x f < quality x Prod.fst
  nonzero : ∀ i, rowSum x i ≠ 0
  last : ∀ i ≠ z, ⟪q, meanDirection x z⟫_ℝ < ⟪q, meanDirection x i⟫_ℝ
  first : ∀ i ≠ z, ⟪q, meanDirection x i⟫_ℝ < ⟪q, x (z, z)⟫_ℝ

open Vectors

private theorem base_conditions (hk : 2 ≤ k) (z : Fin k) :
    StrictConditions (base z) (query k) z := by
  have halign : 4 / 5 < ⟪tip z, anchor z z⟫_ℝ := by
    rw [tip_anchor]
    nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]
  have hr := repeated_routing hk z (query k) (tip z) (anchor z) (base z)
    (fun _ => rfl) (anchor_unit z) (by simp [query_anchor])
    (fun i hi => by simp [query_anchor, hi]) (query_tip z) halign
  refine ⟨?_, hr.1, hr.2.1, ?_⟩
  · exact repeated_strict z (anchor z) (tip z) (anchor_unit z) (tip_unit z) halign
      (fun _ _ h => base_cross z h)
  · simpa only [base, if_pos rfl] using hr.2.2

/-- The repeated-anchor instance already has a unique optimum over assignments and unit centers. -/
theorem repeated_optimum (hk : 2 ≤ k) (z : Fin k) : UniqueOptimum (base z) :=
  let h := base_conditions hk z
  uniqueOptimum_of_quality _ h.nonzero h.optimal

/-- After optimizing out the centers, there are only finitely many assignment scores.
Their strict comparisons and the routing comparisons persist in one neighborhood. -/
private theorem conditions_near (x : ℝ → Fin k × Fin k → E) (q : E) (z : Fin k)
    (hx : ∀ p, Continuous (fun t => x t p)) (h : StrictConditions (x 0) q z) :
    ∀ᶠ t in 𝓝 (0 : ℝ), StrictConditions (x t) q z := by
  have hquality (f : Fin k × Fin k → Fin k) : Continuous (fun t => quality (x t) f) :=
    continuous_finset_sum _ fun _ _ => (continuous_finset_sum _ fun p _ => hx p).norm
  have hsum (i : Fin k) : ContinuousAt (fun t => rowSum (x t) i) 0 :=
    (continuous_finset_sum _ fun j _ => hx (i, j)).continuousAt
  have hmean (i : Fin k) : ContinuousAt (fun t => ⟪q, meanDirection (x t) i⟫_ℝ) 0 :=
    continuousAt_const.inner (((hsum i).norm.inv₀ (norm_ne_zero_iff.mpr (h.nonzero i))).smul
      (hsum i))
  have htip : ContinuousAt (fun t => ⟪q, x t (z, z)⟫_ℝ) 0 :=
    continuousAt_const.inner (hx (z, z)).continuousAt
  have opt : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ f, ¬Relabeled f → quality (x t) f < quality (x t) Prod.fst :=
    eventually_all.mpr fun f => eventually_imp_distrib_left.mpr fun hf =>
      (hquality f).continuousAt.eventually_lt (hquality Prod.fst).continuousAt (h.optimal f hf)
  have nz : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ i, rowSum (x t) i ≠ 0 :=
    eventually_all.mpr fun i => (hsum i).eventually_ne (h.nonzero i)
  have last : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ i ≠ z,
      ⟪q, meanDirection (x t) z⟫_ℝ < ⟪q, meanDirection (x t) i⟫_ℝ :=
    eventually_all.mpr fun i => eventually_imp_distrib_left.mpr fun hi =>
      (hmean z).eventually_lt (hmean i) (h.last i hi)
  have first : ∀ᶠ t in 𝓝 (0 : ℝ), ∀ i ≠ z, ⟪q, meanDirection (x t) i⟫_ℝ < ⟪q, x t (z, z)⟫_ℝ :=
    eventually_all.mpr fun i => eventually_imp_distrib_left.mpr fun hi =>
      (hmean i).eventually_lt htip (h.first i hi)
  filter_upwards [opt, nz, last, first] with t ho hn hl hf using ⟨ho, hn, hl, hf⟩

/-- Every sufficiently small positive rotation along the explicit address axes works. -/
theorem cosine_gap_near (hk : 2 ≤ k) (z : Fin k) :
    ∀ᶠ t in 𝓝[>] (0 : ℝ), CosineGap (site z t) (query k) z := by
  have h0 : StrictConditions (site z 0) (query k) z := by
    rw [site_zero]
    exact base_conditions hk z
  have h := conditions_near (site z) (query k) z (site_continuous z) h0
  filter_upwards [h.filter_mono nhdsWithin_le_nhds,
    (self_mem_nhdsWithin : ∀ᶠ t in 𝓝[>] (0 : ℝ), 0 < t)] with t ht hpos
  have first (i : Fin k) (hi : i ≠ z) :
      ⟪query k, retunedDirection (site z t) z i⟫_ℝ <
        ⟪query k, retunedDirection (site z t) z z⟫_ℝ := by
    simpa only [retunedDirection, if_neg hi, if_pos rfl] using ht.first i hi
  refine ⟨by simp [sq], row_card, site_injective z hpos, site_unit z t, query_unit k,
    uniqueOptimum_of_quality _ ht.nonzero ht.optimal, ?_, site_margin z t, ht.last, first,
    scan_gap _ _ _ _ ?_ (fun i hi => neg_lt_neg (ht.last i hi))
      (fun i hi => neg_lt_neg (first i hi))⟩
  · rw [site_tip, query_tip]
  · intro b hb
    exact neg_lt_neg (by linarith [site_margin z t b hb])

/-- For every `k ≥ 2` there are `k²` distinct unit documents and a unit query whose
unique best document has a cosine lead greater than 0.09. Its globally optimal
spherical k-means list ranks last at the normalized means and first when routed
by the tip. `CosineGap.scans` records both exact scan costs and their ratio. -/
theorem cosine_gap (k : ℕ) (hk : 2 ≤ k) :
    ∃ (x : Fin k × Fin k → Space k) (q : Space k) (z : Fin k), CosineGap x q z := by
  let z : Fin k := ⟨0, by omega⟩
  obtain ⟨t, ht⟩ := (cosine_gap_near hk z).exists
  exact ⟨site z t, query k, z, ht⟩

end IndexRecall.SphericalGap

#print axioms IndexRecall.SphericalGap.cosine_gap
