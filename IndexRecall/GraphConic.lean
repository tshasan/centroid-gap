import Mathlib.Analysis.Convex.Cone.InnerDual
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # Conic-hull navigability and micro-shard navigation (Open Problems A + B)

This file formalizes:
- **Closedness of finitely generated cones** (Carathéodory + finite-dimensionality) and
  finite-dimensional **Farkas' Lemma** in `EuclideanSpace ℝ (Fin d)`.
- **Theorem A2**: A directed graph on a finite database `S ⊂ ℝᵈ` has zero suboptimal
  local maxima for greedy MIPS search across *every* query `q ∈ ℝᵈ` if and only if every
  displacement `w - u` lies in the conic hull of `{v - u : v ∈ N(u)}`.
- **Theorem B2 ("Farkas for Shards")**: A directed graph over micro-shards scored by each
  shard's exact support function `h_c(q) = max_{x ∈ C_c} ⟪x, q⟫` has zero suboptimal local
  maxima for every `q ∈ ℝᵈ` if and only if for every shard `c`, every `x ∈ C_c` (equivalently
  every extreme point of `C_c`), and every `w ∈ S`, `w - x` lies in the conic hull of
  `{z - x : z ∈ U_c}` where `U_c = C_c ∪ ⋃_{c' ∈ N(c)} C_{c'}`.
- **Theorem B3 (Quantitative Practical Bound)**: With a finite tip subset `T_c ⊆ C_c`
  (intra-shard hull error `ε`) and bounded-degree neighbor shards (inter-shard conic
  residual `η`), every local maximum `c*` of greedy tip-scored search satisfies
  `max_{x ∈ C_{c*}} ⟪x, q⟫ ≥ max_{w ∈ S} ⟪w, q⟫ - (ε + η) * ‖q‖` for every `q ∈ ℝᵈ`,
  with an exact top-1 recovery corollary under margin `> (ε + η) * ‖q‖`.
-/

noncomputable section

open scoped RealInnerProductSpace BigOperators

namespace IndexRecall.GraphConic

variable {d : ℕ}

abbrev Vec (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- The finitely generated convex cone spanned by a finite set of vectors `G ⊆ ℝᵈ`:
non-negative linear combinations `∑ g ∈ G, w g • g`. -/
def finCone (G : Finset (Vec d)) : Set (Vec d) :=
  {y | ∃ w : Vec d → ℝ, (∀ g ∈ G, 0 ≤ w g) ∧ ∑ g ∈ G, w g • g = y}

theorem zero_mem_finCone (G : Finset (Vec d)) : (0 : Vec d) ∈ finCone G :=
  ⟨fun _ => 0, fun _ _ => le_rfl, by simp⟩

theorem add_mem_finCone {G : Finset (Vec d)} {x y : Vec d}
    (hx : x ∈ finCone G) (hy : y ∈ finCone G) : x + y ∈ finCone G := by
  rcases hx with ⟨wx, hwx, rfl⟩
  rcases hy with ⟨wy, hwy, rfl⟩
  refine ⟨fun g => wx g + wy g, fun g hg => add_nonneg (hwx g hg) (hwy g hg), ?_⟩
  simp [add_smul, Finset.sum_add_distrib]

theorem smul_mem_finCone {G : Finset (Vec d)} {c : ℝ} (hc : 0 ≤ c) {x : Vec d}
    (hx : x ∈ finCone G) : c • x ∈ finCone G := by
  rcases hx with ⟨w, hw, rfl⟩
  refine ⟨fun g => c * w g, fun g hg => mul_nonneg hc (hw g hg), ?_⟩
  rw [Finset.smul_sum]
  refine Finset.sum_congr rfl fun g _ => ?_
  rw [mul_smul]

theorem mem_finCone_of_mem {G : Finset (Vec d)} {g : Vec d} (hg : g ∈ G) :
    g ∈ finCone G := by
  classical
  refine ⟨fun x => if x = g then 1 else 0, fun x _ => by dsimp only; split_ifs <;> norm_num, ?_⟩
  simp [hg]

/-- Conic Carathéodory lemma: any vector in `finCone G` is a non-negative linear combination
of a *linearly independent* subset `I ⊆ G`. -/
theorem mem_finCone_exists_linearIndependent {G : Finset (Vec d)} {y : Vec d}
    (hy : y ∈ finCone G) :
    ∃ I : Finset (Vec d), I ⊆ G ∧ LinearIndependent ℝ (fun i : I => (i : Vec d)) ∧
      y ∈ finCone I := by
  classical
  -- Choose a subset I ⊆ G of minimum cardinality with y ∈ finCone I.
  let P : Finset (Vec d) → Prop := fun I => I ⊆ G ∧ y ∈ finCone I
  have hP : ∃ n, ∃ I, I.card = n ∧ P I := ⟨G.card, G, rfl, Finset.Subset.rfl, hy⟩
  let n₀ := Nat.find hP
  obtain ⟨I, hIcard, hIsub, hIy⟩ := Nat.find_spec hP
  refine ⟨I, hIsub, ?_, hIy⟩
  -- Suppose I is linearly dependent; derive a contradiction with minimality of I.card.
  by_contra hdep
  rw [Fintype.linearIndependent_iff] at hdep
  push_neg at hdep
  obtain ⟨c, hc_sum, i₀, hi₀_ne⟩ := hdep
  -- Flip sign of c if needed so some coefficient is strictly positive.
  have hex_pos : ∃ c' : I → ℝ, (∑ i : I, c' i • (i : Vec d) = 0) ∧ ∃ j : I, 0 < c' j := by
    rcases lt_or_gt_of_ne hi₀_ne with hlt | hgt
    · refine ⟨fun i => -c i, ?_, i₀, neg_pos.mpr hlt⟩
      simp_rw [neg_smul, Finset.sum_neg_distrib, hc_sum, neg_zero]
    · exact ⟨c, hc_sum, i₀, hgt⟩
  obtain ⟨c', hc'_sum, j₀, hj₀_pos⟩ := hex_pos
  rcases hIy with ⟨w, hw, hw_sum⟩
  -- Check if any i ∈ I already has w i = 0; if so we can drop i immediately.
  by_cases hzero : ∃ i ∈ I, w i = 0
  · obtain ⟨i, hiI, hwi0⟩ := hzero
    have hcard_lt : (I.erase i).card < Nat.find hP := by
      rw [← hIcard]; exact Finset.card_erase_lt_of_mem hiI
    have hmem_erase : y ∈ finCone (I.erase i) := by
      refine ⟨w, fun g hg => hw g (Finset.mem_of_mem_erase hg), ?_⟩
      rw [← hw_sum, ← Finset.add_sum_erase I _ hiI, hwi0, zero_smul, zero_add]
    exact Nat.find_min hP hcard_lt ⟨I.erase i, rfl, (Finset.erase_subset i I).trans hIsub, hmem_erase⟩
  · push_neg at hzero
    -- All w i > 0 on I. Choose j* ∈ I with c' j* > 0 minimizing w j / c' j.
    let posSet : Finset I := Finset.univ.filter fun i : I => 0 < c' i
    have hpos_ne : posSet.Nonempty := ⟨j₀, by simp [posSet, hj₀_pos]⟩
    obtain ⟨j_star, hj_mem, hj_min⟩ :=
      posSet.exists_min_image (fun i : I => w i / c' i) hpos_ne
    have hcj_pos : 0 < c' j_star := (Finset.mem_filter.mp hj_mem).2
    let t : ℝ := w j_star / c' j_star
    have ht_nonneg : 0 ≤ t := div_nonneg (hw j_star.1 j_star.2) hcj_pos.le
    -- Define new weights w'(g) = w g - t * c' g on I; then w' ≥ 0 and w'(j_star) = 0.
    let w' : Vec d → ℝ := fun g =>
      if hg : g ∈ I then w g - t * c' ⟨g, hg⟩ else 0
    have hw'_nonneg : ∀ g ∈ I, 0 ≤ w' g := by
      intro g hg
      simp only [w', dif_pos hg]
      by_cases hcg : 0 < c' ⟨g, hg⟩
      · have hle : t ≤ w g / c' ⟨g, hg⟩ :=
          hj_min ⟨g, hg⟩ (by simp [posSet, hcg])
        have hmul : t * c' ⟨g, hg⟩ ≤ w g := (le_div_iff₀ hcg).mp hle
        linarith
      · push_neg at hcg
        have hmul : t * c' ⟨g, hg⟩ ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ht_nonneg hcg
        linarith [hw g hg]
    have hw'_zero : w' j_star.1 = 0 := by
      simp only [w', dif_pos j_star.2, t]
      rw [div_mul_cancel₀ _ hcj_pos.ne']
      ring
    have hw'_sum : ∑ g ∈ I, w' g • g = y := by
      have h1 : ∑ g ∈ I, w' g • g = ∑ i : I, (w i.1 - t * c' i) • (i : Vec d) := by
        rw [← Finset.sum_coe_sort I (fun g => w' g • g)]
        refine Finset.sum_congr rfl fun i _ => ?_
        simp [w', i.2]
      rw [h1]
      have h2 : ∑ i : I, (w i.1 - t * c' i) • (i : Vec d) =
          (∑ i : I, w i.1 • (i : Vec d)) - t • (∑ i : I, c' i • (i : Vec d)) := by
        simp_rw [sub_smul, Finset.sum_sub_distrib, Finset.smul_sum, mul_smul]
      rw [h2, hc'_sum, smul_zero, sub_zero]
      rw [Finset.sum_coe_sort I (fun g => w g • g), hw_sum]
    -- Drop j_star from I to get a smaller spanning set.
    have hcard_lt : (I.erase j_star.1).card < Nat.find hP := by
      rw [← hIcard]; exact Finset.card_erase_lt_of_mem j_star.2
    have hmem_erase : y ∈ finCone (I.erase j_star.1) := by
      refine ⟨w', fun g hg => hw'_nonneg g (Finset.mem_of_mem_erase hg), ?_⟩
      rw [← hw'_sum, ← Finset.add_sum_erase I _ j_star.2, hw'_zero, zero_smul, zero_add]
    exact Nat.find_min hP hcard_lt
      ⟨I.erase j_star.1, rfl, (Finset.erase_subset _ I).trans hIsub, hmem_erase⟩

/-- The cone spanned by a linearly independent finite set `I` in `ℝᵈ` is closed. -/
private theorem isClosed_finCone_of_linearIndependent {I : Finset (Vec d)}
    (hI : LinearIndependent ℝ (fun i : I => (i : Vec d))) :
    IsClosed (finCone I) := by
  -- The linear map L : (I → ℝ) → Vec d, c ↦ ∑ i, c i • i is an injective linear map
  -- from a finite-dimensional space to a finite-dimensional space, hence a closed embedding.
  let L : (I → ℝ) →ₗ[ℝ] Vec d :=
    { toFun := fun c => ∑ i : I, c i • (i : Vec d)
      map_add' := fun a b => by simp [add_smul, Finset.sum_add_distrib]
      map_smul' := fun r a => by
        simp only [RingHom.id_apply, Pi.smul_apply, smul_eq_mul, Finset.smul_sum, mul_smul] }
  have hLinj : Function.Injective L := by
    rw [injective_iff_map_eq_zero]
    intro c hc
    funext i
    exact Fintype.linearIndependent_iff.mp hI c hc i
  -- Every injective linear map from a finite-dimensional normed space is a closed embedding!
  let e : (I → ℝ) ≃L[ℝ] LinearMap.range L :=
    (LinearEquiv.ofInjective L hLinj).toContinuousLinearEquiv
  have hS_closed : IsClosed {c : I → ℝ | ∀ i : I, 0 ≤ c i} := by
    have : {c : I → ℝ | ∀ i : I, 0 ≤ c i} = ⋂ i : I, {c : I → ℝ | 0 ≤ c i} := by
      ext c; simp
    rw [this]
    exact isClosed_iInter fun i => isClosed_Ici.preimage (continuous_apply i)
  -- Express finCone I as the image of {c | ∀ i, 0 ≤ c i} under L.
  have himg : finCone I = L '' {c : I → ℝ | ∀ i : I, 0 ≤ c i} := by
    classical
    ext y
    constructor
    · rintro ⟨w, hw, rfl⟩
      refine ⟨fun i : I => w i.1, fun i => hw i.1 i.2, ?_⟩
      simp only [L, LinearMap.coe_mk, AddHom.coe_mk]
      exact Finset.sum_coe_sort I (fun g => w g • g)
    · rintro ⟨c, hc, rfl⟩
      refine ⟨fun g => if hg : g ∈ I then c ⟨g, hg⟩ else 0, ?_, ?_⟩
      · intro g hg; simp [dif_pos hg, hc ⟨g, hg⟩]
      · simp only [L, LinearMap.coe_mk, AddHom.coe_mk]
        rw [← Finset.sum_coe_sort I]
        refine Finset.sum_congr rfl fun i _ => by simp [i.2]
  rw [himg]
  -- Image under L equals subtype inclusion of image under the homeomorphism e.
  have hrange_closed : IsClosed (LinearMap.range L : Set (Vec d)) :=
    Submodule.closed_of_finiteDimensional _
  have himg_sub : L '' {c : I → ℝ | ∀ i : I, 0 ≤ c i} =
      Subtype.val '' (e '' {c : I → ℝ | ∀ i : I, 0 ≤ c i}) := by
    ext y
    simp only [Set.mem_image]
    constructor
    · rintro ⟨c, hc, rfl⟩
      exact ⟨e c, ⟨c, hc, rfl⟩, rfl⟩
    · rintro ⟨⟨z, hz⟩, ⟨c, hc, hce⟩, rfl⟩
      refine ⟨c, hc, ?_⟩
      exact congrArg Subtype.val hce
  rw [himg_sub]
  have he_closed : IsClosed (e '' {c : I → ℝ | ∀ i : I, 0 ≤ c i}) :=
    e.toHomeomorph.isClosed_image.mpr hS_closed
  exact hrange_closed.isClosedEmbedding_subtypeVal.isClosed_iff_image_isClosed.mp he_closed

/-- **Weyl-Minkowski closedness**: every finitely generated cone in `ℝᵈ` is closed. -/
theorem isClosed_finCone (G : Finset (Vec d)) : IsClosed (finCone G) := by
  classical
  have hunion : finCone G =
      ⋃ I ∈ G.powerset.filter (fun I : Finset (Vec d) =>
          LinearIndependent ℝ (fun i : I => (i : Vec d))),
        finCone I := by
    ext y
    simp only [Set.mem_iUnion, Finset.mem_filter, Finset.mem_powerset, exists_prop]
    constructor
    · intro hy
      obtain ⟨I, hIsub, hIli, hIy⟩ := mem_finCone_exists_linearIndependent hy
      exact ⟨I, ⟨hIsub, hIli⟩, hIy⟩
    · rintro ⟨I, ⟨hIsub, _⟩, ⟨w, hw, rfl⟩⟩
      refine ⟨fun g => if g ∈ I then w g else 0, fun g _ => ?_, ?_⟩
      · dsimp only
        split_ifs with hg
        · exact hw g hg
        · exact le_rfl
      · rw [← Finset.sum_subset hIsub]
        · refine Finset.sum_congr rfl fun g hg => by simp [hg]
        · intro g _ hgI; simp [hgI]
  rw [hunion]
  refine Set.Finite.isClosed_biUnion (Finset.finite_toSet _) fun I hI => ?_
  exact isClosed_finCone_of_linearIndependent (Finset.mem_filter.mp hI).2

/-- Package `finCone G` as a Mathlib `ProperCone ℝ (Vec d)`. -/
def finProperCone (G : Finset (Vec d)) : ProperCone ℝ (Vec d) where
  carrier := finCone G
  add_mem' hx hy := add_mem_finCone hx hy
  zero_mem' := zero_mem_finCone G
  smul_mem' c _ hx := smul_mem_finCone c.2 hx
  isClosed' := isClosed_finCone G

/-- **Finite-Dimensional Farkas' Lemma** in `EuclideanSpace ℝ (Fin d)`:
`b ∈ finCone G` iff every `q` with `⟪g, q⟫ ≤ 0` for all `g ∈ G` satisfies `⟪b, q⟫ ≤ 0`. -/
theorem mem_finCone_iff_dual {G : Finset (Vec d)} {b : Vec d} :
    b ∈ finCone G ↔ ∀ q : Vec d, (∀ g ∈ G, ⟪g, q⟫ ≤ 0) → ⟪b, q⟫ ≤ 0 := by
  constructor
  · rintro ⟨w, hw, rfl⟩ q hq
    rw [sum_inner]
    refine Finset.sum_nonpos fun g hg => ?_
    rw [real_inner_smul_left]
    exact mul_nonpos_of_nonneg_of_nonpos (hw g hg) (hq g hg)
  · intro hdual
    by_contra hb
    have hb' : b ∉ finProperCone G := hb
    obtain ⟨y, hy_cone, hy_b⟩ := (finProperCone G).hyperplane_separation' hb'
    -- Set q = -y so ⟪g, q⟫ ≤ 0 for all g ∈ G while ⟪b, q⟫ > 0.
    have hq_nonpos : ∀ g ∈ G, ⟪g, -y⟫ ≤ 0 := by
      intro g hg
      rw [inner_neg_right, neg_nonpos]
      exact hy_cone g (mem_finCone_of_mem hg)
    have hbq_pos : 0 < ⟪b, -y⟫ := by
      rw [inner_neg_right, neg_pos]
      exact hy_b
    linarith [hdual (-y) hq_nonpos]

/-! ## Theorem A2: Universal Zero-Query-Log MIPS Navigability (Conic-Hull Pruning) -/

/-- A vertex `u ∈ S` is a MIPS local maximum for query `q` in a directed graph with
out-neighborhoods `N` if no out-neighbor scores strictly higher than `u`. -/
def IsLocalMax (N : Vec d → Finset (Vec d)) (u q : Vec d) : Prop :=
  ∀ v ∈ N u, ⟪v, q⟫ ≤ ⟪u, q⟫

/-- **Theorem A2 (Exact Characterization of Zero-Query-Log MIPS Navigability):**
A directed graph `(S, N)` on a finite database `S ⊂ ℝᵈ` has zero suboptimal local maxima
for *every* query `q ∈ ℝᵈ` if and only if for every `u ∈ S` and every `w ∈ S`,
`w - u ∈ finCone {v - u | v ∈ N u}`. -/
theorem universal_mips_navigable_iff (S : Finset (Vec d)) (N : Vec d → Finset (Vec d)) :
    (∀ q : Vec d, ∀ u ∈ S, IsLocalMax N u q → ∀ w ∈ S, ⟪w, q⟫ ≤ ⟪u, q⟫) ↔
      (∀ u ∈ S, ∀ w ∈ S, w - u ∈ finCone ((N u).image (fun v => v - u))) := by
  classical
  constructor
  · intro hnav u hu w hw
    rw [mem_finCone_iff_dual]
    intro q hq
    have hloc : IsLocalMax N u q := by
      intro v hv
      have h1 : ⟪v - u, q⟫ ≤ 0 := hq (v - u) (Finset.mem_image_of_mem _ hv)
      rw [inner_sub_left] at h1
      linarith
    have hopt := hnav q u hu hloc w hw
    rw [inner_sub_left]
    linarith
  · intro hcone q u hu hloc w hw
    have hmem := hcone u hu w hw
    have hdual := (mem_finCone_iff_dual.mp hmem) q (by
      intro g hg
      rcases Finset.mem_image.mp hg with ⟨v, hv, rfl⟩
      rw [inner_sub_left]
      linarith [hloc v hv])
    rw [inner_sub_left] at hdual
    linarith

/-! ## Theorem B2: Exact Sharded Conic-Hull Characterization ("Farkas for Shards") -/

/-- Support function (exact maximum inner product) of a nonempty finite shard `C`. -/
def shardSupport (C : Finset (Vec d)) (hC : C.Nonempty) (q : Vec d) : ℝ :=
  (C.image (fun x => ⟪x, q⟫)).max' (hC.image _)

theorem le_shardSupport {C : Finset (Vec d)} (hC : C.Nonempty) (q : Vec d)
    {x : Vec d} (hx : x ∈ C) : ⟪x, q⟫ ≤ shardSupport C hC q :=
  Finset.le_max' (C.image (fun x => ⟪x, q⟫)) (⟪x, q⟫) (Finset.mem_image_of_mem (fun x => ⟪x, q⟫) hx)

theorem exists_eq_shardSupport {C : Finset (Vec d)} (hC : C.Nonempty) (q : Vec d) :
    ∃ x ∈ C, ⟪x, q⟫ = shardSupport C hC q := by
  have hmem := Finset.max'_mem (C.image (fun x => ⟪x, q⟫)) (hC.image _)
  rcases Finset.mem_image.mp hmem with ⟨x, hx, hxq⟩
  exact ⟨x, hx, hxq⟩

theorem shardSupport_le_iff {C : Finset (Vec d)} (hC : C.Nonempty) (q : Vec d) (r : ℝ) :
    shardSupport C hC q ≤ r ↔ ∀ x ∈ C, ⟪x, q⟫ ≤ r := by
  simp [shardSupport]

/-- The union of shard `c` and all its neighbor shards `c' ∈ N c`. -/
def shardUnion {ι : Type*} (C : ι → Finset (Vec d)) (N : ι → Finset ι) (c : ι) :
    Finset (Vec d) :=
  C c ∪ (N c).biUnion C

/-- **Theorem B2 (Farkas for Shards):**
Let `{C c}_{c : ι}` be a finite collection of nonempty micro-shards with coarse directed
graph `N : ι → Finset ι`. Greedy shard search scored by the exact support function
`h_c(q) = max_{x ∈ C c} ⟪x, q⟫` has zero suboptimal local maxima for *every* query `q ∈ ℝᵈ`
if and only if for every shard `c`, every `x ∈ C c`, every shard `c'`, and every `w ∈ C c'`,
`w - x ∈ finCone {z - x | z ∈ U_c}` where `U_c = C c ∪ ⋃_{nb ∈ N c} C nb`. -/
theorem sharded_conic_navigable_iff {ι : Type*} [DecidableEq ι]
    (C : ι → Finset (Vec d)) (hC : ∀ c, (C c).Nonempty) (N : ι → Finset ι) :
    (∀ q : Vec d, ∀ c : ι,
        (∀ nb ∈ N c, shardSupport (C nb) (hC nb) q ≤ shardSupport (C c) (hC c) q) →
        ∀ c' : ι, shardSupport (C c') (hC c') q ≤ shardSupport (C c) (hC c) q) ↔
      (∀ c : ι, ∀ x ∈ C c, ∀ c' : ι, ∀ w ∈ C c',
        w - x ∈ finCone ((shardUnion C N c).image (fun z => z - x))) := by
  classical
  constructor
  · intro hnav c x hx c' w hw
    rw [mem_finCone_iff_dual]
    intro q hq
    -- Every z ∈ U_c satisfies ⟪z, q⟫ ≤ ⟪x, q⟫.
    have hU : ∀ z ∈ shardUnion C N c, ⟪z, q⟫ ≤ ⟪x, q⟫ := by
      intro z hz
      have h1 := hq (z - x) (Finset.mem_image_of_mem _ hz)
      rw [inner_sub_left] at h1
      linarith
    -- Since x ∈ C c ⊆ U_c, h_c(q) = ⟪x, q⟫.
    have hhc_eq : shardSupport (C c) (hC c) q = ⟪x, q⟫ := by
      apply le_antisymm
      · rw [shardSupport_le_iff]
        intro y hy
        exact hU y (Finset.mem_union_left _ hy)
      · exact le_shardSupport (hC c) q hx
    -- Every neighbor shard satisfies h_nb(q) ≤ h_c(q).
    have hloc : ∀ nb ∈ N c, shardSupport (C nb) (hC nb) q ≤ shardSupport (C c) (hC c) q := by
      intro nb hnb
      rw [hhc_eq, shardSupport_le_iff]
      intro z hz
      exact hU z (Finset.mem_union_right _ (Finset.mem_biUnion.mpr ⟨nb, hnb, hz⟩))
    have hopt := hnav q c hloc c'
    rw [hhc_eq] at hopt
    have hw_le : ⟪w, q⟫ ≤ ⟪x, q⟫ := (le_shardSupport (hC c') q hw).trans hopt
    rw [inner_sub_left]
    linarith
  · intro hcone q c hloc c'
    obtain ⟨x, hx, hxq⟩ := exists_eq_shardSupport (hC c) q
    rw [shardSupport_le_iff]
    intro w hw
    have hmem := hcone c x hx c' w hw
    have hdual := (mem_finCone_iff_dual.mp hmem) q (by
      intro g hg
      rcases Finset.mem_image.mp hg with ⟨z, hz, rfl⟩
      rw [inner_sub_left]
      rcases Finset.mem_union.mp hz with hzC | hzN
      · linarith [le_shardSupport (hC c) q hzC, hxq]
      · rcases Finset.mem_biUnion.mp hzN with ⟨nb, hnb, hznb⟩
        have h1 : ⟪z, q⟫ ≤ shardSupport (C nb) (hC nb) q := le_shardSupport (hC nb) q hznb
        have h2 : shardSupport (C nb) (hC nb) q ≤ shardSupport (C c) (hC c) q := hloc nb hnb
        linarith)
    rw [inner_sub_left] at hdual
    linarith

/-! ## Theorem B3: Quantitative Practical Bound (Bounded Tips + Approximate Conic Pruning) -/

/-- Convex combination membership in a finite tip set `T`. -/
def InFinConv (T : Finset (Vec d)) (y : Vec d) : Prop :=
  ∃ a : Vec d → ℝ, (∀ t ∈ T, 0 ≤ a t) ∧ (∑ t ∈ T, a t = 1) ∧ ∑ t ∈ T, a t • t = y

/-- **Theorem B3 (Quantitative Practical Bound):**
Suppose each shard `c` retains a finite nonempty tip subset `T c ⊆ C c` such that:
1. Every vector `w ∈ C c'` lies within Euclidean distance `ε` of `conv(T c')`, and
2. At shard `c*`, for every tip `x ∈ T c*` and every tip `t ∈ T c'`, the displacement
   `t - x` lies within Euclidean distance `η` of `finCone {z - x | z ∈ U_{c*}^T}`
   where `U_{c*}^T = T c* ∪ ⋃_{nb ∈ N c*} T nb`.

Then at *any* local maximum `c*` of greedy tip-scored search (`h_{T_nb}(q) ≤ h_{T_{c*}}(q)`
for all `nb ∈ N c*`), for *every* query `q ∈ ℝᵈ` and *every* database point `w ∈ C c'`:
`⟪w, q⟫ ≤ shardSupport (T c*) q + (ε + η) * ‖q‖ ≤ shardSupport (C c*) q + (ε + η) * ‖q‖`. -/
theorem approximate_sharded_conic_bound {ι : Type*} [DecidableEq ι]
    (C T : ι → Finset (Vec d))
    (hTsub : ∀ c, T c ⊆ C c)
    (hTne : ∀ c, (T c).Nonempty)
    (hCne : ∀ c, (C c).Nonempty)
    (N : ι → Finset ι)
    {ε η : ℝ}
    (h_eps : ∀ c', ∀ w ∈ C c', ∃ y, InFinConv (T c') y ∧ ‖w - y‖ ≤ ε)
    (h_eta : ∀ c, ∀ x ∈ T c, ∀ c', ∀ t ∈ T c',
      ∃ u ∈ finCone ((shardUnion T N c).image (fun z => z - x)), ‖(t - x) - u‖ ≤ η)
    (q : Vec d) (c_star : ι)
    (hloc : ∀ nb ∈ N c_star, shardSupport (T nb) (hTne nb) q ≤ shardSupport (T c_star) (hTne c_star) q)
    (c' : ι) (w : Vec d) (hw : w ∈ C c') :
    ⟪w, q⟫ ≤ shardSupport (T c_star) (hTne c_star) q + (ε + η) * ‖q‖ ∧
      shardSupport (T c_star) (hTne c_star) q ≤ shardSupport (C c_star) (hCne c_star) q := by
  classical
  -- Second inequality: T c* ⊆ C c* implies h_{T c*}(q) ≤ h_{C c*}(q).
  have hsub_le : shardSupport (T c_star) (hTne c_star) q ≤ shardSupport (C c_star) (hCne c_star) q := by
    rw [shardSupport_le_iff]
    intro t ht
    exact le_shardSupport (hCne c_star) q (hTsub c_star ht)
  refine ⟨?_, hsub_le⟩
  -- Pick x* ∈ T c* attaining h_{T c*}(q).
  obtain ⟨x_star, hx_mem, hx_eq⟩ := exists_eq_shardSupport (hTne c_star) q
  -- Every z ∈ U_{c*}^T satisfies ⟪z - x*, q⟫ ≤ 0.
  have hU_nonpos : ∀ g ∈ (shardUnion T N c_star).image (fun z => z - x_star), ⟪g, q⟫ ≤ 0 := by
    intro g hg
    rcases Finset.mem_image.mp hg with ⟨z, hz, rfl⟩
    rw [inner_sub_left]
    rcases Finset.mem_union.mp hz with hzT | hzN
    · linarith [le_shardSupport (hTne c_star) q hzT, hx_eq]
    · rcases Finset.mem_biUnion.mp hzN with ⟨nb, hnb, hznb⟩
      have h1 := le_shardSupport (hTne nb) q hznb
      have h2 := hloc nb hnb
      linarith
  -- Every tip t ∈ T c' satisfies ⟪t - x*, q⟫ ≤ η * ‖q‖.
  have htip_bound : ∀ t ∈ T c', ⟪t - x_star, q⟫ ≤ η * ‖q‖ := by
    intro t ht
    obtain ⟨u, hu_cone, hu_norm⟩ := h_eta c_star x_star hx_mem c' t ht
    have hu_le : ⟪u, q⟫ ≤ 0 := (mem_finCone_iff_dual.mp hu_cone) q hU_nonpos
    have hcs : ⟪(t - x_star) - u, q⟫ ≤ ‖(t - x_star) - u‖ * ‖q‖ :=
      real_inner_le_norm _ _
    have hmul : ‖(t - x_star) - u‖ * ‖q‖ ≤ η * ‖q‖ :=
      mul_le_mul_of_nonneg_right hu_norm (norm_nonneg q)
    have hsplit : ⟪t - x_star, q⟫ = ⟪u, q⟫ + ⟪(t - x_star) - u, q⟫ := by
      rw [inner_sub_left (t - x_star) u q]; ring
    linarith
  -- Approximate w ∈ C c' by y ∈ conv(T c') with ‖w - y‖ ≤ ε.
  obtain ⟨y, ⟨a, ha_nonneg, ha_sum, hy_eq⟩, hwy_norm⟩ := h_eps c' w hw
  have hy_bound : ⟪y - x_star, q⟫ ≤ η * ‖q‖ := by
    have hsub : y - x_star = ∑ t ∈ T c', a t • (t - x_star) := by
      simp_rw [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, ha_sum, one_smul, hy_eq]
    rw [hsub, sum_inner]
    have hsum_le : ∑ t ∈ T c', ⟪a t • (t - x_star), q⟫ ≤ ∑ t ∈ T c', a t * (η * ‖q‖) := by
      refine Finset.sum_le_sum fun t ht => ?_
      rw [real_inner_smul_left]
      exact mul_le_mul_of_nonneg_left (htip_bound t ht) (ha_nonneg t ht)
    rw [← Finset.sum_mul, ha_sum, one_mul] at hsum_le
    exact hsum_le
  have hwy_inner : ⟪w - y, q⟫ ≤ ε * ‖q‖ :=
    (real_inner_le_norm (w - y) q).trans (mul_le_mul_of_nonneg_right hwy_norm (norm_nonneg q))
  have hdecomp : ⟪w, q⟫ - ⟪x_star, q⟫ = ⟪w - y, q⟫ + ⟪y - x_star, q⟫ := by
    rw [inner_sub_left, inner_sub_left]; ring
  linarith

/-- **Corollary (Exact Recovery Under Margin):**
Under the hypotheses of Theorem B3, if the unique optimal shard `c_opt` contains `w*`
with margin `⟪w*, q⟫ > ⟪x, q⟫ + (ε + η) * ‖q‖` over all `x` in every other shard `c ≠ c_opt`,
then any local maximum `c_star` of greedy tip-scored search must satisfy `c_star = c_opt`. -/
theorem exact_shard_recovery_under_margin {ι : Type*} [DecidableEq ι]
    (C T : ι → Finset (Vec d))
    (hTsub : ∀ c, T c ⊆ C c)
    (hTne : ∀ c, (T c).Nonempty)
    (hCne : ∀ c, (C c).Nonempty)
    (N : ι → Finset ι)
    {ε η : ℝ}
    (h_eps : ∀ c', ∀ w ∈ C c', ∃ y, InFinConv (T c') y ∧ ‖w - y‖ ≤ ε)
    (h_eta : ∀ c, ∀ x ∈ T c, ∀ c', ∀ t ∈ T c',
      ∃ u ∈ finCone ((shardUnion T N c).image (fun z => z - x)), ‖(t - x) - u‖ ≤ η)
    (q : Vec d) (c_opt c_star : ι) (w_star : Vec d) (hw : w_star ∈ C c_opt)
    (hmargin : ∀ c ≠ c_opt, ∀ x ∈ C c, ⟪x, q⟫ + (ε + η) * ‖q‖ < ⟪w_star, q⟫)
    (hloc : ∀ nb ∈ N c_star, shardSupport (T nb) (hTne nb) q ≤ shardSupport (T c_star) (hTne c_star) q) :
    c_star = c_opt := by
  by_contra hne
  have hbound := approximate_sharded_conic_bound C T hTsub hTne hCne N h_eps h_eta q c_star hloc c_opt w_star hw
  obtain ⟨x, hx, hxq⟩ := exists_eq_shardSupport (hCne c_star) q
  have hlt := hmargin c_star hne x hx
  linarith [hbound.1, hbound.2]

/-! ## Explicit Greedy Hill-Climbing State Machine (`L = 1` MIPS & Sharded Search) -/

/-- One strict-ascent step of greedy (`L = 1`) hill climbing on a finite graph `(V, N)`
with objective `score : α → ℝ`: move from `u` to a neighbor `v ∈ N u` with `score u < score v`. -/
def GreedyStep {α : Type*} (N : α → Finset α) (score : α → ℝ) (u v : α) : Prop :=
  v ∈ N u ∧ score u < score v

/-- Multi-step execution of greedy (`L = 1`) hill climbing. -/
inductive GreedyReach {α : Type*} (N : α → Finset α) (score : α → ℝ) : α → α → Prop
  | refl (u : α) : GreedyReach N score u u
  | step {u v w : α} : GreedyStep N score u v → GreedyReach N score v w → GreedyReach N score u w

/-- Terminal state of greedy hill climbing: no neighbor strictly improves `score`. -/
def GreedyTerminal {α : Type*} (N : α → Finset α) (score : α → ℝ) (u : α) : Prop :=
  ∀ v ∈ N u, score v ≤ score u

theorem greedyReach_score_le {α : Type*} {N : α → Finset α} {score : α → ℝ}
    {u v : α} (h : GreedyReach N score u v) : score u ≤ score v := by
  induction h with
  | refl _ => exact le_rfl
  | step hstep _ ih => exact le_trans (le_of_lt hstep.2) ih

/-- **Termination of Greedy (`L = 1`) Hill Climbing on Any Finite Graph:**
If `N` stays inside a finite set `U` (`∀ u ∈ U, N u ⊆ U`), then from every starting
vertex `u ∈ U`, there exists a reachable terminal state `u_star ∈ U` (`GreedyTerminal N score u_star`). -/
theorem greedy_terminates {α : Type*} [DecidableEq α]
    (U : Finset α) (N : α → Finset α) (score : α → ℝ)
    (hclosed : ∀ u ∈ U, N u ⊆ U) :
    ∀ u ∈ U, ∃ u_star ∈ U, GreedyReach N score u u_star ∧ GreedyTerminal N score u_star := by
  -- Strong induction on the number of vertices in U strictly outscoring u.
  have haux : ∀ n : ℕ, ∀ u ∈ U,
      (U.filter (fun w => score u < score w)).card ≤ n →
      ∃ u_star ∈ U, GreedyReach N score u u_star ∧ GreedyTerminal N score u_star := by
    intro n
    induction n with
    | zero =>
      intro u hu hcard
      refine ⟨u, hu, GreedyReach.refl u, ?_⟩
      intro v hv
      by_contra hlt
      push_neg at hlt
      have hv_mem : v ∈ U.filter (fun w => score u < score w) :=
        Finset.mem_filter.mpr ⟨hclosed u hu hv, hlt⟩
      have hpos : 0 < (U.filter (fun w => score u < score w)).card :=
        Finset.card_pos.mpr ⟨v, hv_mem⟩
      omega
    | succ k ih =>
      intro u hu hcard
      by_cases hterm : GreedyTerminal N score u
      · exact ⟨u, hu, GreedyReach.refl u, hterm⟩
      · unfold GreedyTerminal at hterm
        push_neg at hterm
        rcases hterm with ⟨v, hvN, hlt⟩
        have hvU : v ∈ U := hclosed u hu hvN
        have hsub : U.filter (fun w => score v < score w) ⊂
            U.filter (fun w => score u < score w) := by
          rw [Finset.ssubset_iff_of_subset]
          · refine ⟨v, Finset.mem_filter.mpr ⟨hvU, hlt⟩, ?_⟩
            simp
          · intro w hw
            rcases Finset.mem_filter.mp hw with ⟨hwU, hwv⟩
            exact Finset.mem_filter.mpr ⟨hwU, hlt.trans hwv⟩
        have hcard_lt : (U.filter (fun w => score v < score w)).card <
            (U.filter (fun w => score u < score w)).card :=
          Finset.card_lt_card hsub
        obtain ⟨u_star, hu_starU, hreach, hterm'⟩ := ih v hvU (by omega)
        exact ⟨u_star, hu_starU, GreedyReach.step ⟨hvN, hlt⟩ hreach, hterm'⟩
  intro u hu
  exact haux U.card u hu (Finset.card_le_card (Finset.filter_subset _ U))

/-- **End-to-End Point-Level Greedy `L = 1` MIPS Exactness (Theorem B1 + Termination):**
On any finite dataset `S` with a conic-pruned graph `N` closed inside `S`, every greedy
walk starting from any `u₀ ∈ S` terminates at a global MIPS maximizer, and at least one
such terminating trace exists. -/
theorem greedy_mips_exact_from_any_start
    (S : Finset (Vec d)) (N : Vec d → Finset (Vec d))
    (hclosed : ∀ u ∈ S, N u ⊆ S)
    (hcone : ∀ u ∈ S, ∀ v ∈ S, v - u ∈ finCone ((N u).image (fun z => z - u)))
    (q : Vec d) (u₀ : Vec d) (hu₀ : u₀ ∈ S) :
    (∃ u_star ∈ S, GreedyReach N (fun x => ⟪x, q⟫) u₀ u_star ∧
      GreedyTerminal N (fun x => ⟪x, q⟫) u_star ∧
      ∀ v ∈ S, ⟪v, q⟫ ≤ ⟪u_star, q⟫) ∧
    (∀ u_star ∈ S, GreedyReach N (fun x => ⟪x, q⟫) u₀ u_star →
      GreedyTerminal N (fun x => ⟪x, q⟫) u_star →
      ∀ v ∈ S, ⟪v, q⟫ ≤ ⟪u_star, q⟫) := by
  have hexact : ∀ u_star ∈ S, GreedyTerminal N (fun x => ⟪x, q⟫) u_star →
      ∀ v ∈ S, ⟪v, q⟫ ≤ ⟪u_star, q⟫ :=
    fun u_star hu hterm =>
      (universal_mips_navigable_iff S N).mpr hcone q u_star hu hterm
  obtain ⟨u_star, hu_star, hreach, hterm⟩ :=
    greedy_terminates S N (fun x => ⟪x, q⟫) hclosed u₀ hu₀
  exact ⟨⟨u_star, hu_star, hreach, hterm, hexact u_star hu_star hterm⟩,
    fun u_star hu _ hterm => hexact u_star hu hterm⟩

/-- **End-to-End Sharded Greedy `L = 1` Search Exactness (Theorem B3 + Termination):**
On any finite shard index set `I` with tip subsets `T c ⊆ C c` and approximate conic-pruned
shard graph `N` closed in `I`, greedy tip-scored hill climbing from any start shard `c₀ ∈ I`
terminates in finitely many steps, and *every* terminal shard `c_star` satisfies the
`(ε + η)‖q‖` bound (and exact recovery `c_star = c_opt` under the margin condition). -/
theorem greedy_sharded_terminates_with_bound {ι : Type*} [DecidableEq ι]
    (I : Finset ι) (C T : ι → Finset (Vec d))
    (hTsub : ∀ c, T c ⊆ C c)
    (hTne : ∀ c, (T c).Nonempty)
    (hCne : ∀ c, (C c).Nonempty)
    (N : ι → Finset ι)
    (hclosed : ∀ c ∈ I, N c ⊆ I)
    {ε η : ℝ}
    (h_eps : ∀ c', ∀ w ∈ C c', ∃ y, InFinConv (T c') y ∧ ‖w - y‖ ≤ ε)
    (h_eta : ∀ c, ∀ x ∈ T c, ∀ c', ∀ t ∈ T c',
      ∃ u ∈ finCone ((shardUnion T N c).image (fun z => z - x)), ‖(t - x) - u‖ ≤ η)
    (q : Vec d) (c₀ : ι) (hc₀ : c₀ ∈ I) :
    ∃ c_star ∈ I,
      GreedyReach N (fun c => shardSupport (T c) (hTne c) q) c₀ c_star ∧
      GreedyTerminal N (fun c => shardSupport (T c) (hTne c) q) c_star ∧
      ∀ c', ∀ w ∈ C c',
        ⟪w, q⟫ ≤ shardSupport (T c_star) (hTne c_star) q + (ε + η) * ‖q‖ := by
  obtain ⟨c_star, hc_star, hreach, hterm⟩ :=
    greedy_terminates I N (fun c => shardSupport (T c) (hTne c) q) hclosed c₀ hc₀
  exact ⟨c_star, hc_star, hreach, hterm, fun c' w hw =>
    (approximate_sharded_conic_bound C T hTsub hTne hCne N h_eps h_eta q c_star hterm c' w hw).1⟩

end IndexRecall.GraphConic
