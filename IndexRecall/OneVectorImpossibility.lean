import IndexRecall.FlatScan
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # One-vector router impossibility vs. two-tip exact recovery

For every number of lists `k` and list size `m ≥ 2`, we construct an explicit
balanced database of `k * m` pairwise distinct unit vectors in `ℝ^{k+1}`
partitioned into `k` lists of size `m`, together with `2 * k` unit queries
(`±e_i` for `i : Fin k`), each having a unique true nearest neighbor in list `i`
with constant cosine margin at least `1/2`, such that:

1. **Two-tip exact recovery (`twoTip_returns_all`, `twoTip_totalWork`)**:
   Storing two unit tips per list (the list's own `±e_j` documents) and routing
   by `max_{t ∈ T_j} ⟪q, t⟫` ranks the true owner strictly first (`rank = 1`,
   `1` list probed) on every query, achieving 100% Recall@1 at probe budget `1`
   with total work `2 * k + m` per query (counting all `2 * k` tip inner
   products plus the `m` scanned documents). At `m = k` (`N = k²` documents),
   total work is `3 * k = 3√N` on every query.

2. **Impossibility for any 1-vector affine / power-diagram / Euclidean router**
   (`affine_answered_le`, `affine_fullRecall_budget_ge`, `affine_sum_tieRank_ge`,
   `affine_sum_work_ge`, `powerDist_eq_negAffine`):
   For *arbitrary* per-list vectors `r : Fin k → ℝ^{k+1}`, *arbitrary* scalar
   biases `b : Fin k → ℝ` (including all power-diagram / Euclidean-distance
   routers `‖q - c_j‖² - w_j`), and *arbitrary* query-independent tie-breaking:
   - At any fixed probe budget `p ≥ 1`, at most `k + 2 * p - 1` of the `2 * k`
     queries are recovered (Recall@1 ≤ `1/2 + (2p - 1) / (2k)`).
   - Achieving 100% Recall@1 at a fixed probe budget `p` requires `k + 1 ≤ 2 * p`
     (at least `⌈(k + 1) / 2⌉` probes).
   - Even with an oracle per-query stopping budget, the sum of lists probed
     across the `2 * k` queries satisfies `k * (k + 3) ≤ 2 * ∑ probes` (average
     at least `(k + 3) / 4` lists probed), and total `FlatScan.work` satisfies
     `4 * k² + m * k * (k + 3) ≤ 2 * ∑ work`. At `m = k` (`N = k²`), average
     total work is at least `(k² + 7 * k) / 4 = (N + 7√N) / 4` — an `Ω(√N)`
     separation against the two-tip router's `3√N`.

3. **Sharper impossibility for pure 1-vector linear routers (`b = 0`) under
   tie-inclusive `FlatScan.probed`** (`linear_pair_minProbedCount_ge`,
   `linear_recall_le_half`, `linear_sum_minProbedCount_ge`, `linear_sum_work_ge`):
   - Every antipodal pair `±e_i` requires at least `k + 1` total probed lists.
   - At any probe budget `p` with `2 * p ≤ k`, at most `k` of the `2 * k`
     queries are recovered (Recall@1 ≤ `1/2`).
   - Average lists probed is at least `(k + 1) / 2`, and at `m = k` average
     total work is at least `(k² + 3 * k) / 2 = (N + 3√N) / 2`.
-/

noncomputable section

open scoped RealInnerProductSpace BigOperators Classical

namespace IndexRecall.OneVectorImpossibility

abbrev Space (k : ℕ) := EuclideanSpace ℝ (Fin (k + 1))

/-- List axis basis vector `e_i ∈ ℝ^{k+1}` (`i < k`). -/
def axis (i : Fin k) : Space k := EuclideanSpace.single i.castSucc 1

/-- Shared slack axis `e_k ∈ ℝ^{k+1}` used to place `m - 2` distinct filler
unit vectors on each list's semicircle between `+e_i` and `-e_i`. -/
def slack (k : ℕ) : Space k := EuclideanSpace.single (Fin.last k) 1

/-- Coordinate along `e_i` for slot `t : Fin m`:
slot `0` gets `+1`, slot `1` gets `-1`, and slot `t ≥ 2` gets `1 / t ∈ (0, 1/2]`. -/
def slotCoord {m : ℕ} (t : Fin m) : ℝ :=
  if t.val = 0 then 1
  else if t.val = 1 then -1
  else 1 / (t.val : ℝ)

/-- Balanced database of `k * m` unit vectors in `ℝ^{k+1}` partitioned into `k`
lists of size `m`. -/
def doc {k m : ℕ} (x : Fin k × Fin m) : Space k :=
  slotCoord x.2 • axis x.1 + Real.sqrt (1 - (slotCoord x.2) ^ 2) • slack k

/-- Sign factor for the two antipodal queries `±e_i`. -/
def querySign (s : Bool) : ℝ := if s then 1 else -1

/-- The `2 * k` unit queries: `(i, true) ↦ +e_i`, `(i, false) ↦ -e_i`. -/
def query {k : ℕ} (q : Fin k × Bool) : Space k := querySign q.2 • axis q.1

/-- True target document for query `q` (slot `0` for `+e_i`, slot `1` for `-e_i`). -/
def targetDoc {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) : Fin k × Fin m :=
  (q.1, if q.2 then ⟨0, by omega⟩ else ⟨1, by omega⟩)

/-- Two-tip router score using list `j`'s own extreme documents `±e_j`. -/
def tipScore {k m : ℕ} (hm : 2 ≤ m) (q : Space k) (j : Fin k) : ℝ :=
  max ⟪q, doc (j, (⟨0, by omega⟩ : Fin m))⟫ ⟪q, doc (j, (⟨1, by omega⟩ : Fin m))⟫

/-- Negated two-tip score for `FlatScan` (smaller is better). -/
def negTipScore {k m : ℕ} (hm : 2 ≤ m) (q : Space k) (j : Fin k) : ℝ :=
  -tipScore hm q j

/-- Total work for a two-tip router: `2 * k` tip dot products + scanned docs. -/
def twoTipWork {k : ℕ} (B : Type*) [Fintype B] (score : Fin k → ℝ) (p : ℕ) : ℕ :=
  2 * k + (FlatScan.candidates B score p).card

/-- Negated 1-vector affine router score `-(⟪q, r j⟫ + b j)` (smaller is better). -/
def negAffineScore {k : ℕ} (r : Fin k → Space k) (b : Fin k → ℝ)
    (q : Space k) (j : Fin k) : ℝ :=
  -(⟪q, r j⟫ + b j)

/-- Power-diagram / squared-Euclidean router score `‖q - c j‖² - w j` (smaller is better). -/
def powerDistScore {k : ℕ} (c : Fin k → Space k) (w : Fin k → ℝ)
    (q : Space k) (j : Fin k) : ℝ :=
  ‖q - c j‖ ^ 2 - w j

/-- Number of lists with score ≤ score of list `i`: exact lower bound on
`(FlatScan.probed score p).card` whenever `i ∈ FlatScan.probed score p`. -/
def minProbedCount {k : ℕ} (score : Fin k → ℝ) (i : Fin k) : ℕ :=
  (Finset.univ.filter (fun j => score j ≤ score i)).card

/-- List `j` beats list `i` under `score` with query-independent tie-breaker `lt`. -/
def beats {k : ℕ} (score : Fin k → ℝ) (lt : Fin k → Fin k → Prop) (j i : Fin k) : Prop :=
  score j < score i ∨ (score j = score i ∧ lt j i)

/-- 1-indexed rank of list `i` under `score` with query-independent tie-breaker `lt`. -/
def tieRank {k : ℕ} (score : Fin k → ℝ) (lt : Fin k → Fin k → Prop) (i : Fin k) : ℕ :=
  1 + (Finset.univ.filter (fun j => beats score lt j i)).card

/-! ### Basic geometry of the construction -/

theorem axis_inner {k : ℕ} (i j : Fin k) :
    ⟪axis i, axis j⟫ = if i = j then 1 else 0 := by
  simp [axis, EuclideanSpace.inner_single_left]

theorem axis_inner_slack {k : ℕ} (i : Fin k) :
    ⟪axis i, slack k⟫ = 0 := by
  have hne : i.castSucc ≠ Fin.last k := Fin.castSucc_ne_last i
  simp [axis, slack, EuclideanSpace.inner_single_left, hne]

theorem slack_inner_axis {k : ℕ} (i : Fin k) :
    ⟪slack k, axis i⟫ = 0 := by
  rw [real_inner_comm]
  exact axis_inner_slack i

theorem slack_inner_slack (k : ℕ) :
    ⟪slack k, slack k⟫ = 1 := by
  simp [slack, EuclideanSpace.inner_single_left]

theorem axis_unit {k : ℕ} (i : Fin k) : ‖axis i‖ = 1 := by
  simp [axis]

theorem slotCoord_sq_le_one {m : ℕ} (t : Fin m) : (slotCoord t) ^ 2 ≤ 1 := by
  unfold slotCoord
  split_ifs with h0 h1
  · norm_num
  · norm_num
  · have ht2 : (2 : ℝ) ≤ (t.val : ℝ) := by
      exact_mod_cast (show 2 ≤ t.val by omega)
    have hpos : (0 : ℝ) < (t.val : ℝ) := by linarith
    have hle : 1 / (t.val : ℝ) ≤ 1 := by
      rw [div_le_one hpos]
      linarith
    have hnn : 0 ≤ 1 / (t.val : ℝ) := by positivity
    nlinarith

theorem slotCoord_ne_zero {m : ℕ} (t : Fin m) : slotCoord t ≠ 0 := by
  unfold slotCoord
  split_ifs with h0 h1
  · norm_num
  · norm_num
  · have hpos : (0 : ℝ) < (t.val : ℝ) := by
      exact_mod_cast (show 0 < t.val by omega)
    exact div_ne_zero one_ne_zero (ne_of_gt hpos)

theorem slotCoord_le_half {m : ℕ} {t : Fin m} (ht : t.val ≠ 0) : slotCoord t ≤ 1 / 2 := by
  unfold slotCoord
  rw [if_neg ht]
  split_ifs with h1
  · norm_num
  · have ht2 : (2 : ℝ) ≤ (t.val : ℝ) := by
      exact_mod_cast (show 2 ≤ t.val by omega)
    have hpos : (0 : ℝ) < (t.val : ℝ) := by linarith
    rw [div_le_div_iff₀ hpos (by norm_num : (0 : ℝ) < 2)]
    linarith

theorem slotCoord_injective {m : ℕ} : Function.Injective (slotCoord (m := m)) := by
  intro t u h
  have ht0 := h
  unfold slotCoord at ht0
  split_ifs at ht0 with h1 h2 h3 h4 h5 h6
  · exact Fin.ext (by omega)
  · norm_num at ht0
  · have hpos : (0 : ℝ) < (u.val : ℝ) := by exact_mod_cast (show 0 < u.val by omega)
    have hu2 : (2 : ℝ) ≤ (u.val : ℝ) := by exact_mod_cast (show 2 ≤ u.val by omega)
    have : 1 / (u.val : ℝ) ≤ 1 / 2 := by
      rw [div_le_div_iff₀ hpos (by norm_num : (0 : ℝ) < 2)]; linarith
    linarith
  · norm_num at ht0
  · exact Fin.ext (by omega)
  · have hpos : (0 : ℝ) < (u.val : ℝ) := by exact_mod_cast (show 0 < u.val by omega)
    have : 0 < 1 / (u.val : ℝ) := by positivity
    linarith
  · have hpos : (0 : ℝ) < (t.val : ℝ) := by exact_mod_cast (show 0 < t.val by omega)
    have ht2 : (2 : ℝ) ≤ (t.val : ℝ) := by exact_mod_cast (show 2 ≤ t.val by omega)
    have : 1 / (t.val : ℝ) ≤ 1 / 2 := by
      rw [div_le_div_iff₀ hpos (by norm_num : (0 : ℝ) < 2)]; linarith
    linarith
  · have hpos : (0 : ℝ) < (t.val : ℝ) := by exact_mod_cast (show 0 < t.val by omega)
    have : 0 < 1 / (t.val : ℝ) := by positivity
    linarith
  · have hpt : (0 : ℝ) < (t.val : ℝ) := by exact_mod_cast (show 0 < t.val by omega)
    have hpu : (0 : ℝ) < (u.val : ℝ) := by exact_mod_cast (show 0 < u.val by omega)
    have htu : (t.val : ℝ) = (u.val : ℝ) := by
      field_simp at ht0
      linarith
    exact Fin.ext (Nat.cast_injective htu)

theorem doc_unit {k m : ℕ} (x : Fin k × Fin m) : ‖doc x‖ = 1 := by
  have hsq : ‖doc x‖ ^ 2 = 1 := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [doc, inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
      axis_inner, ↓reduceIte, axis_inner_slack, slack_inner_axis, slack_inner_slack]
    have hnn : 0 ≤ 1 - (slotCoord x.2) ^ 2 := by linarith [slotCoord_sq_le_one x.2]
    have hsqrt : Real.sqrt (1 - (slotCoord x.2) ^ 2) * Real.sqrt (1 - (slotCoord x.2) ^ 2) =
        1 - (slotCoord x.2) ^ 2 := Real.mul_self_sqrt hnn
    nlinarith
  nlinarith [norm_nonneg (doc x)]

theorem query_unit {k : ℕ} (q : Fin k × Bool) : ‖query q‖ = 1 := by
  have habs : |querySign q.2| = 1 := by cases q.2 <;> norm_num [querySign]
  simp [query, norm_smul, Real.norm_eq_abs, habs, axis_unit]

theorem inner_axis_doc {k m : ℕ} (i : Fin k) (x : Fin k × Fin m) :
    ⟪axis i, doc x⟫ = if i = x.1 then slotCoord x.2 else 0 := by
  simp only [doc, inner_add_right, real_inner_smul_right, axis_inner, axis_inner_slack]
  split_ifs <;> ring

theorem inner_query_doc {k m : ℕ} (q : Fin k × Bool) (x : Fin k × Fin m) :
    ⟪query q, doc x⟫ = if q.1 = x.1 then querySign q.2 * slotCoord x.2 else 0 := by
  simp only [query, real_inner_smul_left, inner_axis_doc]
  split_ifs <;> ring

theorem doc_injective {k m : ℕ} : Function.Injective (doc (k := k) (m := m)) := by
  rintro ⟨i, t⟩ ⟨j, u⟩ h
  have h1 := congrArg (fun v => ⟪axis i, v⟫) h
  simp only [inner_axis_doc, ↓reduceIte] at h1
  by_cases hij : i = j
  · subst j
    simp only [↓reduceIte] at h1
    exact Prod.ext rfl (slotCoord_injective h1)
  · simp only [if_neg hij] at h1
    exact False.elim (slotCoord_ne_zero t h1)

theorem database_size (k m : ℕ) : Fintype.card (Fin k × Fin m) = k * m := by
  simp [Fintype.card_prod]

theorem list_size {k m : ℕ} (i : Fin k) :
    (Finset.univ.filter (fun x : Fin k × Fin m => x.1 = i)).card = m := by
  rw [← Finset.univ_product_univ, Finset.filter_product_left (fun j : Fin k => j = i)]
  simp [Finset.filter_eq']

theorem doc_slot_zero {k m : ℕ} (hm : 2 ≤ m) (i : Fin k) :
    doc (i, (⟨0, by omega⟩ : Fin m)) = axis i := by
  simp [doc, slotCoord]

theorem doc_slot_one {k m : ℕ} (hm : 2 ≤ m) (i : Fin k) :
    doc (i, (⟨1, by omega⟩ : Fin m)) = -axis i := by
  simp [doc, slotCoord]

theorem query_target_inner {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    ⟪query q, doc (targetDoc hm q)⟫ = 1 := by
  rcases q with ⟨i, b⟩
  cases b <;> simp [inner_query_doc, targetDoc, querySign, slotCoord]

/-- Every query's unique true nearest neighbor is `targetDoc hm q`, with
constant cosine margin at least `1/2` over every other document in the database. -/
theorem query_margin {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool)
    (x : Fin k × Fin m) (hx : x ≠ targetDoc hm q) :
    ⟪query q, doc x⟫ + 1 / 2 ≤ ⟪query q, doc (targetDoc hm q)⟫ := by
  rw [query_target_inner hm q, inner_query_doc]
  split_ifs with h1
  · rcases q with ⟨i, b⟩
    rcases x with ⟨j, t⟩
    simp only at h1
    subst h1
    cases b
    · have ht1 : t.val ≠ 1 := by
        intro hval
        apply hx
        exact Prod.ext rfl (Fin.ext hval)
      have hneg : -slotCoord t ≤ 1 / 2 := by
        by_cases h0 : t.val = 0
        · simp [slotCoord, h0]
          norm_num
        · have hpos : (0 : ℝ) < (t.val : ℝ) := by
            exact_mod_cast (show 0 < t.val by omega)
          have : 0 < 1 / (t.val : ℝ) := by positivity
          unfold slotCoord
          rw [if_neg h0, if_neg ht1]
          linarith
      simp only [querySign, Bool.false_eq_true, ↓reduceIte]
      linarith
    · have ht0 : t.val ≠ 0 := by
        intro hval
        apply hx
        exact Prod.ext rfl (Fin.ext hval)
      have hle := slotCoord_le_half ht0
      simp only [querySign, ↓reduceIte]
      linarith
  · norm_num

/-! ### Part 1: Two-tip exact recovery -/

theorem tipScore_query {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) (j : Fin k) :
    tipScore hm (query q) j = if q.1 = j then 1 else 0 := by
  unfold tipScore
  rw [doc_slot_zero hm j, doc_slot_one hm j]
  rcases q with ⟨i, b⟩
  cases b <;> simp only [query, querySign, Bool.false_eq_true, ↓reduceIte,
    one_smul, neg_smul, inner_neg_left, inner_neg_right, neg_neg, axis_inner]
  · split_ifs <;> norm_num
  · split_ifs <;> norm_num

theorem twoTip_rank_eq_one {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    FlatScan.rank (negTipScore hm (query q)) q.1 = 1 := by
  apply FlatScan.rank_first
  intro j
  simp only [negTipScore, tipScore_query, ↓reduceIte]
  by_cases h : q.1 = j
  · subst h; simp
  · simp [h]

theorem twoTip_minProbedCount_eq_one {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    minProbedCount (negTipScore hm (query q)) q.1 = 1 := by
  unfold minProbedCount
  have hfilt : Finset.univ.filter (fun j : Fin k =>
      negTipScore hm (query q) j ≤ negTipScore hm (query q) q.1) = {q.1} := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
      negTipScore, tipScore_query, ↓reduceIte]
    by_cases h : q.1 = j
    · subst h; simp
    · simp [h, Ne.symm h]
  rw [hfilt, Finset.card_singleton]

theorem twoTip_probed_card_eq_one {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    (FlatScan.probed (negTipScore hm (query q)) 1).card = 1 := by
  have hfilt : FlatScan.probed (negTipScore hm (query q)) 1 = {q.1} := by
    ext j
    simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_singleton]
    constructor
    · intro hr
      by_contra hne
      have hlt : negTipScore hm (query q) q.1 < negTipScore hm (query q) j := by
        simp [negTipScore, tipScore_query, Ne.symm hne]
      have hmem : q.1 ∈ Finset.univ.filter (fun l =>
          negTipScore hm (query q) l < negTipScore hm (query q) j) :=
        Finset.mem_filter.mpr ⟨Finset.mem_univ _, hlt⟩
      have hpos : 1 ≤ (Finset.univ.filter (fun l =>
          negTipScore hm (query q) l < negTipScore hm (query q) j)).card :=
        Finset.card_pos.mpr ⟨q.1, hmem⟩
      unfold FlatScan.rank at hr
      omega
    · rintro rfl
      rw [twoTip_rank_eq_one hm q]
  rw [hfilt, Finset.card_singleton]

/-- The two-tip router achieves 100% Recall@1 at probe budget `1` on every query. -/
theorem twoTip_returns_all {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
      (negTipScore hm (query q)) 1 (targetDoc hm q) := by
  refine ⟨?_, ?_⟩
  · simp only [FlatScan.candidates, FlatScan.probed, Finset.mem_product, Finset.mem_filter,
      Finset.mem_univ, true_and, and_true, targetDoc]
    rw [twoTip_rank_eq_one hm q]
  · intro y _
    by_cases hy : y = targetDoc hm q
    · rw [hy]
    · linarith [query_margin hm q y hy]

/-- On every query, the two-tip router's total work (all `2 * k` tip inner
products plus the `m` scanned documents in the single probed list) is `2 * k + m`. -/
theorem twoTip_totalWork {k m : ℕ} (hm : 2 ≤ m) (q : Fin k × Bool) :
    twoTipWork (Fin m) (negTipScore hm (query q)) 1 = 2 * k + m := by
  simp [twoTipWork, FlatScan.candidates, twoTip_probed_card_eq_one hm q]

/-- Sum of two-tip total work over all `2 * k` queries. -/
theorem twoTip_sum_work {k m : ℕ} (hm : 2 ≤ m) :
    ∑ q : Fin k × Bool, twoTipWork (Fin m) (negTipScore hm (query q)) 1 =
      2 * k * (2 * k + m) := by
  simp [twoTip_totalWork hm, mul_comm]

/-! ### Relationship between `tieRank`, `minProbedCount`, `FlatScan.probed`, and `FlatScan.work` -/

theorem tieRank_pos {k : ℕ} (score : Fin k → ℝ) (lt : Fin k → Fin k → Prop) (i : Fin k) :
    1 ≤ tieRank score lt i := by
  unfold tieRank; omega

theorem tieRank_le_minProbedCount {k : ℕ} (score : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
    [IsIrrefl (Fin k) lt] (i : Fin k) :
    tieRank score lt i ≤ minProbedCount score i := by
  unfold tieRank minProbedCount
  let S := Finset.univ.filter (fun j : Fin k => beats score lt j i)
  have hi : i ∉ S := by
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and, beats, lt_irrefl, false_or,
      true_and]
    exact irrefl_of lt i
  have hsub : insert i S ⊆ Finset.univ.filter (fun j : Fin k => score j ≤ score i) := by
    intro j hj
    rcases Finset.mem_insert.mp hj with rfl | hjS
    · exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, le_rfl⟩
    · have hb : beats score lt j i := (Finset.mem_filter.mp hjS).2
      refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
      rcases hb with hlt | ⟨heq, _⟩
      · exact le_of_lt hlt
      · exact le_of_eq heq
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_insert_of_notMem hi] at hcard
  change 1 + S.card ≤ _
  omega

theorem minProbedCount_le_probed_card {k : ℕ} {score : Fin k → ℝ} {i : Fin k} {p : ℕ}
    (hi : FlatScan.rank score i ≤ p) :
    minProbedCount score i ≤ (FlatScan.probed score p).card := by
  apply Finset.card_le_card
  intro j hj
  have hle : score j ≤ score i := (Finset.mem_filter.mp hj).2
  simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and]
  have hrank : FlatScan.rank score j ≤ FlatScan.rank score i := by
    apply Nat.add_le_add_left
    apply Finset.card_le_card
    intro l hl
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      lt_of_lt_of_le (Finset.mem_filter.mp hl).2 hle⟩
  exact hrank.trans hi

theorem work_ge_of_returns {k m : ℕ} (hm : 2 ≤ m) {score : Fin k → ℝ}
    {q : Fin k × Bool} {budget : ℕ}
    (hret : FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
      score budget (targetDoc hm q)) :
    k + m * minProbedCount score q.1 ≤ FlatScan.work (Fin m) score budget := by
  have hi : FlatScan.rank score q.1 ≤ budget := by
    simpa [FlatScan.candidates, FlatScan.probed, targetDoc] using hret.1
  have hc := minProbedCount_le_probed_card hi
  simp only [FlatScan.work, FlatScan.candidates, Finset.card_product, Finset.card_univ,
    Fintype.card_fin]
  have hmul : m * minProbedCount score q.1 ≤ (FlatScan.probed score budget).card * m := by
    rw [mul_comm]
    exact Nat.mul_le_mul_right m hc
  omega

/-! ### Part 2: Impossibility for any 1-vector affine / power-diagram / Euclidean router -/

/-- Exact equivalence on unit queries: every power-diagram / squared-Euclidean
router `‖q - c j‖² - w j` is an affine inner-product router. -/
theorem powerDist_eq_negAffine {k : ℕ} (c : Fin k → Space k) (w : Fin k → ℝ)
    {q : Space k} (hq : ‖q‖ = 1) (j : Fin k) :
    powerDistScore c w q j =
      negAffineScore (fun l => (2 : ℝ) • c l) (fun l => w l - ‖c l‖ ^ 2 - 1) q j := by
  unfold powerDistScore negAffineScore
  rw [norm_sub_sq_real, hq, real_inner_smul_right]
  ring

/-- The strict total order on lists induced by `(b, lt)`: `i ≺ j` means `j` has
strictly larger bias than `i`, or equal bias and wins ties over `i`. -/
def biasOrder {k : ℕ} (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
    (i j : Fin k) : Prop :=
  b i < b j ∨ (b i = b j ∧ lt j i)

theorem biasOrder_trans {k : ℕ} (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
    [IsTrans (Fin k) lt] {i j l : Fin k}
    (hij : biasOrder b lt i j) (hjl : biasOrder b lt j l) :
    biasOrder b lt i l := by
  rcases hij with h1 | ⟨heq1, hlt1⟩
  · rcases hjl with h2 | ⟨heq2, _⟩
    · exact Or.inl (lt_trans h1 h2)
    · exact Or.inl (heq2 ▸ h1)
  · rcases hjl with h2 | ⟨heq2, hlt2⟩
    · exact Or.inl (heq1.symm ▸ h2)
    · exact Or.inr ⟨heq1.trans heq2, trans_of lt hlt2 hlt1⟩

theorem biasOrder_total {k : ℕ} (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
    [IsTrichotomous (Fin k) lt] {i j : Fin k} (hne : i ≠ j) :
    biasOrder b lt i j ∨ biasOrder b lt j i := by
  rcases lt_trichotomy (b i) (b j) with hlt | heq | hgt
  · exact Or.inl (Or.inl hlt)
  · rcases trichotomous_of lt i j with hij | rfl | hji
    · exact Or.inr (Or.inr ⟨heq.symm, hij⟩)
    · exact False.elim (hne rfl)
    · exact Or.inl (Or.inr ⟨heq, hji⟩)
  · exact Or.inr (Or.inl hgt)

theorem exists_biasOrder_min {k : ℕ} (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
    [IsTrans (Fin k) lt] [IsTrichotomous (Fin k) lt]
    (G : Finset (Fin k)) (hG : G.Nonempty) :
    ∃ i₀ ∈ G, ∀ j ∈ G, j ≠ i₀ → biasOrder b lt i₀ j := by
  induction G using Finset.induction_on with
  | empty => exact False.elim (Finset.not_nonempty_empty hG)
  | insert a s ha ih =>
    by_cases hs : s.Nonempty
    · rcases ih hs with ⟨i₀, hi₀, hmin⟩
      have hne : a ≠ i₀ := fun h => ha (h ▸ hi₀)
      rcases biasOrder_total b lt hne with hai | hia
      · refine ⟨a, Finset.mem_insert_self a s, fun j hj hja => ?_⟩
        rcases Finset.mem_insert.mp hj with rfl | hjs
        · exact False.elim (hja rfl)
        · by_cases hji : j = i₀
          · exact hji ▸ hai
          · exact biasOrder_trans b lt hai (hmin j hjs hji)
      · refine ⟨i₀, Finset.mem_insert_of_mem hi₀, fun j hj hji => ?_⟩
        rcases Finset.mem_insert.mp hj with rfl | hjs
        · exact hia
        · exact hmin j hjs hji
    · rw [Finset.not_nonempty_iff_eq_empty.mp hs]
      exact ⟨a, Finset.mem_singleton_self a, fun j hj hja =>
        False.elim (hja (Finset.mem_singleton.mp hj))⟩

/-- Core antipodal pairing lemma: whenever `i ≺ j` in `(b, lt)`, list `j` beats
list `i` on at least one of the two antipodal queries `±e_i`. -/
private theorem beats_of_biasOrder {k : ℕ} (r : Fin k → Space k) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) {i j : Fin k} (hij : biasOrder b lt i j) :
    beats (negAffineScore r b (query (i, false))) lt j i ∨
    beats (negAffineScore r b (query (i, true))) lt j i := by
  have hsum :
      (negAffineScore r b (query (i, false)) j - negAffineScore r b (query (i, false)) i) +
      (negAffineScore r b (query (i, true)) j - negAffineScore r b (query (i, true)) i) =
      2 * (b i - b j) := by
    simp only [negAffineScore, query, querySign, Bool.false_eq_true, ↓reduceIte,
      neg_smul, one_smul, inner_neg_left]
    ring
  rcases hij with hlt | ⟨heq, hlt_tie⟩
  · by_contra h
    push_neg at h
    have h1 : ¬ negAffineScore r b (query (i, false)) j <
        negAffineScore r b (query (i, false)) i := fun hc => h.1 (Or.inl hc)
    have h2 : ¬ negAffineScore r b (query (i, true)) j <
        negAffineScore r b (query (i, true)) i := fun hc => h.2 (Or.inl hc)
    linarith
  · rw [heq, sub_self, mul_zero] at hsum
    by_cases h1 : negAffineScore r b (query (i, false)) j ≤
        negAffineScore r b (query (i, false)) i
    · rcases lt_or_eq_of_le h1 with hlt1 | heq1
      · exact Or.inl (Or.inl hlt1)
      · exact Or.inl (Or.inr ⟨heq1, hlt_tie⟩)
    · have h2 : negAffineScore r b (query (i, true)) j <
          negAffineScore r b (query (i, true)) i := by linarith
      exact Or.inr (Or.inl h2)

/-- **Fixed-budget Recall@1 ceiling for ANY 1-vector affine router** (with arbitrary
vectors `r`, arbitrary scalar biases `b`, and arbitrary query-independent
tie-breaking `lt`): at probe budget `p ≥ 1`, at most `k + 2 * p - 1` of the
`2 * k` queries can have rank ≤ `p` (Recall@1 ≤ `1/2 + (2p - 1)/(2k)`). -/
theorem affine_answered_le {k : ℕ} (r : Fin k → Space k) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt]
    {p : ℕ} (hp : 1 ≤ p) :
    (Finset.univ.filter (fun q : Fin k × Bool =>
      tieRank (negAffineScore r b (query q)) lt q.1 ≤ p)).card ≤ k + 2 * p - 1 := by
  let ok (i : Fin k) (s : Bool) : Prop :=
    tieRank (negAffineScore r b (query (i, s))) lt i ≤ p
  let G : Finset (Fin k) := Finset.univ.filter (fun i => ok i false ∧ ok i true)
  -- Step 1: Total answered queries ≤ k + G.card.
  have hsplit :
      (Finset.univ.filter (fun q : Fin k × Bool =>
        tieRank (negAffineScore r b (query q)) lt q.1 ≤ p)).card ≤ k + G.card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    have hstep :
        (∑ i : Fin k, ∑ s : Bool, if ok i s then (1 : ℕ) else 0) ≤
        ∑ i : Fin k, (1 + if i ∈ G then 1 else 0) := by
      refine Finset.sum_le_sum fun i _ => ?_
      rw [Fintype.sum_bool]
      have hmem : (i ∈ G) ↔ (ok i false ∧ ok i true) := by
        simp [G]
      by_cases hf : ok i false <;> by_cases ht : ok i true
      · have hG : i ∈ G := hmem.mpr ⟨hf, ht⟩
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => ht (hmem.mp h).2
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => hf (hmem.mp h).1
        simp only [hf, ht, hG, ↓reduceIte]
        omega
      · have hG : i ∉ G := fun h => hf (hmem.mp h).1
        simp only [hf, ht, hG, ↓reduceIte]
        omega
    have hsum_eq : (∑ i : Fin k, (1 + if i ∈ G then (1 : ℕ) else 0)) = k + G.card := by
      rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, mul_one, Finset.card_univ,
        Fintype.card_fin, Finset.sum_boole, Nat.cast_id]
      simp [Finset.filter_mem_eq_inter]
    exact hstep.trans_eq hsum_eq
  -- Step 2: G.card ≤ 2 * p - 1.
  have hG : G.card ≤ 2 * p - 1 := by
    by_cases hne : G.Nonempty
    · rcases exists_biasOrder_min b lt G hne with ⟨i₀, hi₀, hmin⟩
      have hi₀_ok : ok i₀ false ∧ ok i₀ true := (Finset.mem_filter.mp hi₀).2
      let Bf := Finset.univ.filter (fun j =>
        beats (negAffineScore r b (query (i₀, false))) lt j i₀)
      let Bt := Finset.univ.filter (fun j =>
        beats (negAffineScore r b (query (i₀, true))) lt j i₀)
      have hsub : G.erase i₀ ⊆ Bf ∪ Bt := by
        intro j hj
        have hjne : j ≠ i₀ := (Finset.mem_erase.mp hj).1
        have hjG : j ∈ G := (Finset.mem_erase.mp hj).2
        rcases beats_of_biasOrder r b lt (hmin j hjG hjne) with hbf | hbt
        · exact Finset.mem_union_left Bt (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hbf⟩)
        · exact Finset.mem_union_right Bf (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hbt⟩)
      have hcard : (G.erase i₀).card ≤ Bf.card + Bt.card :=
        (Finset.card_le_card hsub).trans (Finset.card_union_le Bf Bt)
      rw [Finset.card_erase_of_mem hi₀] at hcard
      have hGpos : 1 ≤ G.card := Finset.one_le_card.mpr hne
      have hBf : 1 + Bf.card ≤ p := hi₀_ok.1
      have hBt : 1 + Bt.card ≤ p := hi₀_ok.2
      omega
    · rw [Finset.not_nonempty_iff_eq_empty.mp hne, Finset.card_empty]
      omega
  omega

/-- **Worst-case probe lower bound for full Recall@1**: any 1-vector affine /
power-diagram router (with arbitrary tie-breaking) achieving 100% Recall@1 at a
fixed probe budget `p` must satisfy `k + 1 ≤ 2 * p` (i.e. `p ≥ ⌈(k + 1) / 2⌉`). -/
theorem affine_fullRecall_budget_ge {k : ℕ} (hk : 1 ≤ k) (r : Fin k → Space k) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] {p : ℕ}
    (hall : ∀ q : Fin k × Bool, tieRank (negAffineScore r b (query q)) lt q.1 ≤ p) :
    k + 1 ≤ 2 * p := by
  let i0 : Fin k := ⟨0, by omega⟩
  have hp : 1 ≤ p := by
    have h0 : tieRank (negAffineScore r b (query (i0, true))) lt i0 ≤ p := hall (i0, true)
    have hpos := tieRank_pos (negAffineScore r b (query (i0, true))) lt i0
    omega
  have hle := affine_answered_le r b lt hp
  have hfull : (Finset.univ.filter (fun q : Fin k × Bool =>
      tieRank (negAffineScore r b (query q)) lt q.1 ≤ p)).card = 2 * k := by
    rw [Finset.filter_true_of_mem (fun q _ => hall q)]
    simp [Fintype.card_prod, mul_comm]
  omega

/-- **Average-rank lower bound for ANY 1-vector affine router** (with arbitrary
vectors `r`, arbitrary scalar biases `b`, and arbitrary tie-breaking `lt`):
`k * (k + 3) ≤ 2 * ∑ q, tieRank q`, so the average rank across the `2 * k`
queries is at least `(k + 3) / 4`. -/
theorem affine_sum_tieRank_ge {k : ℕ} (r : Fin k → Space k) (b : Fin k → ℝ)
    (lt : Fin k → Fin k → Prop) [IsStrictTotalOrder (Fin k) lt] :
    k * (k + 3) ≤ 2 * ∑ q : Fin k × Bool, tieRank (negAffineScore r b (query q)) lt q.1 := by
  let ind (i j : Fin k) (s : Bool) : ℕ :=
    if beats (negAffineScore r b (query (i, s))) lt j i then 1 else 0
  let pairCount (i j : Fin k) : ℕ := ind i j false + ind i j true
  have hrewrite :
      (∑ q : Fin k × Bool, tieRank (negAffineScore r b (query q)) lt q.1) =
      2 * k + ∑ i : Fin k, ∑ j : Fin k, pairCount i j := by
    rw [Fintype.sum_prod_type]
    have h1 :
        (∑ i : Fin k, ∑ s : Bool, tieRank (negAffineScore r b (query (i, s))) lt i) =
        ∑ i : Fin k, (2 + ∑ j : Fin k, pairCount i j) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      simp only [Fintype.sum_bool, tieRank, Finset.card_filter, pairCount, ind]
      rw [Finset.sum_add_distrib]
      omega
    rw [h1, Finset.sum_add_distrib]
    simp [mul_comm]
  rw [hrewrite]
  have hsym :
      2 * (∑ i : Fin k, ∑ j : Fin k, pairCount i j) =
      ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
    rw [two_mul]
    conv_lhs => rhs; rw [Finset.sum_comm]
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
  have hoff :
      k * (k - 1) ≤ ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
    have hpt :
        (∑ i : Fin k, ∑ j : Fin k, if i = j then (0 : ℕ) else 1) ≤
        ∑ i : Fin k, ∑ j : Fin k, (pairCount i j + pairCount j i) := by
      refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
      by_cases hij : i = j
      · simp [hij]
      · rw [if_neg hij]
        rcases biasOrder_total b lt hij with h | h
        · rcases beats_of_biasOrder r b lt h with hb | hb <;>
            · simp only [pairCount, ind, hb, ↓reduceIte]; omega
        · rcases beats_of_biasOrder r b lt h with hb | hb <;>
            · simp only [pairCount, ind, hb, ↓reduceIte]; omega
    have hcount :
        (∑ i : Fin k, ∑ j : Fin k, if i = j then (0 : ℕ) else 1) = k * (k - 1) := by
      have hinner (i : Fin k) :
          (∑ j : Fin k, if i = j then (0 : ℕ) else 1) = k - 1 := by
        have hfilt : Finset.univ.filter (fun j : Fin k => ¬ i = j) = Finset.univ.erase i := by
          ext j
          simp [ne_comm]
        rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, smul_eq_mul,
          mul_one, hfilt, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
          Fintype.card_fin]
      simp [hinner]
    omega
  have hmul_eq : 4 * k + k * (k - 1) = k * (k + 3) := by
    cases k with
    | zero => simp
    | succ n =>
      simp only [Nat.add_sub_cancel]
      ring
  omega

/-- Average `minProbedCount` lower bound for any 1-vector affine router (corollary
of `affine_sum_tieRank_ge` via `Fin.lt`). -/
theorem affine_sum_minProbedCount_ge {k : ℕ} (r : Fin k → Space k) (b : Fin k → ℝ) :
    k * (k + 3) ≤ 2 * ∑ q : Fin k × Bool,
      minProbedCount (negAffineScore r b (query q)) q.1 := by
  have h1 := affine_sum_tieRank_ge r b (· < ·)
  have h2 :
      (∑ q : Fin k × Bool, tieRank (negAffineScore r b (query q)) (· < ·) q.1) ≤
      ∑ q : Fin k × Bool, minProbedCount (negAffineScore r b (query q)) q.1 :=
    Finset.sum_le_sum fun q _ => tieRank_le_minProbedCount _ _ q.1
  omega

/-- **End-to-end `FlatScan.work` lower bound for ANY 1-vector affine router**
(even with an oracle per-query stopping budget `budget q`):
achieving full Recall@1 across the `2 * k` queries requires total work satisfying
`4 * k² + m * k * (k + 3) ≤ 2 * ∑ q, work q`. -/
theorem affine_sum_work_ge {k m : ℕ} (hm : 2 ≤ m)
    (r : Fin k → Space k) (b : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
        (negAffineScore r b (query q)) (budget q) (targetDoc hm q)) :
    4 * k ^ 2 + m * (k * (k + 3)) ≤
      2 * ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (negAffineScore r b (query q)) (budget q) := by
  have hsum :
      (∑ q : Fin k × Bool,
        (k + m * minProbedCount (negAffineScore r b (query q)) q.1)) ≤
      ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (negAffineScore r b (query q)) (budget q) :=
    Finset.sum_le_sum fun q _ => work_ge_of_returns hm (hret q)
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have hk_sum : (∑ _q : Fin k × Bool, k) = 2 * k ^ 2 := by
    simp [Fintype.card_prod]
    ring
  rw [hk_sum] at hsum
  have hmp := affine_sum_minProbedCount_ge r b
  have hmul : m * (k * (k + 3)) ≤
      m * (2 * ∑ q : Fin k × Bool, minProbedCount (negAffineScore r b (query q)) q.1) :=
    Nat.mul_le_mul_left m hmp
  linarith

/-- Corollary for any power-diagram / squared-Euclidean router `‖q - c j‖² - w j`:
identical `FlatScan.work` lower bound. -/
theorem powerDist_sum_work_ge {k m : ℕ} (hm : 2 ≤ m)
    (c : Fin k → Space k) (w : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
        (powerDistScore c w (query q)) (budget q) (targetDoc hm q)) :
    4 * k ^ 2 + m * (k * (k + 3)) ≤
      2 * ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (powerDistScore c w (query q)) (budget q) := by
  let r' : Fin k → Space k := fun l => (2 : ℝ) • c l
  let b' : Fin k → ℝ := fun l => w l - ‖c l‖ ^ 2 - 1
  have heq (q : Fin k × Bool) :
      powerDistScore c w (query q) = negAffineScore r' b' (query q) := by
    funext j
    exact powerDist_eq_negAffine c w (query_unit q) j
  have hret' : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
        (negAffineScore r' b' (query q)) (budget q) (targetDoc hm q) := by
    intro q; rw [← heq q]; exact hret q
  have h := affine_sum_work_ge hm r' b' budget hret'
  simpa [heq] using h

/-- **Balanced `N = k²` (`m = k`) separation between ANY 1-vector affine /
power-diagram router and the 2-tip router**:
- 2-tip total work summed over the `2 * k` queries is `6 * k²` (`3 * k = 3√N` per query).
- Any 1-vector affine router achieving 100% Recall@1 has `k² * (k + 7) ≤ 2 * ∑ work`
  (average at least `(k² + 7 * k) / 4 = (N + 7√N) / 4` per query), an `Ω(k) = Ω(√N)` blowup. -/
theorem balanced_affine_vs_twoTip_separation {k : ℕ} (hk : 2 ≤ k)
    (r : Fin k → Space k) (b : Fin k → ℝ) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin k => -⟪query q, doc x⟫)
        (negAffineScore r b (query q)) (budget q) (targetDoc hk q)) :
    (∑ q : Fin k × Bool, twoTipWork (Fin k) (negTipScore hk (query q)) 1 = 6 * k ^ 2) ∧
    (k ^ 2 * (k + 7) ≤
      2 * ∑ q : Fin k × Bool,
        FlatScan.work (Fin k) (negAffineScore r b (query q)) (budget q)) := by
  refine ⟨?_, ?_⟩
  · rw [twoTip_sum_work hk]
    ring
  · have h := affine_sum_work_ge hk r b budget hret
    have heq : 4 * k ^ 2 + k * (k * (k + 3)) = k ^ 2 * (k + 7) := by ring
    omega

/-! ### Part 3: Sharper impossibility for pure 1-vector linear routers (`b = 0`) -/

/-- For any pure 1-vector linear router `⟪q, r j⟫` (no scalar bias) and every
list `i`, every list `j` scores at least as well as `i` on at least one of `±e_i`,
so the two antipodal queries of list `i` together probe at least `k + 1` lists. -/
theorem linear_pair_minProbedCount_ge {k : ℕ} (r : Fin k → Space k) (i : Fin k) :
    k + 1 ≤ minProbedCount (negAffineScore r 0 (query (i, false))) i +
      minProbedCount (negAffineScore r 0 (query (i, true))) i := by
  unfold minProbedCount
  rw [Finset.card_filter, Finset.card_filter, ← Finset.sum_add_distrib]
  have hpt :
      (∑ j : Fin k, if j = i then (2 : ℕ) else 1) ≤
      ∑ j : Fin k,
        ((if negAffineScore r 0 (query (i, false)) j ≤
            negAffineScore r 0 (query (i, false)) i then 1 else 0) +
         (if negAffineScore r 0 (query (i, true)) j ≤
            negAffineScore r 0 (query (i, true)) i then 1 else 0)) := by
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases hij : j = i
    · subst hij
      simp
    · rw [if_neg hij]
      have hsum :
          (negAffineScore r 0 (query (i, false)) j - negAffineScore r 0 (query (i, false)) i) +
          (negAffineScore r 0 (query (i, true)) j - negAffineScore r 0 (query (i, true)) i) = 0 := by
        simp only [negAffineScore, Pi.zero_apply, add_zero, query, querySign,
          Bool.false_eq_true, ↓reduceIte, neg_smul, one_smul, inner_neg_left]
        ring
      by_cases h1 : negAffineScore r 0 (query (i, false)) j ≤
          negAffineScore r 0 (query (i, false)) i
      · simp only [h1, ↓reduceIte]
        omega
      · have h2 : negAffineScore r 0 (query (i, true)) j ≤
            negAffineScore r 0 (query (i, true)) i := by linarith
        simp only [h1, h2, ↓reduceIte]
        omega
  have hsum_lhs : (∑ j : Fin k, if j = i then (2 : ℕ) else 1) = k + 1 := by
    have : (∑ j : Fin k, if j = i then (2 : ℕ) else 1) =
        (∑ _j : Fin k, (1 : ℕ)) + ∑ j : Fin k, if j = i then (1 : ℕ) else 0 := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun j _ => by split_ifs <;> omega
    rw [this]
    simp
  omega

/-- **50% Recall@1 ceiling for pure 1-vector linear routers**: whenever `2 * p ≤ k`,
at most `k` of the `2 * k` queries can have `minProbedCount ≤ p` (Recall@1 ≤ 1/2). -/
theorem linear_recall_le_half {k : ℕ} (r : Fin k → Space k) {p : ℕ} (hp : 2 * p ≤ k) :
    (Finset.univ.filter (fun q : Fin k × Bool =>
      minProbedCount (negAffineScore r 0 (query q)) q.1 ≤ p)).card ≤ k := by
  rw [Finset.card_filter, Fintype.sum_prod_type]
  have hpt :
      (∑ i : Fin k, ∑ s : Bool,
        if minProbedCount (negAffineScore r 0 (query (i, s))) i ≤ p then (1 : ℕ) else 0) ≤
      ∑ _i : Fin k, (1 : ℕ) := by
    refine Finset.sum_le_sum fun i _ => ?_
    have hpair := linear_pair_minProbedCount_ge r i
    simp only [Fintype.sum_bool]
    by_cases hf : minProbedCount (negAffineScore r 0 (query (i, false))) i ≤ p <;>
      by_cases ht : minProbedCount (negAffineScore r 0 (query (i, true))) i ≤ p
    · omega
    · simp [hf, ht]
    · simp [hf, ht]
    · simp [hf, ht]
  simpa using hpt

/-- **Average-probe lower bound for pure 1-vector linear routers**:
`k * (k + 1) ≤ ∑ q, minProbedCount q` (average at least `(k + 1) / 2` lists probed). -/
theorem linear_sum_minProbedCount_ge {k : ℕ} (r : Fin k → Space k) :
    k * (k + 1) ≤ ∑ q : Fin k × Bool,
      minProbedCount (negAffineScore r 0 (query q)) q.1 := by
  rw [Fintype.sum_prod_type]
  have h := Finset.sum_le_sum (s := Finset.univ) fun (i : Fin k) _ =>
    linear_pair_minProbedCount_ge r i
  simpa [Fintype.sum_bool, add_comm] using h

/-- **End-to-end `FlatScan.work` lower bound for pure 1-vector linear routers**:
`2 * k² + m * k * (k + 1) ≤ ∑ q, work q` (at `m = k`, average total work is at
least `(k² + 3 * k) / 2 = (N + 3√N) / 2` per query). -/
theorem linear_sum_work_ge {k m : ℕ} (hm : 2 ≤ m)
    (r : Fin k → Space k) (budget : Fin k × Bool → ℕ)
    (hret : ∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
        (negAffineScore r 0 (query q)) (budget q) (targetDoc hm q)) :
    2 * k ^ 2 + m * (k * (k + 1)) ≤
      ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (negAffineScore r 0 (query q)) (budget q) := by
  have hsum :
      (∑ q : Fin k × Bool,
        (k + m * minProbedCount (negAffineScore r 0 (query q)) q.1)) ≤
      ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (negAffineScore r 0 (query q)) (budget q) :=
    Finset.sum_le_sum fun q _ => work_ge_of_returns hm (hret q)
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have hk_sum : (∑ _q : Fin k × Bool, k) = 2 * k ^ 2 := by
    simp [Fintype.card_prod]
    ring
  rw [hk_sum] at hsum
  have hmp := linear_sum_minProbedCount_ge r
  have hmul : m * (k * (k + 1)) ≤
      m * ∑ q : Fin k × Bool, minProbedCount (negAffineScore r 0 (query q)) q.1 :=
    Nat.mul_le_mul_left m hmp
  omega

/-! ### Part 4: Tightness witnesses for the affine bounds

Fix any sign pattern `σ : Fin k → Fin k → Bool`. The witness router stores
`r_j = ∑_{i < j} (±k) e_i` (sign `+` exactly when `σ i j`) and bias `b_j = j`.
On `±e_i`, list `l < i` scores `l < i`, and list `l > i` scores `l ± k`, so it
beats list `i` on exactly one of the two antipodal queries (the one whose sign
matches `σ i l`). Every unordered pair contributes exactly one beat, so the
`(k + 3) / 4` average-rank bound and the `4 k² + m k (k + 3)` work bound are
attained exactly, for every `σ`. Choosing `σ` by parity balances the beats and
also attains the fixed-budget bound `k + 1 ≤ 2 p`. -/

/-- Coordinate `⟪e_i, r_j⟫` of the witness vector `r_j`. -/
def witnessCoef {k : ℕ} (σ : Fin k → Fin k → Bool) (j i : Fin k) : ℝ :=
  if i < j then (if σ i j then (k : ℝ) else -(k : ℝ)) else 0

/-- Witness router vector `r_j = ∑_i witnessCoef σ j i • e_i`. -/
def witnessVec {k : ℕ} (σ : Fin k → Fin k → Bool) (j : Fin k) : Space k :=
  ∑ i : Fin k, witnessCoef σ j i • axis i

/-- Witness router bias `b_j = j`. -/
def witnessBias {k : ℕ} (j : Fin k) : ℝ := (j.val : ℝ)

/-- Negated witness score at query `q` (smaller is better). -/
def witnessScore {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) : Fin k → ℝ :=
  negAffineScore (witnessVec σ) witnessBias (query q)

/-- Number of lists that beat list `q.1` on query `q` under the witness. -/
def witnessBeatCount {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) : ℕ :=
  (Finset.univ.filter (fun l : Fin k => q.1 < l ∧ σ q.1 l = q.2)).card

theorem inner_axis_witnessVec {k : ℕ} (σ : Fin k → Fin k → Bool) (i j : Fin k) :
    ⟪axis i, witnessVec σ j⟫ = witnessCoef σ j i := by
  simp only [witnessVec, inner_sum, real_inner_smul_right, axis_inner, mul_ite, mul_one,
    mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]

theorem witnessScore_apply {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) (l : Fin k) :
    witnessScore σ q l = -(querySign q.2 * witnessCoef σ l q.1 + (l.val : ℝ)) := by
  simp only [witnessScore, negAffineScore, query, real_inner_smul_left, inner_axis_witnessVec,
    witnessBias]

theorem witnessScore_self {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) :
    witnessScore σ q q.1 = -((q.1.val : ℝ)) := by
  simp [witnessScore_apply, witnessCoef]

/-- The witness score of list `l` on `q` is `-(ε k + l)` with `ε ∈ {-1, 0, 1}`. -/
private theorem witnessScore_form {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool)
    (l : Fin k) :
    ∃ ε : ℝ, (ε = -1 ∨ ε = 0 ∨ ε = 1) ∧
      witnessScore σ q l = -(ε * (k : ℝ) + (l.val : ℝ)) := by
  rw [witnessScore_apply]
  unfold witnessCoef
  rcases q with ⟨i, s⟩
  by_cases hil : i < l
  · cases s <;> cases hσ : σ i l <;>
      simp only [querySign, hil, ↓reduceIte, Bool.false_eq_true]
    · exact ⟨1, by norm_num, by ring⟩
    · exact ⟨-1, by norm_num, by ring⟩
    · exact ⟨-1, by norm_num, by ring⟩
    · exact ⟨1, by norm_num, by ring⟩
  · exact ⟨0, by norm_num, by simp [hil]⟩

/-- Witness scores are pairwise distinct on every query. -/
theorem witnessScore_injective {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) :
    Function.Injective (witnessScore σ q) := by
  intro l l' h
  obtain ⟨ε, hε, he⟩ := witnessScore_form σ q l
  obtain ⟨ε', hε', he'⟩ := witnessScore_form σ q l'
  rw [he, he'] at h
  have hl : (l.val : ℝ) + 1 ≤ k := by exact_mod_cast l.isLt
  have hl' : (l'.val : ℝ) + 1 ≤ k := by exact_mod_cast l'.isLt
  have h0 : (0 : ℝ) ≤ l.val := Nat.cast_nonneg _
  have h0' : (0 : ℝ) ≤ l'.val := Nat.cast_nonneg _
  have hv : (l.val : ℝ) = l'.val := by
    rcases hε with rfl | rfl | rfl <;> rcases hε' with rfl | rfl | rfl <;> linarith
  exact Fin.ext (Nat.cast_injective hv)

/-- On `q = ±e_i`, list `l` scores strictly better than list `i` exactly when
`i < l` and the sign of `⟪e_i, r_l⟫` matches the query sign. -/
theorem witnessScore_lt_iff {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) (l : Fin k) :
    witnessScore σ q l < witnessScore σ q q.1 ↔ q.1 < l ∧ σ q.1 l = q.2 := by
  rw [witnessScore_self, witnessScore_apply]
  rcases q with ⟨i, s⟩
  simp only
  have hl : (l.val : ℝ) + 1 ≤ k := by exact_mod_cast l.isLt
  have hi0 : (0 : ℝ) ≤ i.val := Nat.cast_nonneg _
  unfold witnessCoef
  by_cases hil : i < l
  · have hilr : (i.val : ℝ) < l.val := by exact_mod_cast hil
    cases s <;> cases hσ : σ i l <;>
      simp only [querySign, hil, ↓reduceIte, Bool.false_eq_true, true_and,
        reduceCtorEq, iff_false, iff_true, not_lt] <;> nlinarith
  · have hli : (l.val : ℝ) ≤ i.val := by exact_mod_cast (not_lt.mp hil)
    simp only [hil, ↓reduceIte, mul_zero, zero_add, false_and, iff_false, not_lt]
    linarith

/-- Exact `tieRank` of the witness, for every tie-breaking order. -/
theorem witness_tieRank {k : ℕ} (σ : Fin k → Fin k → Bool) (lt : Fin k → Fin k → Prop)
    [IsIrrefl (Fin k) lt] (q : Fin k × Bool) :
    tieRank (witnessScore σ q) lt q.1 = 1 + witnessBeatCount σ q := by
  unfold tieRank witnessBeatCount
  congr 2
  ext l
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, beats]
  rw [← witnessScore_lt_iff σ q l]
  constructor
  · rintro (h | ⟨heq, hlt⟩)
    · exact h
    · have hl : l = q.1 := witnessScore_injective σ q heq
      subst hl
      exact absurd hlt (irrefl_of lt _)
  · exact Or.inl

/-- Exact `FlatScan.rank` of the witness. -/
theorem witness_rank {k : ℕ} (σ : Fin k → Fin k → Bool) (q : Fin k × Bool) :
    FlatScan.rank (witnessScore σ q) q.1 = 1 + witnessBeatCount σ q := by
  unfold FlatScan.rank witnessBeatCount
  congr 2
  ext l
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  exact witnessScore_lt_iff σ q l

/-- Each unordered pair of lists contributes exactly one beat. -/
theorem witness_sum_beatCount {k : ℕ} (σ : Fin k → Fin k → Bool) :
    2 * ∑ q : Fin k × Bool, witnessBeatCount σ q = k * (k - 1) := by
  have hpair (i : Fin k) :
      (∑ s : Bool, witnessBeatCount σ (i, s)) =
        ∑ l : Fin k, if i < l then (1 : ℕ) else 0 := by
    simp only [witnessBeatCount, Finset.card_filter, Fintype.sum_bool]
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    by_cases hil : i < l <;> cases σ i l <;> simp [hil]
  rw [Fintype.sum_prod_type]
  simp only [hpair]
  have hsym :
      2 * (∑ i : Fin k, ∑ l : Fin k, if i < l then (1 : ℕ) else 0) =
        ∑ i : Fin k, ∑ l : Fin k,
          ((if i < l then (1 : ℕ) else 0) + if l < i then (1 : ℕ) else 0) := by
    rw [two_mul]
    conv_lhs => rhs; rw [Finset.sum_comm]
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
  rw [hsym]
  have hpt (i l : Fin k) :
      ((if i < l then (1 : ℕ) else 0) + if l < i then (1 : ℕ) else 0) =
        if i = l then 0 else 1 := by
    rcases lt_trichotomy i l with h | rfl | h
    · simp [h, h.ne, h.not_gt]
    · simp
    · simp [h, h.ne', h.not_gt]
  simp only [hpt]
  have hinner (i : Fin k) :
      (∑ l : Fin k, if i = l then (0 : ℕ) else 1) = k - 1 := by
    have hfilt : Finset.univ.filter (fun l : Fin k => ¬ i = l) = Finset.univ.erase i := by
      ext l
      simp [ne_comm]
    rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, smul_eq_mul,
      mul_one, hfilt, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
      Fintype.card_fin]
  simp [hinner]

theorem witness_sum_one_add {k : ℕ} (σ : Fin k → Fin k → Bool) :
    2 * ∑ q : Fin k × Bool, (1 + witnessBeatCount σ q) = k * (k + 3) := by
  rw [Finset.sum_add_distrib, mul_add, witness_sum_beatCount]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
    Fintype.card_bool, smul_eq_mul, mul_one]
  cases k with
  | zero => simp
  | succ n =>
    simp only [Nat.add_sub_cancel]
    ring

/-- **The `(k + 3) / 4` average-rank bound is tight**: for every sign pattern
`σ` and every tie-breaking order, the witness affine router has
`2 * ∑ q, tieRank = k * (k + 3)`, matching `affine_sum_tieRank_ge` exactly. -/
theorem witness_sum_tieRank_eq {k : ℕ} (σ : Fin k → Fin k → Bool) (lt : Fin k → Fin k → Prop)
    [IsIrrefl (Fin k) lt] :
    2 * ∑ q : Fin k × Bool, tieRank (witnessScore σ q) lt q.1 = k * (k + 3) := by
  simp only [witness_tieRank σ lt]
  exact witness_sum_one_add σ

/-- **The work bound is tight**: stopping each query at its own true rank, the
witness affine router returns every exact answer and has
`2 * ∑ q, work q = 4 * k² + m * (k * (k + 3))`, matching `affine_sum_work_ge`. -/
theorem witness_sum_work_eq {k m : ℕ} (hm : 2 ≤ m) (σ : Fin k → Fin k → Bool) :
    (∀ q : Fin k × Bool,
      FlatScan.Returns (fun x : Fin k × Fin m => -⟪query q, doc x⟫)
        (witnessScore σ q) (FlatScan.rank (witnessScore σ q) q.1) (targetDoc hm q)) ∧
    2 * ∑ q : Fin k × Bool,
        FlatScan.work (Fin m) (witnessScore σ q) (FlatScan.rank (witnessScore σ q) q.1) =
      4 * k ^ 2 + m * (k * (k + 3)) := by
  refine ⟨fun q => ?_, ?_⟩
  · refine (FlatScan.returns_iff (witnessScore σ q) _ (targetDoc hm q) (fun y hy => ?_)
      (n := FlatScan.rank (witnessScore σ q) q.1) rfl).mpr le_rfl
    have := query_margin hm q y hy
    linarith
  · have hw (q : Fin k × Bool) :
        FlatScan.work (Fin m) (witnessScore σ q) (FlatScan.rank (witnessScore σ q) q.1) =
          k + (1 + witnessBeatCount σ q) * m := by
      rw [FlatScan.work_at_rank (Fin m) (witnessScore_injective σ q) q.1, witness_rank]
      simp
    simp only [hw]
    rw [Finset.sum_add_distrib, ← Finset.sum_mul]
    have hs := witness_sum_one_add σ
    have hk : (∑ _q : Fin k × Bool, k) = 2 * k ^ 2 := by
      simp [Fintype.card_prod]
      ring
    rw [hk]
    nlinarith [hs]

/-- Parity sign pattern: `σ i j` holds exactly when `j - i` is odd. -/
def paritySign {k : ℕ} (i j : Fin k) : Bool := decide ((j.val - i.val) % 2 = 1)

theorem parity_beatCount_le {k : ℕ} (q : Fin k × Bool) :
    witnessBeatCount paritySign q ≤ k / 2 := by
  unfold witnessBeatCount
  have h := Finset.card_le_card_of_injOn (s := Finset.univ.filter (fun l : Fin k =>
      q.1 < l ∧ paritySign q.1 l = q.2)) (t := Finset.range (k / 2))
    (fun l : Fin k => (l.val - q.1.val - 1) / 2) ?_ ?_
  · simpa using h
  · intro l hl
    have hl' := (Finset.mem_filter.mp hl).2
    have hlt : q.1.val < l.val := hl'.1
    have := l.isLt
    simp only [Finset.coe_range, Set.mem_Iio]
    omega
  · intro l hl l' hl' heq
    have h1 := (Finset.mem_filter.mp hl).2
    have h2 := (Finset.mem_filter.mp hl').2
    have hlt : q.1.val < l.val := h1.1
    have hlt' : q.1.val < l'.val := h2.1
    have hpar : paritySign q.1 l = paritySign q.1 l' := h1.2.trans h2.2.symm
    simp only [paritySign, decide_eq_decide] at hpar
    simp only at heq
    exact Fin.ext (by omega)

/-- **The fixed-budget full-recall bound is tight**: for `k ≥ 1`, the smallest
fixed probe budget at which some one-vector affine router recovers all `2 * k`
answers is exactly `k / 2 + 1 = ⌈(k + 1) / 2⌉`. The parity witness attains it,
and `affine_fullRecall_budget_ge` rules out anything smaller. The same witness
also attains the average bound (`witness_sum_tieRank_eq`). -/
theorem affine_min_fullRecall_budget {k : ℕ} (hk : 1 ≤ k) :
    (∀ (lt : Fin k → Fin k → Prop) [IsIrrefl (Fin k) lt] (q : Fin k × Bool),
      tieRank (negAffineScore (witnessVec paritySign) witnessBias (query q)) lt q.1 ≤
        k / 2 + 1) ∧
    (∀ (r : Fin k → Space k) (b : Fin k → ℝ) (lt : Fin k → Fin k → Prop)
      [IsStrictTotalOrder (Fin k) lt] (p : ℕ),
      (∀ q : Fin k × Bool, tieRank (negAffineScore r b (query q)) lt q.1 ≤ p) →
        k / 2 + 1 ≤ p) := by
  refine ⟨fun lt _ q => ?_, fun r b lt _ p hall => ?_⟩
  · have h := witness_tieRank paritySign lt q
    unfold witnessScore at h
    rw [h]
    have := parity_beatCount_le q
    omega
  · have := affine_fullRecall_budget_ge hk r b lt hall
    omega

/-- Per-query tie-breaking defeats the floor: the zero router ties every list, and a
query-dependent strict total order that puts the owner first gives every owner rank 1. -/
theorem zero_router_perQuery_ties {k : ℕ} (q : Fin k × Bool) :
    ∃ (lt : Fin k → Fin k → Prop), IsStrictTotalOrder (Fin k) lt ∧
      tieRank (negAffineScore (fun _ => 0) (fun _ => 0) (query q)) lt q.1 = 1 := by
  let e : Fin k ≃ Fin k := Equiv.swap q.1 ⟨0, Fin.pos q.1⟩
  let lt : Fin k → Fin k → Prop := fun a b => (e a).val < (e b).val
  refine ⟨lt, ?_, ?_⟩
  · refine { trichotomous := ?_, irrefl := ?_, trans := ?_ }
    · intro a b
      rcases lt_trichotomy (e a).val (e b).val with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl (e.injective (Fin.ext h)))
      · exact Or.inr (Or.inr h)
    · intro a h; exact lt_irrefl _ h
    · intro a b c h1 h2; exact lt_trans h1 h2
  · unfold tieRank
    rw [Nat.add_eq_left, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro j _ h
    rcases h with h | ⟨_, h⟩
    · simp [negAffineScore] at h
    · have : (e q.1).val = 0 := by simp [e, Equiv.swap_apply_left]
      simp only [lt, this] at h
      exact Nat.not_lt_zero _ h

end IndexRecall.OneVectorImpossibility
