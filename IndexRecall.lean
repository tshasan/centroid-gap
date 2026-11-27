import IndexRecall.EMeansSeedLimit
import IndexRecall.EMeansTraining
import IndexRecall.GraphConic
import IndexRecall.IVFGap
import IndexRecall.ManyQueryIVF
import IndexRecall.OneProbe
import IndexRecall.OneVectorCircle
import IndexRecall.OneVectorImpossibility
import IndexRecall.PlanarIVF
import IndexRecall.PlanarPowerRouter
import IndexRecall.RobustPruneFailure
import IndexRecall.Round
import IndexRecall.TopTwo
import Mathlib.Order.Interval.Set.LinearOrder
import Mathlib.Tactic.NormNum

/-!
# Predicting recall after one Lloyd round on four real sites

The index probes the nearer of two centers, then searches its posting list
exactly. Distance ties go to the smaller coordinate. `round_prediction`
tests both per-query work and strict Recall@1 improvement on a query list.
The test uses only cut sizes and the number of queries in two intervals.
Repeated queries contribute with their multiplicities.
-/

noncomputable section

namespace IndexRecall

open scoped Classical

def Four.sites (D : Four) : List ℝ := [D.a, D.b, D.c, D.d]

/-- The posting list scanned for a query. Both centers and data use the
nearest-center comparison proved in `dist_le_iff`. -/
def Centers.candidates {D : Four} (I : Centers D) (q : ℝ) : List ℝ :=
  if q ≤ I.boundary then D.sites.filter (fun x => decide (x ≤ I.boundary))
  else D.sites.filter (fun x => decide (I.boundary < x))

theorem Centers.mem_candidates_iff {D : Four} (I : Centers D) (q x : ℝ) :
    x ∈ I.candidates q ↔ D.IsSite x ∧ (q ≤ I.boundary ↔ x ≤ I.boundary) := by
  have hsite : x ∈ D.sites ↔ D.IsSite x := by simp [Four.sites, Four.IsSite]
  by_cases hq : q ≤ I.boundary <;> simp [Centers.candidates, hq, hsite, not_le]

/-- Exact reranking within the scanned list, with ties going to the smaller site. -/
def Centers.Returns {D : Four} (I : Centers D) (q x : ℝ) : Prop :=
  x ∈ I.candidates q ∧ ∀ y ∈ I.candidates q,
    (q - x) ^ 2 ≤ (q - y) ^ 2 ∧ ((q - x) ^ 2 = (q - y) ^ 2 → x ≤ y)

theorem Centers.returns_nearest_iff {D : Four} (I : Centers D) (q : ℝ) :
    I.Returns q (D.nearest q) ↔ Hit D I.boundary q := by
  constructor
  · intro h
    exact ((I.mem_candidates_iff q _).1 h.1).2
  · intro h
    refine ⟨(I.mem_candidates_iff q _).2 ⟨D.nearest_isSite q, h⟩, ?_⟩
    intro y hy
    exact D.nearest_min q y ((I.mem_candidates_iff q y).1 hy).1

/-- Two centroid distances plus one distance for each scanned site. -/
def Centers.distanceWork {D : Four} (I : Centers D) (q : ℝ) : ℕ :=
  2 + (I.candidates q).length

/-- The fixed cost covers the two-center lookup and one list probe. Each
scanned site adds the same cost. -/
def Centers.modeledLatency {D : Four} (I : Centers D) (q fixed perSite : ℝ) : ℝ :=
  fixed + perSite * (I.candidates q).length

theorem latency_eq_iff {D : Four} (I J : Centers D) (q fixed perSite : ℝ)
    (hp : 0 < perSite) :
    I.modeledLatency q fixed perSite = J.modeledLatency q fixed perSite ↔
      I.distanceWork q = J.distanceWork q := by
  simp only [Centers.modeledLatency, Centers.distanceWork, add_right_inj, mul_right_inj' hp.ne',
    Nat.cast_inj]

def Cut.leftSize : Cut → ℕ
  | .one => 1
  | .two => 2
  | .three => 3

/-- Predict the scan length from the cut and the side selected by the query. -/
def scanSize (D : Four) (b q : ℝ) : ℕ :=
  let n := (D.cut b).leftSize
  if q ≤ b then n else 4 - n

theorem Centers.distanceWork_eq {D : Four} (I : Centers D) (q : ℝ) :
    I.distanceWork q = 2 + scanSize D I.boundary q := by
  have ha : D.a ≤ I.boundary := I.first
  have hd : I.boundary < D.d := I.last
  by_cases hb : I.boundary < D.b
  · have hc := hb.trans D.bc
    by_cases hq : q ≤ I.boundary <;>
      simp [distanceWork, candidates, Four.sites, scanSize, Four.cut, Cut.leftSize,
        hq, ha, hb, hc, hd, ha.not_gt, hb.not_ge, hc.not_ge, hd.not_ge]
  · have hble := le_of_not_gt hb
    by_cases hc : I.boundary < D.c
    · by_cases hq : q ≤ I.boundary <;>
        simp [distanceWork, candidates, Four.sites, scanSize, Four.cut, Cut.leftSize,
          hq, ha, hb, hble, hc, hd, ha.not_gt, hc.not_ge, hd.not_ge]
    · have hcle := le_of_not_gt hc
      by_cases hq : q ≤ I.boundary <;>
        simp [distanceWork, candidates, Four.sites, scanSize, Four.cut, Cut.leftSize,
          hq, ha, hb, hble, hc, hcle, hd, ha.not_gt, hd.not_ge]

theorem Centers.work_of_balanced {D : Four} (I : Centers D)
    (hc : D.cut I.boundary = .two) (q : ℝ) : I.distanceWork q = 4 := by
  rw [I.distanceWork_eq q]
  by_cases hq : q ≤ I.boundary <;> simp [scanSize, hc, Cut.leftSize, hq]

private theorem both_balanced_of_same_scan (D : Four) {b b' : ℝ} (hb : b < b')
    (hs : ∀ q, scanSize D b q = scanSize D b' q) :
    D.cut b = .two ∧ D.cut b' = .two := by
  have hleft := hs b
  have hswitch := hs b'
  simp only [scanSize, le_refl, hb.le, if_true] at hleft
  simp only [scanSize, le_refl, hb.not_ge, if_false, if_true] at hswitch
  cases hi : D.cut b <;> cases hj : D.cut b' <;> simp_all [Cut.leftSize]

/-- Equal work for every real query forces both cuts to be balanced whenever
the router moves. At the smaller boundary both probes go left, and at the
larger boundary they go to opposite sides. -/
theorem same_work_all_iff {D : Four} (I J : Centers D) :
    (∀ q, I.distanceWork q = J.distanceWork q) ↔
      I.boundary = J.boundary ∨ (D.cut I.boundary = .two ∧ D.cut J.boundary = .two) := by
  constructor
  · intro h
    have hs (q : ℝ) : scanSize D I.boundary q = scanSize D J.boundary q := by
      simpa only [Centers.distanceWork_eq, Nat.add_left_cancel_iff] using h q
    rcases lt_trichotomy I.boundary J.boundary with hb | hb | hb
    · exact Or.inr (both_balanced_of_same_scan D hb hs)
    · exact Or.inl hb
    · exact Or.inr (both_balanced_of_same_scan D hb (fun q => (hs q).symm)).symm
  · rintro (hb | ⟨hi, hj⟩) q
    · rw [I.distanceWork_eq, J.distanceWork_eq, hb]
    · rw [I.work_of_balanced hi, J.work_of_balanced hj]

def Centers.hitCount {D : Four} (I : Centers D) (qs : List ℝ) : ℕ :=
  qs.countP (fun q => decide (Hit D I.boundary q))

def Centers.recall {D : Four} (I : Centers D) (qs : List ℝ) : ℝ :=
  (I.hitCount qs : ℝ) / qs.length

def gapCount (b h : ℝ) (qs : List ℝ) : ℕ :=
  qs.countP (fun q => decide (InGap b h q))

/-- The two outcomes partition the workload, including queries exactly at
either endpoint of the miss interval. -/
theorem Centers.hits_add_gap {D : Four} (I : Centers D) (qs : List ℝ) :
    I.hitCount qs + gapCount I.boundary (D.edge (D.cut I.boundary)) qs = qs.length := by
  induction qs with
  | nil => rfl
  | cons q qs ih =>
    have hm := miss_iff D I.boundary q I.first I.last
    by_cases h : Hit D I.boundary q
    · have hg := mt hm.mpr (not_not_intro h)
      simp [hitCount, gapCount, h, hg] at ih ⊢
      omega
    · have hg := hm.mp h
      simp [hitCount, gapCount, h, hg] at ih ⊢
      omega

/-- Prefix counts allow the interval test to use a sorted workload. -/
def countLE (t : ℝ) (qs : List ℝ) : ℕ := qs.countP (fun q => decide (q ≤ t))

theorem gapCount_eq_sub (b h : ℝ) (qs : List ℝ) :
    gapCount b h qs = countLE (max b h) qs - countLE (min b h) qs := by
  suffices hh : countLE (min b h) qs + gapCount b h qs = countLE (max b h) qs by omega
  induction qs with
  | nil => rfl
  | cons q qs ih =>
    have ho := min_le_max (a := b) (b := h)
    simp only [countLE, gapCount, List.countP_cons, decide_eq_true_eq] at ih ⊢
    by_cases hlo : q ≤ min b h
    · have hhi := hlo.trans ho
      have hg : ¬ InGap b h q := fun hg => hg.1.not_ge hlo
      simp only [if_pos hlo, if_pos hhi, if_neg hg, Nat.add_zero]
      omega
    · have hlt := lt_of_not_ge hlo
      by_cases hhi : q ≤ max b h
      · have hg : InGap b h q := ⟨hlt, hhi⟩
        simp only [if_neg hlo, if_pos hhi, if_pos hg, Nat.add_zero]
        omega
      · have hg : ¬ InGap b h q := fun hg => hhi hg.2
        simpa only [if_neg hlo, if_neg hhi, if_neg hg, Nat.add_zero] using ih

theorem recall_improves_iff {D : Four} (I J : Centers D) (qs : List ℝ) (hqs : qs ≠ []) :
    I.recall qs < J.recall qs ↔
      gapCount J.boundary (D.edge (D.cut J.boundary)) qs <
      gapCount I.boundary (D.edge (D.cut I.boundary)) qs := by
  have hn : 0 < (qs.length : ℝ) := by exact_mod_cast List.length_pos_iff.mpr hqs
  rw [Centers.recall, Centers.recall, div_lt_div_iff_of_pos_right hn, Nat.cast_lt]
  have hi := I.hits_add_gap qs
  have hj := J.hits_add_gap qs
  omega

/-- A necessary and sufficient test for strict workload improvement at the
same modeled time on every query after an actual weighted Lloyd round. -/
theorem round_prediction {D : Four} (I : Centers D) (w : Weights) (qs : List ℝ)
    (hqs : qs ≠ []) (fixed perSite : ℝ) (hp : 0 < perSite) :
    let J := I.next w
    ((∀ q ∈ qs, I.modeledLatency q fixed perSite = J.modeledLatency q fixed perSite) ∧
      I.recall qs < J.recall qs) ↔
      (∀ q ∈ qs, scanSize D I.boundary q = scanSize D J.boundary q) ∧
      gapCount J.boundary (D.edge (D.cut J.boundary)) qs <
        gapCount I.boundary (D.edge (D.cut I.boundary)) qs := by
  dsimp only
  simp only [latency_eq_iff _ _ _ _ _ hp, Centers.distanceWork_eq, Nat.add_left_cancel_iff,
    recall_improves_iff I (I.next w) qs hqs]

private theorem gap_subset_iff (b h b' h' : ℝ) :
    (Set.Ioc (min b' h') (max b' h') ⊆ Set.Ioc (min b h) (max b h)) ↔
      b' = h' ∨ (min b h ≤ min b' h' ∧ max b' h' ≤ max b h) := by
  by_cases he : b' = h'
  · subst h'
    simp
  · rw [Set.Ioc_subset_Ioc_iff (min_lt_max.mpr he)]
    tauto

/-- A workload-independent test. Every previously successful query still
hits exactly when the new miss interval is empty or lies inside the old one. -/
theorem no_query_lost_iff {D : Four} (I J : Centers D) :
    let h := D.edge (D.cut I.boundary)
    let h' := D.edge (D.cut J.boundary)
    (∀ q, Hit D I.boundary q → Hit D J.boundary q) ↔
      J.boundary = h' ∨ (min I.boundary h ≤ min J.boundary h' ∧
        max J.boundary h' ≤ max I.boundary h) := by
  let gap (K : Centers D) := Set.Ioc
    (min K.boundary (D.edge (D.cut K.boundary))) (max K.boundary (D.edge (D.cut K.boundary)))
  have hg (K : Centers D) (q : ℝ) : q ∈ gap K ↔ ¬ Hit D K.boundary q :=
    (miss_iff D K.boundary q K.first K.last).symm
  calc
    _ ↔ gap J ⊆ gap I := by
      constructor
      · intro h q hq
        exact (hg I q).2 (fun hi => (hg J q).1 hq (h q hi))
      · intro h q hi
        by_contra hj
        exact (hg I q).1 (h ((hg J q).2 hj)) hi
    _ ↔ _ := gap_subset_iff _ _ _ _

/-- With unit weights and two old sites per list, the next routing
boundary is available directly from the four input coordinates. -/
theorem balanced_next_boundary {D : Four} (I : Centers D)
    (hc : D.cut I.boundary = .two) :
    (I.next Weights.unit).boundary = (D.a + D.b + D.c + D.d) / 4 := by
  simp only [Centers.next, hc]
  dsimp [Centers.ofInterval, Centers.boundary, Weights.unit, mean2, mid]
  ring

/-! ## Retrieving the two nearest neighbors -/

theorem Centers.candidates_of_balanced {D : Four} (I : Centers D)
    (hc : D.cut I.boundary = .two) (q : ℝ) :
    (I.candidates q).toFinset = D.balancedProbe I.boundary q := by
  have hb : D.b ≤ I.boundary := by
    by_contra! h
    simp [Four.cut, h] at hc
  have hc' : I.boundary < D.c := by
    by_contra h
    simp [Four.cut, hb.not_gt, h] at hc
  have ha : D.a ≤ I.boundary := I.first
  have hd : I.boundary < D.d := I.last
  by_cases hq : q ≤ I.boundary <;>
    simp [Centers.candidates, Four.sites, Four.balancedProbe, hq, ha, hb, hc', hd,
      ha.not_gt, hb.not_gt, hc'.not_ge, hd.not_ge]

/-- Fraction of the two global neighbors present in the scanned list.
With two scanned sites the index returns the entire list. -/
def Centers.recallAtTwo {D : Four} (I : Centers D) (q : ℝ) : ℝ :=
  ((D.trueTopTwo q ∩ (I.candidates q).toFinset).card : ℝ) / 2

/-- If both partitions are balanced and the old centers lie within their
respective pairs, one weighted Lloyd round preserves Recall@2 at every query. -/
theorem balanced_round_recallAtTwo {D : Four} (I : Centers D) (w : Weights)
    (hi : D.cut I.boundary = .two) (hj : D.cut (I.next w).boundary = .two)
    (hl : D.a ≤ I.left ∧ I.left ≤ D.b) (hr : D.c ≤ I.right ∧ I.right ≤ D.d) (q : ℝ) :
    (I.next w).recallAtTwo q = I.recallAtTwo q := by
  have hn := I.next_in_balanced_hulls w hi
  simp only [Centers.recallAtTwo, I.candidates_of_balanced hi, (I.next w).candidates_of_balanced hj]
  change D.balancedRecallAtTwo (I.next w).boundary q = D.balancedRecallAtTwo I.boundary q
  exact (D.balanced_recallAtTwo_centers _ _ q hn.1.1 hn.1.2 hn.2.1 hn.2.2).trans
    (D.balanced_recallAtTwo_centers _ _ q hl.1 hl.2 hr.1 hr.2).symm

/-! ## A single round with opposite recall changes at identical work -/

def exampleSites : Four := ⟨-1, 0, 2, 5, by norm_num, by norm_num, by norm_num⟩

/-- The two seeds are the first and third data sites. -/
def exampleSeeds : Centers exampleSites :=
  Centers.ofInterval exampleSites.a exampleSites.c le_rfl
    (exampleSites.ab.trans exampleSites.bc) exampleSites.cd.le

/-- Training loss falls from ten to five. The same update gains the true
neighbor at 3/4 and loses it at 5/4, scanning two sites on every query. -/
theorem same_round_opposite_recall :
    let I := exampleSeeds
    let J := I.next Weights.unit
    I.sse Weights.unit = 10 ∧ J.sse Weights.unit = 5 ∧
    (∀ q, I.distanceWork q = 4 ∧ J.distanceWork q = 4) ∧
    (I.recall [3/4] = 0 ∧ J.recall [3/4] = 1) ∧
    (I.recall [5/4] = 1 ∧ J.recall [5/4] = 0) := by
  let I := exampleSeeds
  let J := I.next Weights.unit
  change I.sse _ = 10 ∧ J.sse _ = 5 ∧
    (∀ q, I.distanceWork q = 4 ∧ J.distanceWork q = 4) ∧
    (I.recall [3/4] = 0 ∧ J.recall [3/4] = 1) ∧
    (I.recall [5/4] = 1 ∧ J.recall [5/4] = 0)
  have hI : I.boundary = 1/2 := by norm_num [I, exampleSeeds, Centers.ofInterval,
    Centers.boundary, exampleSites, mid]
  have hcut : exampleSites.cut I.boundary = .two := by rw [hI]; norm_num [Four.cut, exampleSites]
  have hJ : J.boundary = 3/2 := by
    rw [show J.boundary = _ from balanced_next_boundary I hcut]
    norm_num [exampleSites]
  have hcut' : exampleSites.cut J.boundary = .two := by rw [hJ]; norm_num [Four.cut, exampleSites]
  refine ⟨?_, ?_, fun q => ⟨I.work_of_balanced hcut q, J.work_of_balanced hcut' q⟩, ?_, ?_⟩
  · norm_num [I, Centers.sse, exampleSeeds, exampleSites, Centers.ofInterval, Weights.unit,
      min_def]
  · change (I.next Weights.unit).sse _ = 5
    simp only [Centers.next, hcut]
    norm_num [Centers.sse, Centers.ofInterval, exampleSites, Weights.unit, mean2, min_def]
  all_goals
    norm_num [Centers.recall, Centers.hitCount, Hit, hI, hJ, Four.nearest, exampleSites,
      mid, List.countP_cons]

end IndexRecall

