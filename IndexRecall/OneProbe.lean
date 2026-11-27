import IndexRecall.Round
import Mathlib.Data.List.MinMax
import Mathlib.Order.Interval.Set.UnorderedInterval
import Mathlib.Tactic.NormNum

/-!
# Recall@1 for two separated posting lists on the real line

Both ground truth and the one-probe index minimize squared distance by an exact
list scan. The data boundary is the midpoint of the adjacent list extrema.
At a query with a unique nearest point and unequal center distances, the scans
disagree exactly in the open interval between the data and routing boundaries.
-/

noncomputable section

namespace IndexRecall

open scoped Classical

/-- Lists may contain repeated coordinates and can be stored in any order. -/
structure PostingLists where
  left : List ℝ
  right : List ℝ
  left_ne : left ≠ []
  right_ne : right ≠ []
  separated : ∀ x ∈ left, ∀ y ∈ right, x < y

/-- The first center labels the left posting list. -/
structure Router where
  left : ℝ
  right : ℝ
  ordered : left < right

def Centers.toRouter {D : Four} (I : Centers D) : Router := ⟨I.left, I.right, I.ordered⟩

def Router.boundary (C : Router) : ℝ := mid C.left C.right

def Router.RoutesLeft (C : Router) (q : ℝ) : Prop :=
  (q - C.left) ^ 2 ≤ (q - C.right) ^ 2

/-- Assign each database point using the same distance comparison as the query router. -/
def Router.postings (C : Router) (xs : List ℝ) : List ℝ × List ℝ :=
  xs.partition (fun x => decide (C.RoutesLeft x))

def Router.Untied (C : Router) (q : ℝ) : Prop :=
  (q - C.left) ^ 2 ≠ (q - C.right) ^ 2

private theorem dist_eq_iff (x y q : ℝ) (hxy : x < y) :
    (q - x) ^ 2 = (q - y) ^ 2 ↔ q = mid x y := by
  rw [sq_eq_sq_iff_eq_or_eq_neg]
  unfold mid
  constructor
  · rintro (h | h) <;> linarith
  · intro h
    right
    linarith

theorem Router.untied_iff (C : Router) (q : ℝ) : C.Untied q ↔ q ≠ C.boundary :=
  not_congr (dist_eq_iff _ _ _ C.ordered)

theorem Router.routesLeft_iff (C : Router) (q : ℝ) (hq : C.Untied q) :
    C.RoutesLeft q ↔ q < C.boundary :=
  (dist_le_iff _ _ _ C.ordered).trans
    ⟨fun h => lt_of_le_of_ne h ((C.untied_iff q).mp hq), fun h => h.le⟩

/-- A full scan, returning the first minimizer if several points tie. -/
def exactScan (q : ℝ) (xs : List ℝ) (hne : xs ≠ []) : ℝ :=
  (xs.argmin (fun x => (q - x) ^ 2)).get (by simpa [Option.isSome_iff_ne_none] using hne)

theorem exactScan_spec (q : ℝ) (xs : List ℝ) (hne : xs ≠ []) :
    exactScan q xs hne ∈ xs ∧ ∀ x ∈ xs, (q - exactScan q xs hne) ^ 2 ≤ (q - x) ^ 2 := by
  have hm : exactScan q xs hne ∈ xs.argmin (fun x => (q - x) ^ 2) := Option.get_mem _
  exact ⟨List.argmin_mem hm, fun _ hx => List.le_of_mem_argmin hx hm⟩

namespace PostingLists

variable (D : PostingLists)

def points : List ℝ := D.left ++ D.right

def leftMax : ℝ :=
  (D.left.argmax id).get (by simpa [Option.isSome_iff_ne_none] using D.left_ne)

def rightMin : ℝ :=
  (D.right.argmin id).get (by simpa [Option.isSome_iff_ne_none] using D.right_ne)

theorem leftMax_spec : D.leftMax ∈ D.left ∧ ∀ x ∈ D.left, x ≤ D.leftMax := by
  have hm : D.leftMax ∈ D.left.argmax id := Option.get_mem _
  exact ⟨List.argmax_mem hm, fun _ hx => List.le_of_mem_argmax hx hm⟩

theorem rightMin_spec : D.rightMin ∈ D.right ∧ ∀ x ∈ D.right, D.rightMin ≤ x := by
  have hm : D.rightMin ∈ D.right.argmin id := Option.get_mem _
  exact ⟨List.argmin_mem hm, fun _ hx => List.le_of_mem_argmin hx hm⟩

def boundary : ℝ := mid D.leftMax D.rightMin

def nearest (q : ℝ) : ℝ := exactScan q D.points (by simp [points, D.left_ne])

theorem nearest_spec (q : ℝ) :
    D.nearest q ∈ D.points ∧ ∀ x ∈ D.points, (q - D.nearest q) ^ 2 ≤ (q - x) ^ 2 :=
  exactScan_spec _ _ _

/-- Every database coordinate other than the ground-truth result is strictly farther away. -/
def UniqueNearest (q : ℝ) : Prop :=
  ∀ x ∈ D.points, x ≠ D.nearest q → (q - D.nearest q) ^ 2 < (q - x) ^ 2

private theorem nearest_side (q : ℝ) (hu : D.UniqueNearest q) :
    (D.nearest q ∈ D.left → q < D.boundary) ∧
      (D.nearest q ∈ D.right → D.boundary < q) := by
  obtain ⟨hL, hmax⟩ := D.leftMax_spec
  obtain ⟨hU, hmin⟩ := D.rightMin_spec
  constructor
  · intro hl
    have hsep := D.separated _ hl _ hU
    have hlt := hu _ (List.mem_append_right _ hU) hsep.ne'
    have hq : q < mid (D.nearest q) D.rightMin :=
      lt_of_le_of_ne ((dist_le_iff _ _ _ hsep).mp hlt.le)
        (fun he => hlt.ne ((dist_eq_iff _ _ _ hsep).mpr he))
    unfold boundary mid at *
    linarith [hmax _ hl]
  · intro hr
    have hsep := D.separated _ hL _ hr
    have hlt := hu _ (List.mem_append_left _ hL) hsep.ne
    have hq : mid D.leftMax (D.nearest q) < q :=
      lt_of_not_ge (fun he => hlt.not_ge ((dist_le_iff _ _ _ hsep).mpr he))
    unfold boundary mid at *
    linarith [hmin _ hr]

/-- The global scan lands on the left exactly below the midpoint of the gap in the data. -/
theorem nearest_mem_left_iff (q : ℝ) (hu : D.UniqueNearest q) :
    D.nearest q ∈ D.left ↔ q < D.boundary := by
  obtain ⟨hl, hr⟩ := D.nearest_side q hu
  exact ⟨hl, fun hq => (List.mem_append.mp (D.nearest_spec q).1).resolve_right
    (fun h => hq.asymm (hr h))⟩

def probe (C : Router) (q : ℝ) : List ℝ := if C.RoutesLeft q then D.left else D.right

theorem probe_ne (C : Router) (q : ℝ) : D.probe C q ≠ [] := by
  unfold probe
  split
  · exact D.left_ne
  · exact D.right_ne

def search (C : Router) (q : ℝ) : ℝ := exactScan q (D.probe C q) (D.probe_ne C q)

/-- One distance evaluation per center and per point in the selected list. -/
def distanceWork (C : Router) (q : ℝ) : ℕ := 2 + (D.probe C q).length

theorem search_eq_nearest_iff (C : Router) (q : ℝ) (hu : D.UniqueNearest q) :
    D.search C q = D.nearest q ↔ D.nearest q ∈ D.probe C q := by
  obtain ⟨hmem, hmin⟩ := exactScan_spec q (D.probe C q) (D.probe_ne C q)
  change D.search C q ∈ D.probe C q at hmem
  constructor
  · intro he
    rwa [← he]
  · intro h
    by_contra hne
    have hall : D.search C q ∈ D.points := by
      unfold probe at hmem
      split at hmem
      · exact List.mem_append_left _ hmem
      · exact List.mem_append_right _ hmem
    exact (hu _ hall hne).not_ge (hmin _ h)

private theorem mem_probe_iff (C : Router) (q x : ℝ) (hx : x ∈ D.points) :
    x ∈ D.probe C q ↔ (C.RoutesLeft q ↔ x ∈ D.left) := by
  by_cases hc : C.RoutesLeft q
  · simp [probe, hc]
  · simp only [probe, hc, if_false, false_iff]
    exact ⟨fun hr hl => (lt_irrefl x) (D.separated _ hl _ hr),
      (List.mem_append.mp hx).resolve_left⟩

/-- The two scans disagree exactly in the open interval between the two midpoints,
in either order. -/
theorem miss_iff (C : Router) (q : ℝ) (hu : D.UniqueNearest q) (hc : C.Untied q) :
    D.search C q ≠ D.nearest q ↔ q ∈ Set.uIoo D.boundary C.boundary := by
  have hm := (D.nearest_spec q).1
  obtain ⟨hl, hr⟩ := D.nearest_side q hu
  rw [ne_eq, D.search_eq_nearest_iff C q hu, D.mem_probe_iff C q _ hm,
    C.routesLeft_iff q hc, D.nearest_mem_left_iff q hu]
  change ¬ (q < C.boundary ↔ q < D.boundary) ↔
    min D.boundary C.boundary < q ∧ q < max D.boundary C.boundary
  rw [min_lt_iff, lt_max_iff]
  rcases lt_or_gt_of_ne ((C.untied_iff q).mp hc) with h | h <;>
    rcases (List.mem_append.mp hm).imp hl hr with h' | h' <;>
    simp [h, h', h.not_gt, h'.not_gt]

/-- Empirical recall@1 counts equal returned coordinates, including query multiplicities. -/
def recallAtOne (C : Router) (qs : List ℝ) : ℝ :=
  (qs.countP (fun q => decide (D.search C q = D.nearest q)) : ℝ) / qs.length

def missRate (C : Router) (qs : List ℝ) : ℝ := 1 - D.recallAtOne C qs

end PostingLists

def intervalCount (a b : ℝ) (qs : List ℝ) : ℕ :=
  qs.countP (fun q => decide (q ∈ Set.uIoo a b))

namespace PostingLists

variable (D : PostingLists)

theorem missRate_eq_interval_fraction (C : Router) (qs : List ℝ) (hne : qs ≠ [])
    (hu : ∀ q ∈ qs, D.UniqueNearest q) (hc : ∀ q ∈ qs, C.Untied q) :
    D.missRate C qs = (intervalCount D.boundary C.boundary qs : ℝ) / qs.length := by
  let hit := fun q => decide (D.search C q = D.nearest q)
  have hcount : qs.countP (fun q => ¬ hit q) = intervalCount D.boundary C.boundary qs := by
    apply List.countP_congr
    intro q hq
    simpa [hit] using D.miss_iff C q (hu q hq) (hc q hq)
  have hsum := List.length_eq_countP_add_countP (l := qs) hit
  rw [hcount] at hsum
  have hlen : (qs.length : ℝ) ≠ 0 := by exact_mod_cast (List.length_pos_iff.mpr hne).ne'
  change 1 - (qs.countP hit : ℝ) / qs.length = _
  apply (eq_div_iff hlen).mpr
  rw [sub_mul, one_mul, div_mul_cancel₀ _ hlen, sub_eq_iff_eq_add, add_comm]
  exact_mod_cast hsum

/-- With the same data lists, the better router is exactly the one with fewer
batch queries in its miss interval. -/
theorem recall_improves_iff (old new : Router) (qs : List ℝ) (hne : qs ≠ [])
    (hu : ∀ q ∈ qs, D.UniqueNearest q) (ho : ∀ q ∈ qs, old.Untied q)
    (hn : ∀ q ∈ qs, new.Untied q) :
    D.recallAtOne old qs < D.recallAtOne new qs ↔
      intervalCount D.boundary new.boundary qs < intervalCount D.boundary old.boundary qs := by
  have hlen : 0 < (qs.length : ℝ) := by exact_mod_cast List.length_pos_iff.mpr hne
  calc
    _ ↔ D.missRate new qs < D.missRate old qs := (sub_lt_sub_iff_left (1 : ℝ)).symm
    _ ↔ _ := by
      rw [D.missRate_eq_interval_fraction new qs hne hu hn,
        D.missRate_eq_interval_fraction old qs hne hu ho, div_lt_div_iff_of_pos_right hlen,
        Nat.cast_lt]

end PostingLists

/-! ## A Lloyd update that lowers both loss and recall@1 -/

namespace LloydExample

def sites : Four := ⟨0, 4, 8, 10, by norm_num, by norm_num, by norm_num⟩

def lists : PostingLists where
  left := [sites.a, sites.b]
  right := [sites.c, sites.d]
  left_ne := by simp
  right_ne := by simp
  separated := by norm_num [sites, List.forall_mem_cons]

def seeds : Centers sites := Centers.ofInterval sites.b sites.c sites.ab.le sites.bc sites.cd.le

def updated : Centers sites := seeds.next Weights.unit

/-- Starting at 4 and 8, one assignment-and-mean step gives 2 and 9.
Reassigning the four points gives the same two lists, and the training SSE halves. -/
theorem update :
    (updated.left = 2 ∧ updated.right = 9) ∧
    (seeds.toRouter.postings lists.points = (lists.left, lists.right) ∧
      updated.toRouter.postings lists.points = (lists.left, lists.right)) ∧
    (seeds.sse Weights.unit = 20 ∧ updated.sse Weights.unit = 10) := by
  have hcut : sites.cut seeds.boundary = .two := by
    norm_num [Four.cut, Centers.boundary, seeds, Centers.ofInterval, mid, sites]
  have hc : updated.left = 2 ∧ updated.right = 9 := by
    simp only [updated, Centers.next, hcut]
    norm_num [Centers.ofInterval, mean2, Weights.unit, sites]
  refine ⟨hc, ?_, ?_⟩
  · constructor <;>
      norm_num [Router.postings, Router.RoutesLeft, Centers.toRouter, hc.1, hc.2, seeds,
        Centers.ofInterval, PostingLists.points, lists, sites]
  · constructor <;>
      norm_num [Centers.sse, hc.1, hc.2, seeds, Centers.ofInterval, Weights.unit, sites, min_def]

def query : ℝ := 23 / 4

/-- The global nearest neighbor is 4. The first probe returns it and the second
returns 8, with four distance evaluations in each search. -/
theorem query_result :
    lists.UniqueNearest query ∧ (seeds.toRouter.Untied query ∧ updated.toRouter.Untied query) ∧
    lists.nearest query = 4 ∧
    (lists.search seeds.toRouter query = 4 ∧ lists.search updated.toRouter query = 8) ∧
    (lists.distanceWork seeds.toRouter query = 4 ∧ lists.distanceWork updated.toRouter query = 4) := by
  obtain ⟨hc, _, _⟩ := update
  have hnear : lists.nearest query = 4 := by
    norm_num [PostingLists.nearest, exactScan, PostingLists.points, lists, sites, query,
      List.argmin_cons]
  have hr : seeds.toRouter.RoutesLeft query ∧ ¬ updated.toRouter.RoutesLeft query := by
    norm_num [Router.RoutesLeft, Centers.toRouter, hc.1, hc.2, seeds, Centers.ofInterval, sites, query]
  refine ⟨?_, ?_, hnear, ?_, ?_⟩
  · unfold PostingLists.UniqueNearest
    rw [hnear]
    norm_num [PostingLists.points, lists, sites, query, List.forall_mem_cons]
  · norm_num [Router.Untied, Centers.toRouter, hc.1, hc.2, seeds, Centers.ofInterval, sites, query]
  · simp only [PostingLists.search, PostingLists.probe, if_pos hr.1, if_neg hr.2]
    norm_num [exactScan, List.argmin_cons, lists, sites, query]
  · simp [PostingLists.distanceWork, PostingLists.probe, hr.1, hr.2, lists]

theorem singleton_recall :
    lists.recallAtOne seeds.toRouter [query] = 1 ∧ lists.recallAtOne updated.toRouter [query] = 0 := by
  obtain ⟨_, _, hnear, ⟨hold, hnew⟩, _⟩ := query_result
  norm_num [PostingLists.recallAtOne, List.countP_cons, hold, hnew, hnear]

end LloydExample

end IndexRecall

#print axioms IndexRecall.PostingLists.nearest_mem_left_iff
#print axioms IndexRecall.Router.routesLeft_iff
#print axioms IndexRecall.PostingLists.miss_iff
#print axioms IndexRecall.PostingLists.missRate_eq_interval_fraction
#print axioms IndexRecall.PostingLists.recall_improves_iff
#print axioms IndexRecall.LloydExample.update
#print axioms IndexRecall.LloydExample.query_result
#print axioms IndexRecall.LloydExample.singleton_recall
