import IndexRecall.SphericalRouting
import Mathlib.Analysis.Convex.Combination
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Finset.Max

/-! # Moving assignments and EMA updates for spherical NumpyEmeans

Real-arithmetic model of `begin`, the logical-batch snapshot in `accumulate`,
and the position/mass update in `_apply_statistics` at Faiss commit
`2b648f0605f2ef3d14c4705350f8a0b214463878`. The recursion represents steps
on which revival leaves the centers alone.
-/

noncomputable section

open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

abbrev Address (k : ℕ) := Fin k × Fin k

structure State (k : ℕ) (E : Type*) where
  pos : Fin k → E
  mass : Fin k → ℝ

/-- Each logical batch uses one centroid snapshot for every occurrence in its row list. -/
structure Batch (k : ℕ) where
  rows : List (Address k)
  alpha : ℝ
  alpha_nonneg : 0 ≤ alpha
  alpha_le_one : alpha ≤ 1

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def beginState (c : Fin k → E) : State k E :=
  ⟨fun i => unitDir (c i), fun _ => 2 ^ (-16 : ℤ)⟩

private theorem exists_best (x : Address k → E) (c : Fin k → E) (p : Address k) :
    ∃ i : Fin k, ∀ j, ⟪x p, c j⟫_ℝ ≤ ⟪x p, c i⟫_ℝ := by
  obtain ⟨i, _, hi⟩ := Finset.univ.exists_max_image (fun j => ⟪x p, c j⟫_ℝ)
    ⟨p.1, Finset.mem_univ _⟩
  exact ⟨i, fun j => hi j (Finset.mem_univ _)⟩

/-- Assign a document to a highest-scoring representative, also used for fresh IVF insertion. -/
def assignDirection (x : Address k → E) (c : Fin k → E) (p : Address k) : Fin k :=
  (exists_best x c p).choose

theorem assignDirection_max (x : Address k → E) (c : Fin k → E) (p : Address k) (j : Fin k) :
    ⟪x p, c j⟫_ℝ ≤ ⟪x p, c (assignDirection x c p)⟫_ℝ :=
  (exists_best x c p).choose_spec j

/-- Assign by maximal inner product with the normalized raw position. -/
def predict (x : Address k → E) (s : State k E) (p : Address k) : Fin k :=
  assignDirection x (fun i => unitDir (s.pos i)) p

theorem predict_max (x : Address k → E) (s : State k E) (p : Address k) (j : Fin k) :
    ⟪x p, unitDir (s.pos j)⟫_ℝ ≤ ⟪x p, unitDir (s.pos (predict x s p))⟫_ℝ :=
  assignDirection_max x (fun i => unitDir (s.pos i)) p j

/-- The count is the length of this list. The sum is the list sum of its document vectors. -/
def assigned (rows : List (Address k)) (f : Address k → Fin k) (i : Fin k) : List (Address k) :=
  rows.filter fun p => f p = i

/-- At step size one, empty centers keep their old mass and occupied centers replace it. -/
def updateMass (α m : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then (if α = 1 then m else (1 - α) * m)
  else (1 - α) * m + α * n

/-- Faiss sets a weight-one update directly to the batch mean. -/
def interpolate (w : ℝ) (position mean : E) : E :=
  if w = 1 then mean else position + w • (mean - position)

theorem interpolate_eq (w : ℝ) (position mean : E) :
    interpolate w position mean = (1 - w) • position + w • mean := by
  by_cases h : w = 1
  · simp [interpolate, h]
  · simp only [interpolate, if_neg h, sub_smul, one_smul, smul_sub]
    abel

def update (x : Address k → E) (s : State k E) (b : Batch k) : State k E :=
  let members := assigned b.rows (predict x s)
  let mass := fun i => updateMass b.alpha (s.mass i) (members i).length
  { mass
    pos := fun i =>
      let n := (members i).length
      if n = 0 then s.pos i else
        let w := b.alpha * n / mass i
        let mean := (n : ℝ)⁻¹ • ((members i).map x).sum
        interpolate w (s.pos i) mean }

/-- Run an ordered sequence of logical batches without splits. -/
def run (x : Address k → E) (s : State k E) : List (Batch k) → State k E
  | [] => s
  | b :: bs => run x (update x s b) bs

theorem run_append (x : Address k → E) (s : State k E) (as bs : List (Batch k)) :
    run x s (as ++ bs) = run x (run x s as) bs := by
  induction as generalizing s with
  | nil => rfl
  | cons a as ih => exact ih (update x s a)

/-- Last-pass records include the labels actually assigned while the centers move.
Repeated occurrences remain separate. -/
def captured (x : Address k → E) (s : State k E) : List (Batch k) → List (Address k × Fin k)
  | [] => []
  | b :: bs => b.rows.map (fun p => (p, predict x s p)) ++ captured x (update x s b) bs

/-- Convex regions for the raw positions, with strict separation after spherical readout. -/
structure Basin (x : Address k → E) where
  cell : Fin k → Set E
  convex : ∀ i, Convex ℝ (cell i)
  documents : ∀ p, x p ∈ cell p.1
  separates : ∀ c : Fin k → E, (∀ i, c i ∈ cell i) → ∀ (p : Address k) j, j ≠ p.1 →
    ⟪x p, unitDir (c j)⟫_ℝ < ⟪x p, unitDir (c p.1)⟫_ℝ

variable {x : Address k → E}

def Stable (B : Basin x) (s : State k E) : Prop :=
  (∀ i, s.pos i ∈ B.cell i) ∧ ∀ i, 0 < s.mass i

theorem begin_stable (B : Basin x) (c : Fin k → E) (hc : ∀ i, unitDir (c i) ∈ B.cell i) :
    Stable B (beginState c) := ⟨hc, fun _ => by dsimp [beginState]; positivity⟩

theorem predict_owner (B : Basin x) (s : State k E) (hs : Stable B s) (p : Address k) :
    predict x s p = p.1 := by
  by_contra h
  exact (B.separates s.pos hs.1 p _ h).not_ge (predict_max x s p p.1)

/-- The unweighted mean of a nonempty list in a convex cell belongs to that cell. -/
theorem list_mean_mem (C : Set E) (hc : Convex ℝ C) (l : List E) (hn : 0 < l.length)
    (h : ∀ v ∈ l, v ∈ C) : (l.length : ℝ)⁻¹ • l.sum ∈ C := by
  have hsum : (∑ i : Fin l.length, l.get i) = l.sum := by rw [← Fin.sum_ofFn, List.ofFn_get]
  have hm := hc.centerMass_mem (t := Finset.univ) (z := l.get) (w := fun _ => (1 : ℝ))
    (fun _ _ => zero_le_one) (by simpa using (show (0 : ℝ) < l.length by exact_mod_cast hn))
    (fun i _ => h _ (l.get_mem i))
  simpa [Finset.centerMass, hsum] using hm

theorem updateMass_pos {α m : ℝ} (n : ℕ) (ha : 0 ≤ α) (hα : α ≤ 1) (hm : 0 < m) :
    0 < updateMass α m n := by
  rcases ha.eq_or_lt with hz | hpos
  · subst α
    simpa [updateMass] using hm
  unfold updateMass
  split_ifs with hn he
  · exact hm
  · exact mul_pos (sub_pos.mpr (lt_of_le_of_ne hα he)) hm
  · have hn' : 0 < (n : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hn
    exact add_pos_of_nonneg_of_pos (mul_nonneg (sub_nonneg.mpr hα) hm.le) (mul_pos hpos hn')

/-- A nonempty center moves towards its batch mean with a weight in `[0,1]`.
An empty center keeps its raw position. Both updates preserve a convex set. -/
theorem update_pos_mem (x : Address k → E) (s : State k E) (b : Batch k) (i : Fin k)
    {C : Set E} (hc : Convex ℝ C) (hmass : 0 < s.mass i) (hold : s.pos i ∈ C)
    (hmean : let rows := assigned b.rows (predict x s) i
      rows.length ≠ 0 → (rows.length : ℝ)⁻¹ • (rows.map x).sum ∈ C) :
    (update x s b).pos i ∈ C := by
  let rows := assigned b.rows (predict x s) i
  let n := rows.length
  let m := updateMass b.alpha (s.mass i) n
  have hm : 0 < m := updateMass_pos n b.alpha_nonneg b.alpha_le_one hmass
  let w := b.alpha * n / m
  let mean := (n : ℝ)⁻¹ • (rows.map x).sum
  change (if n = 0 then s.pos i else interpolate w (s.pos i) mean) ∈ C
  split_ifs with hn
  · exact hold
  · have hn' : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
    have hw : 0 ≤ w := div_nonneg (mul_nonneg b.alpha_nonneg hn'.le) hm.le
    have hw' : w ≤ 1 := by
      apply (div_le_one hm).2
      change b.alpha * n ≤ updateMass b.alpha (s.mass i) n
      rw [updateMass, if_neg hn]
      exact le_add_of_nonneg_left (mul_nonneg (sub_nonneg.mpr b.alpha_le_one) hmass.le)
    rw [interpolate_eq]
    exact hc hold (hmean hn) (sub_nonneg.mpr hw') hw (sub_add_cancel _ _)

/-- A snapshot computed from the positions sends each occurrence to its own row. -/
theorem update_stable (B : Basin x) (s : State k E) (hs : Stable B s) (b : Batch k) :
    Stable B (update x s b) := by
  refine ⟨fun i => update_pos_mem x s b i (B.convex i) (hs.2 i) (hs.1 i) ?_,
    fun i => updateMass_pos _ b.alpha_nonneg b.alpha_le_one (hs.2 i)⟩
  intro rows hn
  have hcontent (v : E) (hv : v ∈ rows.map x) : v ∈ B.cell i := by
    obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hv
    have he : predict x s p = i := by simpa [assigned] using (List.mem_filter.mp hp).2
    rw [predict_owner B s hs p] at he
    simpa only [← he] using B.documents p
  simpa only [List.length_map] using list_mean_mem (B.cell i) (B.convex i) (rows.map x)
    (by simpa using Nat.pos_of_ne_zero hn) hcontent

theorem run_invariant (B : Basin x) (s : State k E) (hs : Stable B s) (bs : List (Batch k)) :
    Stable B (run x s bs) := by
  induction bs generalizing s with
  | nil => exact hs
  | cons b bs ih => exact ih (update x s b) (update_stable B s hs b)

/-- Taking zero or more batches covers each pre-batch snapshot and the terminal state. -/
theorem snapshot_strict (B : Basin x) (s : State k E) (hs : Stable B s)
    (bs : List (Batch k)) (n : ℕ) (p : Address k) (j : Fin k) (hj : j ≠ p.1) :
    let c := (run x s (bs.take n)).pos
    ⟪x p, unitDir (c j)⟫_ℝ < ⟪x p, unitDir (c p.1)⟫_ℝ :=
  B.separates _ (run_invariant B s hs _).1 p j hj

theorem captured_eq_owners (B : Basin x) (s : State k E) (hs : Stable B s)
    (bs : List (Batch k)) :
    captured x s bs = (bs.flatMap Batch.rows).map (fun p => (p, p.1)) := by
  induction bs generalizing s with
  | nil => rfl
  | cons b bs ih =>
    simp [captured, predict_owner B s hs, ih (update x s b) (update_stable B s hs b)]

/-- Every mass clears the pinned underuse test at a scheduled revival check. -/
def NoUnused (τ : ℝ) (s : State k E) : Prop :=
  ∀ i, τ * ((∑ j, s.mass j) / k) ≤ s.mass i

/-- All rows occur equally often in this logical batch, with multiplicity. -/
def BalancedRows (rows : List (Address k)) : Prop :=
  ∀ i j, (assigned rows Prod.fst i).length = (assigned rows Prod.fst j).length

def EqualMasses (s : State k E) : Prop := ∀ i j, s.mass i = s.mass j

theorem begin_equalMasses (c : Fin k → E) : EqualMasses (beginState c) := fun _ _ => rfl

theorem update_equalMasses (B : Basin x) (s : State k E) (hs : Stable B s) (hm : EqualMasses s)
    (b : Batch k) (hb : BalancedRows b.rows) : EqualMasses (update x s b) := by
  have he : predict x s = Prod.fst := funext (predict_owner B s hs)
  intro i j
  simp only [update, he, hm i j, hb i j]

theorem run_equalMasses (B : Basin x) (s : State k E) (hs : Stable B s) (hm : EqualMasses s)
    (bs : List (Batch k)) (hb : ∀ b ∈ bs, BalancedRows b.rows) : EqualMasses (run x s bs) := by
  induction bs generalizing s with
  | nil => exact hm
  | cons b bs ih =>
    exact ih (update x s b) (update_stable B s hs b)
      (update_equalMasses B s hs hm b (hb _ List.mem_cons_self))
      (fun a ha => hb a (List.mem_cons_of_mem _ ha))

theorem noUnused_zero (B : Basin x) (s : State k E) (hs : Stable B s) : NoUnused 0 s := by
  intro i
  simpa using (hs.2 i).le

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] in
theorem noUnused_equalMasses (s : State k E) (hpos : ∀ j, 0 < s.mass j) (hm : EqualMasses s)
    {τ : ℝ} (hτ : τ ≤ 1) : NoUnused τ s := by
  intro i
  have hk : (k : ℝ) ≠ 0 := by exact_mod_cast (Fin.pos i).ne'
  have hsum : (∑ j, s.mass j) = (k : ℝ) * s.mass i := calc
    _ = ∑ _j : Fin k, s.mass i := Finset.sum_congr rfl fun j _ => hm j i
    _ = _ := by simp
  rw [hsum, mul_div_cancel_left₀ _ hk]
  simpa using mul_le_mul_of_nonneg_right hτ (hpos i).le

/-- Faiss checks after each fifth global update. The offset is the number of steps
already completed, so a new epoch does not restart the counter. -/
def ChecksClear (τ : ℝ) (x : Address k → E) (s : State k E) (offset : ℕ)
    (bs : List (Batch k)) : Prop :=
  ∀ n, 0 < n → n ≤ bs.length → (offset + n) % 5 = 0 → NoUnused τ (run x s (bs.take n))

theorem checksClear_zero (B : Basin x) (s : State k E) (hs : Stable B s) (offset : ℕ)
    (bs : List (Batch k)) : ChecksClear 0 x s offset bs :=
  fun _ _ _ _ => noUnused_zero B _ (run_invariant B s hs _)

theorem checksClear_balanced (B : Basin x) (s : State k E) (hs : Stable B s)
    (hm : EqualMasses s) (offset : ℕ) (bs : List (Batch k)) (hb : ∀ b ∈ bs, BalancedRows b.rows)
    {τ : ℝ} (hτ : τ ≤ 1) : ChecksClear τ x s offset bs := by
  intro n _ _ _
  apply noUnused_equalMasses _ (run_invariant B s hs _).2 _ hτ
  exact run_equalMasses B s hs hm _ (fun b h => hb _ (List.mem_of_mem_take h))

/-- Stop at the first scheduled checkpoint with an under-threshold center. A successful
run includes exactly the EMA updates and empty-target checks of the pinned implementation. -/
def checkedRun (τ : ℝ) (x : Address k → E) (s : State k E) (offset : ℕ) :
    List (Batch k) → Option (State k E)
  | [] => some s
  | b :: bs => by
    classical
    let next := update x s b
    exact if (offset + 1) % 5 = 0 ∧ ¬NoUnused τ next then none
      else checkedRun τ x next (offset + 1) bs

theorem checkedRun_of_checksClear (τ : ℝ) (x : Address k → E) (s : State k E)
    (offset : ℕ) (bs : List (Batch k)) (hc : ChecksClear τ x s offset bs) :
    checkedRun τ x s offset bs = some (run x s bs) := by
  induction bs generalizing s offset with
  | nil => rfl
  | cons b bs ih =>
    have hfirst : ¬((offset + 1) % 5 = 0 ∧ ¬NoUnused τ (update x s b)) := by
      rintro ⟨hstep, hbad⟩
      exact hbad (hc 1 (by decide) (by simp) hstep)
    rw [checkedRun, if_neg hfirst]
    apply ih
    intro n hn hlen hstep
    have hindex : offset + (n + 1) = offset + 1 + n := by omega
    have h := hc (n + 1) (by omega) (by simpa using hlen) (hindex ▸ hstep)
    exact h

end IndexRecall.SphericalGap.EMeans
