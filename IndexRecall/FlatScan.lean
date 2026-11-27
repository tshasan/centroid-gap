import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.List.Count

/-! # Exact scans of equal-size lists with a flat bucket router -/

noncomputable section
namespace IndexRecall.FlatScan
open scoped Classical

variable {ι τ : Type*} [Fintype ι]

/-- One-based rank by increasing routing distance. -/
def rank (score : ι → ℝ) (i : ι) : ℕ :=
  1 + (Finset.univ.filter (fun j => score j < score i)).card

def probed (score : ι → ℝ) (p : ℕ) : Finset ι :=
  Finset.univ.filter (fun i => rank score i ≤ p)

def candidates (τ : Type*) [Fintype τ] (score : ι → ℝ) (p : ℕ) : Finset (ι × τ) :=
  probed score p ×ˢ Finset.univ

def Returns [Fintype τ] (pointScore : ι × τ → ℝ) (score : ι → ℝ) (p : ℕ) (a : ι × τ) : Prop :=
  a ∈ candidates τ score p ∧ ∀ b ∈ candidates τ score p, pointScore a ≤ pointScore b

def work (τ : Type*) [Fintype τ] (score : ι → ℝ) (p : ℕ) : ℕ :=
  Fintype.card ι + (candidates τ score p).card

theorem returns_iff [Fintype τ] {pointScore : ι × τ → ℝ} (score : ι → ℝ) (p : ℕ) (a : ι × τ)
    (hmin : ∀ b ≠ a, pointScore a < pointScore b) {n : ℕ} (hr : rank score a.1 = n) :
    Returns pointScore score p a ↔ n ≤ p := by
  subst n
  have hm : a ∈ candidates τ score p ↔ rank score a.1 ≤ p := by simp [candidates, probed]
  refine ⟨fun h => hm.mp h.1, fun h => ⟨hm.mpr h, fun b _ => ?_⟩⟩
  by_cases he : b = a
  · subst b; rfl
  · exact (hmin b he).le

theorem rank_first {score : ι → ℝ} {i : ι} (h : ∀ j ≠ i, score i < score j) :
    rank score i = 1 := by
  suffices ∀ j, ¬score j < score i by simp [rank, this]
  intro j
  by_cases hj : j = i
  · simp [hj]
  · exact (h j hj).asymm

theorem rank_last {score : ι → ℝ} {i : ι} (h : ∀ j ≠ i, score j < score i) :
    rank score i = Fintype.card ι := by
  classical
  have hall : Finset.univ.filter (fun j => score j < score i) = Finset.univ.erase i := by
    ext j
    by_cases hj : j = i
    · simp [hj]
    · simp [hj, h j hj]
  rw [rank, hall, Nat.add_comm, Finset.card_erase_add_one (Finset.mem_univ _), Finset.card_univ]

private theorem rank_lt {score : ι → ℝ} {a b : ι} (h : score a < score b) :
    rank score a < rank score b := by
  apply Nat.add_lt_add_left
  apply Finset.card_lt_card
  refine (Finset.ssubset_iff_of_subset ?_).2 ⟨a, by simp [h], by simp⟩
  intro i hi
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, ((Finset.mem_filter.mp hi).2).trans h⟩

/-- At the smallest budget that includes a designated row, count the actual
candidate addresses. Bucket scores must be distinct at this query. -/
theorem work_at_rank (τ : Type*) [Fintype τ] {score : ι → ℝ} (hi : Function.Injective score)
    (a : ι) : work τ score (rank score a) = Fintype.card ι + rank score a * Fintype.card τ := by
  have he : probed score (rank score a) =
      insert a (Finset.univ.filter (fun j => score j < score a)) := by
    ext j
    simp only [probed, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert]
    rcases lt_trichotomy (score j) (score a) with h | h | h
    · exact ⟨fun _ => Or.inr h, fun _ => (rank_lt h).le⟩
    · simp [hi h]
    · exact ⟨fun hr => ((rank_lt h).not_ge hr).elim, fun hj =>
        hj.elim (fun e => (h.ne (congrArg score e).symm).elim) (fun h' => (h.asymm h').elim)⟩
  simp only [work, candidates, he, Finset.card_product, Finset.card_univ]
  simp [rank, Nat.add_comm 1]

/-- Empirical recall for a top-1 hit predicate on a finite query batch. -/
def «recall» {Q : Type*} (hit : Q → Prop) (qs : List Q) : ℝ :=
  (qs.countP (fun q => decide (hit q)) : ℝ) / qs.length

theorem recall_of_constant {Q : Type*} {hit : Q → Prop} {qs : List Q} (hne : qs ≠ [])
    (condition : Prop) [Decidable condition] (h : ∀ q ∈ qs, hit q ↔ condition) :
    «recall» hit qs = if condition then 1 else 0 := by
  have hn : (qs.length : ℝ) ≠ 0 := by exact_mod_cast (List.length_pos_iff.mpr hne).ne'
  by_cases hc : condition
  · have hall : qs.countP (fun q => decide (hit q)) = qs.length :=
      List.countP_eq_length.mpr fun q hq => by simpa using (h q hq).mpr hc
    simp [«recall», hall, hc, hn]
  · have hnone : qs.countP (fun q => decide (hit q)) = 0 :=
      List.countP_eq_zero.mpr fun q hq => by simpa using mt (h q hq).mp hc
    simp [«recall», hnone, hc]

end IndexRecall.FlatScan
