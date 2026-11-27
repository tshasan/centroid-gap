import IndexRecall.FlatScan
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! # Cosine routing and exact scan costs for a strictly last owner -/

noncomputable section

namespace IndexRecall.SphericalGap

open scoped InnerProductSpace

variable {k : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The vector sum of the documents whose owner is `i`. -/
def rowSum (x : Fin k × Fin k → E) (i : Fin k) : E := ∑ j, x (i, j)

/-- Direction of a vector, with unit norm whenever the vector is nonzero. -/
def unitDir (v : E) : E := ‖v‖⁻¹ • v

/-- The ordinary mean and the sum yield the same spherical representative. -/
theorem normalized_mean_eq (hk : 0 < k) (x : Fin k × Fin k → E) (i : Fin k) :
    unitDir ((k : ℝ)⁻¹ • rowSum x i) = unitDir (rowSum x i) := by
  have hk' : 0 < (k : ℝ) := by exact_mod_cast hk
  simp [unitDir, norm_smul, mul_smul, hk'.ne']

/-- The target sum has query projection `1/2`, but its projection on its own anchor
exceeds `k - 1 + 4/5`. Every other normalized sum equals its anchor. -/
theorem repeated_routing (hk : 2 ≤ k) (z : Fin k) (q tip : E) (a : Fin k → E)
    (x : Fin k × Fin k → E) (hx : ∀ p, x p = if p = (z, z) then tip else a p.1)
    (ha : ∀ i, ‖a i‖ = 1) (hqz : ⟪q, a z⟫_ℝ = 0)
    (hqa : ∀ i ≠ z, ⟪q, a i⟫_ℝ = 2 / 5) (hqt : ⟪q, tip⟫_ℝ = 1 / 2)
    (ht : 4 / 5 < ⟪tip, a z⟫_ℝ) :
    (∀ i, rowSum x i ≠ 0) ∧
    (∀ i ≠ z, ⟪q, unitDir (rowSum x z)⟫_ℝ < ⟪q, unitDir (rowSum x i)⟫_ℝ) ∧
    (∀ i ≠ z, ⟪q, unitDir (rowSum x i)⟫_ℝ < ⟪q, tip⟫_ℝ) := by
  classical
  have hk' : 0 < (k : ℝ) := by exact_mod_cast (by omega : 0 < k)
  have htarget : rowSum x z = ((k - 1 : ℕ) : ℝ) • a z + tip := by
    unfold rowSum
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ z)]
    have hsum : (∑ j ∈ Finset.univ.erase z, x (z, j)) =
        ∑ _j ∈ Finset.univ.erase z, a z := by
      apply Finset.sum_congr rfl
      intro j hj
      simp [hx, (Finset.mem_erase.mp hj).1]
    rw [hsum]
    simp [hx, Nat.cast_smul_eq_nsmul]
  have hprojection : ((k - 1 : ℕ) : ℝ) + ⟪tip, a z⟫_ℝ ≤ ‖rowSum x z‖ := calc
    _ = ⟪rowSum x z, a z⟫_ℝ := by
      simp [htarget, inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq, ha]
    _ ≤ ‖rowSum x z‖ := by simpa [ha] using real_inner_le_norm (rowSum x z) (a z)
  have hk1 : 1 ≤ ((k - 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 1 ≤ k - 1)
  have hpos : 0 < ‖rowSum x z‖ := by linarith
  have hdot : ⟪q, rowSum x z⟫_ℝ = 1 / 2 := by
    simp [htarget, inner_add_right, real_inner_smul_right, hqz, hqt]
  have hlow : ⟪q, unitDir (rowSum x z)⟫_ℝ < 2 / 5 := by
    rw [unitDir, real_inner_smul_right, hdot, mul_comm, ← div_eq_mul_inv, div_lt_iff₀ hpos]
    linarith
  have hrow (i : Fin k) (hi : i ≠ z) : rowSum x i = (k : ℝ) • a i := by
    simp [rowSum, hx, hi, Nat.cast_smul_eq_nsmul]
  have hnorm (i : Fin k) (hi : i ≠ z) : ‖rowSum x i‖ = k := by
    rw [hrow i hi, norm_smul, Real.norm_eq_abs, abs_of_pos hk', ha, mul_one]
  have hdir (i : Fin k) (hi : i ≠ z) : unitDir (rowSum x i) = a i := by
    rw [unitDir, hnorm i hi, hrow i hi, inv_smul_smul₀ hk'.ne']
  refine ⟨?_, ?_, ?_⟩
  · intro i
    by_cases hi : i = z
    · exact hi ▸ norm_pos_iff.mp hpos
    · apply norm_pos_iff.mp
      rwa [hnorm i hi]
  · intro i hi
    rwa [hdir i hi, hqa i hi]
  · intro i hi
    rw [hdir i hi, hqa i hi, hqt]
    norm_num

/-- Search for the same document before and after replacing the owner's representative.
Scores are dissimilarities, so a lower value is probed first. -/
structure ScanGap (d : Fin k × Fin k → ℝ) (m r : Fin k → ℝ) (answer : Fin k × Fin k) : Prop where
  mean_rank : FlatScan.rank m answer.1 = k
  retuned_rank : FlatScan.rank r answer.1 = 1
  mean_returns : ∀ p, FlatScan.Returns d m p answer ↔ k ≤ p
  retuned_returns : ∀ p, FlatScan.Returns d r p answer ↔ 1 ≤ p
  mean_work : FlatScan.work (Fin k) m k = k + k * k
  retuned_work : FlatScan.work (Fin k) r 1 = k + k
  ratio : (FlatScan.work (Fin k) m k : ℝ) / FlatScan.work (Fin k) r 1 = ((k : ℝ) + 1) / 2

/-- Only the owner's comparisons must be strict. Ties among the remaining lists do not
affect the first and last probe budgets. -/
theorem scan_gap (d : Fin k × Fin k → ℝ) (m r : Fin k → ℝ) (answer : Fin k × Fin k)
    (hd : ∀ b ≠ answer, d answer < d b) (hm : ∀ i ≠ answer.1, m i < m answer.1)
    (hr : ∀ i ≠ answer.1, r answer.1 < r i) : ScanGap d m r answer := by
  classical
  have hk : 0 < k := Fin.pos answer.1
  have better (s : Fin k → ℝ) (i : Fin k) :
      Finset.univ.filter (fun j => s j < s i) ⊆ Finset.univ.erase i := by
    intro j hj
    exact Finset.mem_erase.mpr ⟨fun e => (Finset.mem_filter.mp hj).2.ne (congrArg s e),
      Finset.mem_univ j⟩
  have hmrank : FlatScan.rank m answer.1 = k := by
    simpa using FlatScan.rank_last hm
  have hrrank := FlatScan.rank_first hr
  have hall : FlatScan.probed m k = Finset.univ := by
    ext i
    simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
    have hcard := Finset.card_le_card (better m i)
    simp only [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ,
      Fintype.card_fin] at hcard
    unfold FlatScan.rank
    omega
  have hone : FlatScan.probed r 1 = {answer.1} := by
    ext i
    simp only [FlatScan.probed, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_singleton]
    by_cases hi : i = answer.1
    · simp [hi, hrrank]
    · have hpos : 0 < (Finset.univ.filter (fun j => r j < r i)).card :=
        Finset.card_pos.mpr ⟨answer.1, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hr i hi⟩⟩
      simp only [hi, iff_false, not_le, FlatScan.rank]
      omega
  have hw : FlatScan.work (Fin k) m k = k + k * k := by
    simp [FlatScan.work, FlatScan.candidates, hall]
  have hwr : FlatScan.work (Fin k) r 1 = k + k := by
    simp [FlatScan.work, FlatScan.candidates, hone]
  refine ⟨hmrank, hrrank, fun p => FlatScan.returns_iff _ p _ hd hmrank,
    fun p => FlatScan.returns_iff _ p _ hd hrrank, hw, hwr, ?_⟩
  rw [hw, hwr]
  push_cast
  have hk' : 0 < (k : ℝ) := by exact_mod_cast hk
  rw [div_eq_iff (by positivity)]
  ring

end IndexRecall.SphericalGap
