import IndexRecall.EMeansGeometry
import IndexRecall.EMeansWitnessOptimality
import IndexRecall.SphericalIVF

/-! # The explicit parameter used by the spherical e-means replay -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap.EMeans

open Vectors

variable {k : ℕ}

def replayParam (k : ℕ) : ℝ := 1 / (1000 * (k : ℝ) ^ 2)

theorem replayParam_pos (hk : 2 ≤ k) : 0 < replayParam k := by
  have hk' : (0 : ℝ) < k := by exact_mod_cast (by omega : 0 < k)
  unfold replayParam
  positivity

private theorem k_sq_gt_one (hk : 2 ≤ k) : (1 : ℝ) < (k : ℝ) ^ 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  nlinarith [sq_nonneg ((k : ℝ) - 2)]

/-- Each of the `k²` documents moves by at most `1 / (100 k²)`. -/
theorem replay_close (hk : 2 ≤ k) (z : Fin k) (p : Fin k × Fin k) :
    ‖site z (replayParam k) p - base z p‖ ≤ 1 / (100 * (k : ℝ) ^ 2) := by
  have hk' : (0 : ℝ) < k := by exact_mod_cast (by omega : 0 < k)
  have h1 : replayParam k ≤ 1 := by
    apply (div_le_one (show (0 : ℝ) < 1000 * (k : ℝ)^2 by positivity)).2
    linarith [k_sq_gt_one hk]
  refine (site_displacement z (replayParam_pos hk).le h1 p).trans ?_
  unfold replayParam
  field_simp
  nlinarith [sq_nonneg (k : ℝ)]

private theorem replay_close_one (hk : 2 ≤ k) (z : Fin k) (p : Fin k × Fin k) :
    ‖site z (replayParam k) p - base z p‖ < 1 / 100 := by
  have hk' : (0 : ℝ) < k := by exact_mod_cast (by omega : 0 < k)
  apply (replay_close hk z p).trans_lt
  apply (div_lt_div_iff₀ (by positivity) (by norm_num)).2
  linarith [k_sq_gt_one hk]

/-- The replay parameter belongs to the stable cell regime for every `k ≥ 2`. -/
theorem replay_geometry (hk : 2 ≤ k) (z : Fin k) :
    let x := site z (replayParam k)
    (∀ p, x p ∈ cell x p.1) ∧
      (∀ i (v : Space k), ‖v - anchor z i‖ < 1 / 100 → unitDir v ∈ cell x i) :=
  cells_of_close hk z _ (site_unit z (replayParam k)) (replay_close_one hk z)

private theorem projection_sum_le {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : Fin k × Fin k → E) (i : Fin k) (v : E) (r : ℝ)
    (h : ∀ j, r ≤ ⟪v, x (i,j)⟫_ℝ) : (k : ℝ) * r ≤ ⟪v, rowSum x i⟫_ℝ := calc
  _ = ∑ _j : Fin k, r := by simp [nsmul_eq_mul]
  _ ≤ ∑ j, ⟪v, x (i,j)⟫_ℝ := Finset.sum_le_sum fun j _ => h j
  _ = _ := by rw [rowSum, inner_sum]

/-- At the replay parameter, the target score is at most `1/3` and each other
mean score is strictly between `1/3` and the tip score `1/2`. -/
theorem replay_routing (hk : 2 ≤ k) (z : Fin k) :
    let x := site z (replayParam k)
    (∀ i, rowSum x i ≠ 0) ∧ ⟪query k, meanDirection x z⟫_ℝ ≤ 1 / 3 ∧
      (∀ i ≠ z, 1 / 3 < ⟪query k, meanDirection x i⟫_ℝ ∧ ⟪query k, meanDirection x i⟫_ℝ < 1 / 2) := by
  intro x
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hm (i) : anchor z i ∈ cell x i := by
    simpa [unitDir, anchor_unit] using (replay_geometry hk z).2 i (anchor z i) (by simp)
  have hnorm (i) : (k : ℝ) * (3 / 4) ≤ ‖rowSum x i‖ := calc
    _ ≤ ⟪anchor z i, rowSum x i⟫_ℝ := projection_sum_le _ _ _ _ fun j => by
      rw [real_inner_comm]; exact (hm i).2.1 (i,j) rfl
    _ ≤ _ := by simpa [anchor_unit] using real_inner_le_norm (anchor z i) (rowSum x i)
  have hpos (i) : 0 < ‖rowSum x i‖ := by linarith [hnorm i]
  have htarget : ⟪query k, rowSum x z⟫_ℝ = 1 / 2 := by simp [x, rowSum, inner_sum, query_target]
  have hdir (i) : ⟪query k, meanDirection x i⟫_ℝ = ⟪query k, rowSum x i⟫_ℝ / ‖rowSum x i‖ := by
    simp [meanDirection, unitDir, real_inner_smul_right, div_eq_mul_inv, mul_comm]
  refine ⟨fun i => norm_pos_iff.mp (hpos i), ?_, ?_⟩
  · rw [hdir, htarget, div_le_iff₀ (hpos _)]
    linarith [hnorm z]
  · intro i hi
    have hmovement (v : Space k) (hv : ‖v‖ = 1) (j : Fin k) :
        |⟪v, x (i,j)⟫_ℝ - ⟪v, anchor z i⟫_ℝ| < 1 / 100 := calc
      _ ≤ ‖x (i,j) - anchor z i‖ := by
        simpa only [inner_sub_right, hv, one_mul] using abs_real_inner_le_norm v (x (i,j) - anchor z i)
      _ < _ := by simpa [x, base, hi] using replay_close_one hk z (i,j)
    have hquery (j : Fin k) := hmovement (query k) (query_unit k) j
    simp only [query_anchor, if_neg hi] at hquery
    have hlo := projection_sum_le x i (query k) (39 / 100) fun j => by linarith [(abs_lt.mp (hquery j)).1]
    have hhi : ⟪query k, rowSum x i⟫_ℝ ≤ (k : ℝ) * (41 / 100) := calc
      _ = ∑ j, ⟪query k, x (i,j)⟫_ℝ := by rw [rowSum, inner_sum]
      _ ≤ ∑ _j : Fin k, (41 / 100 : ℝ) := Finset.sum_le_sum fun j _ => by linarith [(abs_lt.mp (hquery j)).2]
      _ = _ := by simp [nsmul_eq_mul]
    have hA := projection_sum_le x i (anchor z i) (99 / 100) fun j => by
      have hj := hmovement (anchor z i) (anchor_unit z i) j
      simp only [real_inner_self_eq_norm_sq, anchor_unit, one_pow] at hj
      linarith [(abs_lt.mp hj).1]
    have hN : ‖rowSum x i‖ ≤ k := calc
      _ ≤ ∑ j, ‖x (i,j)‖ := norm_sum_le _ _
      _ = _ := by simp [x, site_unit]
    have hn' : ⟪anchor z i, rowSum x i⟫_ℝ ≤ ‖rowSum x i‖ := by
      simpa [anchor_unit] using real_inner_le_norm (anchor z i) (rowSum x i)
    rw [hdir, lt_div_iff₀ (hpos _), div_lt_iff₀ (hpos _)]
    constructor <;> linarith

/-- The explicit real-arithmetic replay instance has the global spherical optimum,
the document margin, both exact probe thresholds, and the flat-scan cost ratio. -/
theorem replay_gap (hk : 2 ≤ k) (z : Fin k) :
    CosineGap (site z (replayParam k)) (query k) z := by
  let x := site z (replayParam k)
  obtain ⟨hn, htarget, hother⟩ := replay_routing hk z
  have last (i : Fin k) (hi : i ≠ z) :
      ⟪query k, meanDirection x z⟫_ℝ < ⟪query k, meanDirection x i⟫_ℝ :=
    htarget.trans_lt (hother i hi).1
  have first (i : Fin k) (hi : i ≠ z) :
      ⟪query k, retunedDirection x z i⟫_ℝ < ⟪query k, retunedDirection x z z⟫_ℝ := by
    simpa [retunedDirection, hi, x, site_tip, query_tip] using (hother i hi).2
  have ho : UniqueOptimum x :=
    uniqueOptimum_of_quality x hn (strict_quality_of_close hk z x (replay_close hk z))
  refine ⟨by simp [sq], row_card, site_injective z (replayParam_pos hk),
    site_unit z (replayParam k), query_unit k, ho, ?_, site_margin z (replayParam k), last, first,
    scan_gap _ _ _ _ ?_ (fun i hi => neg_lt_neg (last i hi)) (fun i hi => neg_lt_neg (first i hi))⟩
  · rw [site_tip, query_tip]
  · intro b hb
    exact neg_lt_neg (by linarith [site_margin z (replayParam k) b hb])

end IndexRecall.SphericalGap.EMeans
