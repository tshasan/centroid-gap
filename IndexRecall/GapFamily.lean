import IndexRecall.Geometry
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.NormNum

/-! # A balanced one-dimensional IVF family

Each row has one entry for each column. All entries are close to their
row anchor except the last entry of row zero, placed at `2*k`.
-/

noncomputable section

namespace IndexRecall.GapFamily

def core (k : ℕ) : ℕ := (32 * k) ^ 2
def width (k : ℕ) : ℕ := core k + 1

def anchor {k : ℕ} (i : Fin k) : ℝ :=
  if i.val = 0 then 0 else 6 * (k : ℝ) + (i.val : ℝ)

def offset (k : ℕ) (t : Fin (core k + 1)) : ℝ :=
  (t.val : ℝ) / (width k : ℝ) ^ 2

def site {k : ℕ} (i : Fin k) (t : Fin (core k + 1)) : ℝ :=
  if i.val = 0 ∧ t.val = core k then 2 * (k : ℝ) else anchor i + offset k t

theorem width_pos (k : ℕ) : 0 < width k := Nat.zero_lt_succ _

theorem offset_nonneg (k : ℕ) (t : Fin (core k + 1)) : 0 ≤ offset k t := by
  unfold offset
  positivity

theorem offset_le (k : ℕ) (t : Fin (core k + 1)) : offset k t ≤ 1 / (width k : ℝ) := by
  have hw : 0 < (width k : ℝ) := by exact_mod_cast width_pos k
  have ht : (t.val : ℝ) ≤ width k := by exact_mod_cast t.isLt.le
  rw [offset, sq, div_le_div_iff₀ (mul_pos hw hw) hw]
  simpa only [one_mul] using mul_le_mul_of_nonneg_right ht hw.le

theorem offset_small {k : ℕ} (hk : 2 ≤ k) (t : Fin (core k + 1)) :
    offset k t ≤ 1 / 8 := by
  have hw : 0 < (width k : ℝ) := by exact_mod_cast width_pos k
  have h8 : (8 : ℝ) ≤ width k := by
    exact_mod_cast (show 8 ≤ width k by dsimp [width, core]; nlinarith)
  exact (offset_le k t).trans ((div_le_div_iff₀ hw (by norm_num : (0 : ℝ) < 8)).2 (by linarith))

private theorem anchor_step {k : ℕ} {i j : Fin k} (h : i < j) :
    anchor i + 1 ≤ anchor j := by
  have hij : (i.val : ℝ) + 1 ≤ j.val := by exact_mod_cast (show i.val + 1 ≤ j.val from h)
  have hj : j.val ≠ 0 := Nat.ne_of_gt (Nat.zero_le i.val |>.trans_lt h)
  have hk : 0 ≤ (k : ℝ) := by positivity
  by_cases hi : i.val = 0 <;> simp [anchor, hi, hj] <;> linarith

theorem anchor_separated {k : ℕ} (i j : Fin k) (hij : i ≠ j) :
    1 ≤ |anchor i - anchor j| := by
  rcases lt_or_gt_of_ne hij with h | h
  · linarith [anchor_step h, neg_abs_le (anchor i - anchor j)]
  · linarith [anchor_step h, le_abs_self (anchor i - anchor j)]

theorem core_close {k : ℕ} (hk : 2 ≤ k) (i : Fin k) (t : Fin (core k)) :
    |site i t.castSucc - anchor i| ≤ 1 / 8 := by
  simpa [site, t.isLt.ne, abs_of_nonneg (offset_nonneg k t.castSucc)]
    using offset_small hk t.castSucc

/-- Moving every center by less than a quarter preserves all row assignments. -/
theorem nearest_anchor_row {k : ℕ} (hk : 2 ≤ k) (i j : Fin k) (t : Fin (core k + 1))
    (u v : ℝ) (hij : i ≠ j) (hu : |u - anchor i| < 1 / 4) (hv : |v - anchor j| < 1 / 4) :
    |site i t - u| < |site i t - v| := by
  by_cases tip : i.val = 0 ∧ t.val = core k
  · have hj : j.val ≠ 0 := fun hz => hij (Fin.ext (tip.1.trans hz.symm))
    simp only [anchor, tip.1, if_true, sub_zero, abs_lt] at hu
    simp only [anchor, hj, if_false, abs_lt] at hv
    have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
    have hj' : (0 : ℝ) ≤ j.val := by positivity
    rw [site, if_pos tip, abs_of_nonneg (by linarith : 0 ≤ 2 * (k : ℝ) - u),
      abs_of_nonpos (by linarith : 2 * (k : ℝ) - v ≤ 0)]
    linarith
  · have hx : |site i t - anchor i| ≤ 1 / 8 := by
      simpa [site, tip, abs_of_nonneg (offset_nonneg k t)] using offset_small hk t
    have hown := abs_sub_le (site i t) (anchor i) u
    rw [abs_sub_comm (anchor i) u] at hown
    have hother := abs_sub_le (anchor i) (site i t) (anchor j)
    rw [abs_sub_comm (anchor i) (site i t)] at hother
    linarith [abs_sub_le (site i t) v (anchor j), anchor_separated i j hij]

/-- A cheap feasible clustering at the anchors forces a center near each core. -/
theorem anchor_loss_lt {k : ℕ} (hk : 2 ≤ k) :
    (∑ i : Fin k, ∑ t, (site i t - anchor i) ^ 2) < (core k : ℝ) / 64 := by
  let z : Fin k := ⟨0, by omega⟩
  let s := Fin.last (core k)
  have hw : (width k : ℝ) ≠ 0 := by exact_mod_cast (width_pos k).ne'
  have hpoint (i : Fin k) (t : Fin (core k + 1)) :
      (site i t - anchor i) ^ 2 ≤ 1 / (width k : ℝ) +
        (if t = s then (if i = z then (2 * (k : ℝ)) ^ 2 else 0) else 0) := by
    by_cases h : i.val = 0 ∧ t.val = core k
    · have hi : i = z := Fin.ext h.1
      have ht : t = s := Fin.ext h.2
      simp only [site, if_pos h, anchor, if_pos h.1, sub_zero, if_pos hi, if_pos ht]
      linarith [offset_nonneg k t, offset_le k t]
    · have hsq : (offset k t) ^ 2 ≤ 1 / (width k : ℝ) := by
        have hle : offset k t ≤ 1 := by linarith [offset_small hk t]
        exact (show (offset k t)^2 ≤ offset k t by
          nlinarith [mul_nonneg (offset_nonneg k t) (sub_nonneg.mpr hle)]).trans (offset_le k t)
      simp only [site, if_neg h, add_sub_cancel_left]
      split_ifs <;> nlinarith [sq_nonneg (2 * (k : ℝ))]
  calc
    _ ≤ ∑ i : Fin k, ∑ t : Fin (core k + 1), (1 / (width k : ℝ) +
          (if t = s then (if i = z then (2 * (k : ℝ)) ^ 2 else 0) else 0)) :=
      Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun t _ => hpoint i t
    _ = (k : ℝ) + (2 * k) ^ 2 := by
      simp only [Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, if_true,
        Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      change (k : ℝ) * ((width k : ℝ) * (1 / (width k : ℝ))) + _ = _
      simp [hw]
    _ < (core k : ℝ) / 64 := by
      simp only [core, Nat.cast_pow, Nat.cast_mul, Nat.cast_ofNat]
      have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
      nlinarith [mul_nonneg (by positivity : (0 : ℝ) ≤ k) (show 0 ≤ (k : ℝ) - 2 by linarith)]

private theorem increasing_row {k : ℕ} (hk : 2 ≤ k) (i : Fin k) :
    StrictMono (site i) := by
  intro t s hts
  have hval : t.val < s.val := hts
  have ht : t.val ≠ core k := by have := s.isLt; omega
  simp only [site, ht, and_false, if_false]
  split_ifs with h
  · have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
    simp only [anchor, if_pos h.1, zero_add]
    linarith [offset_small hk t]
  · have hts' : (t.val : ℝ) < s.val := by exact_mod_cast hval
    have hw : 0 < (width k : ℝ) := by exact_mod_cast width_pos k
    exact add_lt_add_of_le_of_lt le_rfl (div_lt_div_of_pos_right hts' (sq_pos_of_pos hw))

/-- Even as coordinates (rather than record IDs), all stored vectors are distinct. -/
theorem distinct_sites {k : ℕ} (hk : 2 ≤ k) :
    Function.Injective (fun p : Fin k × Fin (core k + 1) => site p.1 p.2) := by
  rintro ⟨i,t⟩ ⟨j,s⟩ he
  change site i t = site j s at he
  have hij : i = j := by
    by_contra hn
    have h₁ := nearest_anchor_row hk i j t (anchor i) (anchor j) hn (by norm_num) (by norm_num)
    have h₂ := nearest_anchor_row hk j i s (anchor j) (anchor i) (Ne.symm hn) (by norm_num) (by norm_num)
    rw [he] at h₁
    exact h₁.asymm h₂
  subst j
  rw [(increasing_row hk i).injective he]

end IndexRecall.GapFamily
