import IndexRecall.GapFamily
import IndexRecall.FlatScan
import IndexRecall.OptimalBuckets
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.FieldSimp

/-!
# Probe ranks for the one-dimensional gap family

The query interval places the exceptional point closest to every query.
Its original bucket ranks last by centroid distance. Exchanging the two last
slots moves the point to the first-ranked bucket. Changing only the original
bucket's representative to a point near the tip also makes it rank first.
-/

noncomputable section

open scoped BigOperators

namespace IndexRecall.GapFamily

def firstRow {k : ℕ} (hk : 2 ≤ k) : Fin k := ⟨0, by omega⟩

def secondRow {k : ℕ} (hk : 2 ≤ k) : Fin k := ⟨1, by omega⟩

def lastSlot (k : ℕ) : Fin (core k + 1) := Fin.last (core k)

/-- Exchange the last slots of the first two rows. -/
def swapped {k : ℕ} (hk : 2 ≤ k) (i : Fin k) (t : Fin (core k + 1)) : ℝ :=
  if t = lastSlot k then site (Equiv.swap (firstRow hk) (secondRow hk) i) t
  else site i t

def slotExchange {k : ℕ} (hk : 2 ≤ k) : Equiv.Perm (Fin k × Fin (core k + 1)) :=
  Equiv.swap (firstRow hk, lastSlot k) (secondRow hk, lastSlot k)

/-- The alternative contains exactly the original records, via a permutation of slots. -/
theorem swapped_eq_site_exchange {k : ℕ} (hk : 2 ≤ k) (i : Fin k)
    (t : Fin (core k + 1)) :
    swapped hk i t = site (slotExchange hk (i, t)).1 (slotExchange hk (i, t)).2 := by
  by_cases ht : t = lastSlot k
  · subst t
    by_cases h₀ : i = firstRow hk
    · simp [swapped, slotExchange, h₀]
    by_cases h₁ : i = secondRow hk
    · simp [swapped, slotExchange, h₁]
    simp [swapped, slotExchange, Equiv.swap_apply_def, h₀, h₁]
  · simp [swapped, slotExchange, Equiv.swap_apply_def, ht]

abbrev mean {k : ℕ} (x : Fin k → Fin (core k + 1) → ℝ) (i : Fin k) : ℝ :=
  OptimalBuckets.rowMean x i

/-- Keep the original rows and change only the representative of the first row. -/
def rerouted {k : ℕ} (hk : 2 ≤ k) (i : Fin k) : ℝ :=
  if i = firstRow hk then 2 * k - 1 / 4 else mean site i

def QueryInRange (k : ℕ) (q : ℝ) : Prop :=
  15 * (k : ℝ) / 4 ≤ q ∧ q ≤ 31 * (k : ℝ) / 8

/-- One-based rank by squared distance when representative distances are distinct. -/
abbrev probeRank {k : ℕ} (r : Fin k → ℝ) (q : ℝ) :=
  FlatScan.rank (fun j => (q - r j)^2)

@[simp] theorem firstRow_val {k : ℕ} (hk : 2 ≤ k) : (firstRow hk).val = 0 := rfl

@[simp] theorem secondRow_val {k : ℕ} (hk : 2 ≤ k) : (secondRow hk).val = 1 := rfl

@[simp] theorem lastSlot_val (k : ℕ) : (lastSlot k).val = core k := rfl

private theorem firstRow_ne_secondRow {k : ℕ} (hk : 2 ≤ k) :
    firstRow hk ≠ secondRow hk := by
  intro h
  have h' := congrArg Fin.val h
  simp at h'

@[simp] theorem site_tip {k : ℕ} (hk : 2 ≤ k) :
    site (firstRow hk) (lastSlot k) = 2 * k := by
  simp [site]

@[simp] theorem swapped_tip {k : ℕ} (hk : 2 ≤ k) :
    swapped hk (secondRow hk) (lastSlot k) = 2 * k := by
  simp [swapped]

private theorem site_first_offLast {k : ℕ} (hk : 2 ≤ k)
    (t : Fin (core k + 1)) (ht : t ≠ lastSlot k) :
    site (firstRow hk) t = offset k t := by
  have ht' : t.val ≠ core k := fun h => ht (Fin.ext h)
  simp [site, anchor, ht']

private theorem site_right {k : ℕ} (i : Fin k) (hi : i.val ≠ 0)
    (t : Fin (core k + 1)) : site i t = 6 * (k : ℝ) + i.val + offset k t := by
  simp [site, anchor, hi]

private theorem width_cast_pos (k : ℕ) : (0 : ℝ) < width k := by
  exact_mod_cast width_pos k

private theorem width_large {k : ℕ} (hk : 2 ≤ k) : 64 * (k : ℝ) ≤ width k := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  push_cast [width, core]
  nlinarith [mul_nonneg (show 0 ≤ (k : ℝ) by positivity) (sub_nonneg.mpr hk')]

private def avgOffset (k : ℕ) : ℝ := (∑ t, offset k t) / (width k : ℝ)

private theorem avgOffset_bounds {k : ℕ} (hk : 2 ≤ k) :
    0 ≤ avgOffset k ∧ avgOffset k ≤ 1 / 8 := by
  unfold avgOffset
  refine ⟨div_nonneg (Finset.sum_nonneg (fun t _ => offset_nonneg k t))
    (width_cast_pos k).le, (div_le_iff₀ (width_cast_pos k)).2 ?_⟩
  calc
    _ ≤ ∑ _t : Fin (core k + 1), (1 / 8 : ℝ) :=
      Finset.sum_le_sum (fun t _ => offset_small hk t)
    _ = _ := by simp [width, mul_comm]

/-- The mean of a row with its last slot set to a given point. -/
private theorem mean_with_last {k : ℕ} (x : Fin k → Fin (core k + 1) → ℝ)
    (i : Fin k) (a z : ℝ)
    (h : ∀ t ≠ lastSlot k, x i t = a + offset k t)
    (hl : x i (lastSlot k) = z) :
    mean x i = a + avgOffset k + (z - (a + offset k (lastSlot k))) / (width k : ℝ) := by
  have hw : ((core k : ℕ) : ℝ) + 1 = (width k : ℝ) := by simp [width]
  rw [mean, OptimalBuckets.rowMean, hw]
  unfold avgOffset
  rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc]
  simp only [lastSlot] at hl ⊢
  rw [hl]
  have hsum : (∑ t : Fin (core k), x i t.castSucc) =
      ∑ t : Fin (core k), (a + offset k t.castSucc) := by
    apply Finset.sum_congr rfl
    intro t _
    apply h
    intro he
    have he' : t.val = core k := congrArg Fin.val he
    exact t.isLt.ne he'
  rw [hsum, Finset.sum_add_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp [(width_cast_pos k).ne']
  push_cast [width]
  ring

private theorem mean_first {k : ℕ} (hk : 2 ≤ k) :
    mean site (firstRow hk) =
      avgOffset k + (2 * k - offset k (lastSlot k)) / (width k : ℝ) := by
  simpa using mean_with_last site (firstRow hk) 0 (2 * k)
    (fun t ht => by simpa using site_first_offLast hk t ht) (site_tip hk)

private theorem mean_right {k : ℕ} (i : Fin k) (hi : i.val ≠ 0) :
    mean site i = 6 * (k : ℝ) + i.val + avgOffset k := by
  simpa using mean_with_last site i (6 * (k : ℝ) + i.val)
    (6 * (k : ℝ) + i.val + offset k (lastSlot k))
    (fun t _ => site_right i hi t) (site_right i hi _)

private theorem mean_swapped_first {k : ℕ} (hk : 2 ≤ k) :
    mean (swapped hk) (firstRow hk) = avgOffset k + (6 * (k : ℝ) + 1) / width k := by
  have hl : swapped hk (firstRow hk) (lastSlot k) = 6 * (k : ℝ) + 1 + offset k (lastSlot k) := by
    simpa [swapped] using site_right (secondRow hk) (by simp) (lastSlot k)
  have he := mean_with_last (swapped hk) (firstRow hk) 0
    (6 * (k : ℝ) + 1 + offset k (lastSlot k))
    (fun t ht => by simp [swapped, ht, site_first_offLast hk t ht]) hl
  simpa only [zero_add, add_sub_cancel_right] using he

private theorem mean_swapped_second {k : ℕ} (hk : 2 ≤ k) :
    mean (swapped hk) (secondRow hk) =
      6 * (k : ℝ) + 1 + avgOffset k -
        (4 * (k : ℝ) + 1 + offset k (lastSlot k)) / width k := by
  have he := mean_with_last (swapped hk) (secondRow hk) (6 * (k : ℝ) + 1)
    (2 * k) (fun t ht => by
      simpa [swapped, ht] using site_right (secondRow hk) (by simp) t) (swapped_tip hk)
  rw [he]
  ring

private theorem mean_swapped_other {k : ℕ} (hk : 2 ≤ k)
    (i : Fin k) (hi : i ≠ firstRow hk) (hj : i ≠ secondRow hk) :
    mean (swapped hk) i = mean site i := by
  simp [mean, OptimalBuckets.rowMean, swapped, Equiv.swap_apply_of_ne_of_ne hi hj]

private theorem base_left_upper {k : ℕ} (hk : 2 ≤ k) :
    mean site (firstRow hk) ≤ 1 / 4 := by
  rw [mean_first hk]
  have h : (2 * (k : ℝ) - offset k (lastSlot k)) / width k ≤ 1 / 8 := by
    apply (div_le_iff₀ (width_cast_pos k)).2
    have hk' : (0 : ℝ) ≤ k := by positivity
    linarith [width_large hk, offset_nonneg k (lastSlot k)]
  linarith [(avgOffset_bounds hk).2]

private theorem swap_left_upper {k : ℕ} (hk : 2 ≤ k) :
    mean (swapped hk) (firstRow hk) ≤ 1 / 4 := by
  rw [mean_swapped_first hk]
  have h : (6 * (k : ℝ) + 1) / width k ≤ 1 / 8 := by
    apply (div_le_iff₀ (width_cast_pos k)).2
    have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
    linarith [width_large hk]
  linarith [(avgOffset_bounds hk).2]

private theorem swap_second_bounds {k : ℕ} (hk : 2 ≤ k) :
    6 * (k : ℝ) ≤ mean (swapped hk) (secondRow hk) ∧
      mean (swapped hk) (secondRow hk) ≤ 6 * (k : ℝ) + 1 + avgOffset k := by
  rw [mean_swapped_second hk]
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hnonneg : 0 ≤ (4 * (k : ℝ) + 1 + offset k (lastSlot k)) / width k := by
    apply div_nonneg _ (width_cast_pos k).le
    linarith [offset_nonneg k (lastSlot k)]
  have hupper : (4 * (k : ℝ) + 1 + offset k (lastSlot k)) / width k ≤ 1 / 8 := by
    apply (div_le_iff₀ (width_cast_pos k)).2
    linarith [width_large hk, offset_small hk (lastSlot k)]
  constructor <;> linarith [(avgOffset_bounds hk).1]

private theorem rightBounds {k : ℕ} (hk : 2 ≤ k) (i : Fin k) (hi : i ≠ firstRow hk) :
    6 * (k : ℝ) ≤ mean site i ∧ mean site i ≤ 7 * (k : ℝ) := by
  have hi' : i.val ≠ 0 := fun h => hi (Fin.ext h)
  have hik : (i.val : ℝ) + 1 ≤ k := by exact_mod_cast Nat.succ_le_of_lt i.isLt
  have hip : (0 : ℝ) ≤ i.val := by positivity
  rw [mean_right i hi']
  obtain ⟨hl, hu⟩ := avgOffset_bounds hk
  constructor <;> linarith

private theorem swappedRightBounds {k : ℕ} (hk : 2 ≤ k) (i : Fin k)
    (hi : i ≠ firstRow hk) :
    6 * (k : ℝ) ≤ mean (swapped hk) i ∧ mean (swapped hk) i ≤ 7 * (k : ℝ) := by
  by_cases hj : i = secondRow hk
  · subst i
    obtain ⟨hl, hu⟩ := swap_second_bounds hk
    have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
    exact ⟨hl, by linarith [(avgOffset_bounds hk).2]⟩
  · rw [mean_swapped_other hk i hi hj]
    exact rightBounds hk i hi

/-- The exceptional point is the unique ground-truth result for every query in the interval. -/
theorem uniqueNearest {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q)
    (i : Fin k) (t : Fin (core k + 1))
    (h : (i, t) ≠ (firstRow hk, lastSlot k)) :
    (q - 2 * (k : ℝ)) ^ 2 < (q - site i t) ^ 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  obtain ⟨hql, hqu⟩ := hq
  have hdist : 0 ≤ q - 2 * (k : ℝ) := by linarith
  by_cases hi : i = firstRow hk
  · subst i
    have ht : t ≠ lastSlot k := fun he => h (by rw [he])
    rw [site_first_offLast hk t ht]
    have ho := offset_small hk t
    exact (sq_lt_sq₀ hdist (by linarith)).2 (by linarith)
  · have hi' : i.val ≠ 0 := fun he => hi (Fin.ext he)
    rw [site_right i hi', ← neg_sq (q - (6 * (k : ℝ) + i.val + offset k t))]
    have hip : (0 : ℝ) ≤ i.val := by positivity
    have ho := offset_nonneg k t
    exact (sq_lt_sq₀ hdist (by linarith)).2 (by linarith)

/-- After the exchange, the receiving bucket's last slot is uniquely nearest. -/
theorem swapped_uniqueNearest {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q)
    (i : Fin k) (t : Fin (core k + 1))
    (h : (i, t) ≠ (secondRow hk, lastSlot k)) :
    (q - 2 * (k : ℝ)) ^ 2 < (q - swapped hk i t) ^ 2 := by
  rw [swapped_eq_site_exchange]
  apply uniqueNearest hk hq
  intro he
  apply h
  apply (slotExchange hk).injective
  simpa [slotExchange] using he

private theorem right_dist_lt_left {k : ℕ} (hk : 2 ≤ k) {q x y : ℝ}
    (hq : QueryInRange k q) (hx : 6 * (k : ℝ) ≤ x ∧ x ≤ 7 * (k : ℝ)) (hy : y ≤ 1 / 4) :
    (q - x) ^ 2 < (q - y) ^ 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  obtain ⟨hql, hqu⟩ := hq
  obtain ⟨hxl, hxu⟩ := hx
  rw [← neg_sq (q - x)]
  exact (sq_lt_sq₀ (by linarith) (by linarith)).2 (by linarith)

/-- The original exceptional point's bucket is strictly farther than every other bucket. -/
theorem baseline_last {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q)
    (i : Fin k) (hi : i ≠ firstRow hk) :
    (q - mean site i) ^ 2 < (q - mean site (firstRow hk)) ^ 2 :=
  right_dist_lt_left hk hq (rightBounds hk i hi) (base_left_upper hk)

private theorem right_dist_mono {k : ℕ} (hk : 2 ≤ k) {q x y : ℝ}
    (hq : QueryInRange k q) (hx : 6 * (k : ℝ) ≤ x) (hxy : x < y) :
    (q - x) ^ 2 < (q - y) ^ 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  have hqu := hq.2
  rw [← neg_sq (q - x), ← neg_sq (q - y)]
  exact (sq_lt_sq₀ (by linarith) (by linarith)).2 (by linarith)

private theorem baseline_right_mono {k : ℕ} (hk : 2 ≤ k) {q : ℝ}
    (hq : QueryInRange k q) {i j : Fin k} (hi : i ≠ firstRow hk)
    (hij : i < j) : (q - mean site i) ^ 2 < (q - mean site j) ^ 2 := by
  have hi' : i.val ≠ 0 := fun he => hi (Fin.ext he)
  have hj' : j.val ≠ 0 := by
    have : i.val < j.val := hij
    omega
  apply right_dist_mono hk hq (rightBounds hk i hi).1
  rw [mean_right i hi', mean_right j hj']
  have h : (i.val : ℝ) < j.val := by exact_mod_cast (show i.val < j.val from hij)
  linarith

private theorem swapped_right_mono {k : ℕ} (hk : 2 ≤ k) {q : ℝ}
    (hq : QueryInRange k q) {i j : Fin k} (hi : i ≠ firstRow hk)
    (hij : i < j) :
    (q - mean (swapped hk) i) ^ 2 < (q - mean (swapped hk) j) ^ 2 := by
  have hi' : i.val ≠ 0 := fun he => hi (Fin.ext he)
  have hv : i.val < j.val := hij
  have hj : j ≠ firstRow hk := fun he => by simp [he] at hv
  have hj' : j.val ≠ 0 := fun he => hj (Fin.ext he)
  have h₂ : j ≠ secondRow hk := by
    intro he
    rw [he] at hv
    simp only [secondRow_val] at hv
    omega
  have hcast : (i.val : ℝ) < j.val := by exact_mod_cast hv
  apply right_dist_mono hk hq (swappedRightBounds hk i hi).1
  rw [mean_swapped_other hk j hj h₂, mean_right j hj']
  by_cases hs : i = secondRow hk
  · subst i
    have hu := (swap_second_bounds hk).2
    simp only [secondRow_val, Nat.cast_one] at hcast
    linarith
  · rw [mean_swapped_other hk i hi hs, mean_right i hi']
    linarith

/-- The new centroid of the receiving bucket is strictly closest. -/
theorem swapped_first {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q)
    (i : Fin k) (hi : i ≠ secondRow hk) :
    (q - mean (swapped hk) (secondRow hk)) ^ 2 < (q - mean (swapped hk) i) ^ 2 := by
  have hsecond := firstRow_ne_secondRow hk
  by_cases hf : i = firstRow hk
  · subst i
    exact right_dist_lt_left hk hq (swappedRightBounds hk _ hsecond.symm) (swap_left_upper hk)
  · apply swapped_right_mono hk hq hsecond.symm
    have hv : i.val ≠ 0 := fun he => hf (Fin.ext he)
    have hv' : i.val ≠ 1 := fun he => hi (Fin.ext he)
    change 1 < i.val
    omega

/-- The original bucket becomes strictly closest when its representative is a quarter unit to the left of the tip. -/
theorem rerouted_first {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q)
    (i : Fin k) (hi : i ≠ firstRow hk) :
    (q - rerouted hk (firstRow hk)) ^ 2 < (q - rerouted hk i) ^ 2 := by
  have hk' : (2 : ℝ) ≤ k := by exact_mod_cast hk
  obtain ⟨hql, hqu⟩ := hq
  have hb := (rightBounds hk i hi).1
  simp [rerouted, hi]
  rw [← neg_sq (q - mean site i)]
  exact (sq_lt_sq₀ (by linarith) (by linarith)).2 (by linarith)

private theorem probeRank_last {k : ℕ} {r : Fin k → ℝ} {q : ℝ} {i : Fin k}
    (h : ∀ j ≠ i, (q - r j) ^ 2 < (q - r i) ^ 2) :
    probeRank r q i = k := by
  classical
  have hf : Finset.univ.filter (fun j => (q - r j) ^ 2 < (q - r i) ^ 2) =
      Finset.univ.erase i := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, true_and, and_true]
    exact ⟨fun hlt he => by subst j; exact (lt_irrefl _ hlt), h j⟩
  unfold probeRank FlatScan.rank
  rw [hf, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin]
  have hi := i.isLt
  omega

theorem baseline_rank {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    probeRank (mean site) q (firstRow hk) = k :=
  probeRank_last (baseline_last hk hq)

theorem swapped_rank {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    probeRank (mean (swapped hk)) q (secondRow hk) = 1 :=
  FlatScan.rank_first (swapped_first hk hq)

theorem rerouted_rank {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    probeRank (rerouted hk) q (firstRow hk) = 1 :=
  FlatScan.rank_first (rerouted_first hk hq)

private theorem strict_between_rows {k : ℕ} (hk : 2 ≤ k) {d : Fin k → ℝ}
    (rightMono : ∀ {i j}, i ≠ firstRow hk → i < j → d i < d j)
    (leftOrder : ∀ i ≠ firstRow hk, d i ≠ d (firstRow hk)) :
    Function.Injective d := by
  intro i j he
  by_cases hi : i = firstRow hk
  · by_contra hj
    rw [hi] at he
    exact leftOrder j (fun hj' => hj (hi.trans hj'.symm)) he.symm
  by_cases hj : j = firstRow hk
  · rw [hj] at he
    exact False.elim (leftOrder i hi he)
  rcases lt_trichotomy i j with h | h | h
  · exact False.elim ((rightMono hi h).ne he)
  · exact h
  · exact False.elim ((rightMono hj h).ne he.symm)

theorem baseline_untied {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    Function.Injective (fun i => (q - mean site i) ^ 2 : Fin k → ℝ) :=
  strict_between_rows hk (baseline_right_mono hk hq) (fun i hi => (baseline_last hk hq i hi).ne)

theorem swapped_untied {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    Function.Injective (fun i => (q - mean (swapped hk) i) ^ 2) :=
  strict_between_rows hk (swapped_right_mono hk hq)
    (fun i hi => (right_dist_lt_left hk hq (swappedRightBounds hk i hi) (swap_left_upper hk)).ne)

theorem rerouted_untied {k : ℕ} (hk : 2 ≤ k) {q : ℝ} (hq : QueryInRange k q) :
    Function.Injective (fun i => (q - rerouted hk i) ^ 2) := by
  apply strict_between_rows hk _ (fun i hi => (rerouted_first hk hq i hi).ne')
  intro i j hi hij
  have hj : j ≠ firstRow hk := by
    intro he
    have hv : i.val < j.val := hij
    simp [he] at hv
  simpa [rerouted, hi, hj] using baseline_right_mono hk hq hi hij

end IndexRecall.GapFamily
