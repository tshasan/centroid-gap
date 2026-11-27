import IndexRecall.PlanarMeans
import IndexRecall.FlatScan

/-! # Probe ranks for the paired planar rows -/

noncomputable section

namespace IndexRecall.PlanarGap

private theorem pitch_large {s : ℕ} (hs : 2 ≤ s) : 40 ≤ pitch s := by
  unfold pitch
  linarith [size_four hs]

private theorem height_large {s : ℕ} (hs : 2 ≤ s) : 4 ≤ height s := by
  have h : 4 * pitch s ≤ 4 * pitch s * (size s : ℝ) := by
    simpa using mul_le_mul_of_nonneg_left
      (show (1 : ℝ) ≤ size s by linarith [size_four hs])
      (show 0 ≤ 4 * pitch s by linarith [pitch_large hs])
  unfold height
  rw [show (4 : ℝ) = 2^2 by norm_num]
  exact (sq_le_sq₀ (by norm_num) (by linarith [pitch_large hs])).2 (by linarith [pitch_large hs])

private theorem inv_size_le {s : ℕ} (hs : 2 ≤ s) : 1 / (size s : ℝ) ≤ 1 / 4 :=
  (div_le_iff₀ (by linarith [size_four hs])).2 (by linarith [size_four hs])

private theorem pitch_div_bounds {s : ℕ} (hs : 2 ≤ s) :
    0 ≤ pitch s / (size s : ℝ) ∧ pitch s / (size s : ℝ) ≤ pitch s / 4 := by
  have hp : 0 ≤ pitch s := by linarith [pitch_large hs]
  have hk : (0 : ℝ) < size s := by linarith [size_four hs]
  refine ⟨div_nonneg hp hk.le, ?_⟩
  simpa [div_eq_mul_inv, mul_assoc] using
    mul_le_mul_of_nonneg_left (inv_size_le hs) hp

private def XBound (s : ℕ) (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * pitch s * (size s : ℝ)

private theorem bounded_shift {s : ℕ} (hs : 2 ≤ s) (j : Fin s) {e : ℝ}
    (he : 0 ≤ e) (he' : e ≤ 2 * pitch s) :
    XBound s (origin j + e) := by
  have hp : 0 ≤ pitch s := by linarith [pitch_large hs]
  have hj : (j.val : ℝ) + 1 ≤ s := by exact_mod_cast Nat.succ_le_of_lt j.isLt
  have hj₀ : 0 ≤ (j.val : ℝ) := by positivity
  have hm := mul_le_mul_of_nonneg_right hj hp
  have h₀ := mul_nonneg hj₀ hp
  have h₁ := mul_nonneg (show 0 ≤ (s : ℝ) by positivity) hp
  constructor
  · unfold origin
    nlinarith
  · simp only [origin, size, Nat.cast_mul, Nat.cast_ofNat]
    nlinarith

private theorem query_x_bound {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    XBound s (query i).1 := by
  simp only [query, add_assoc]
  have h := drift_bounds hs
  have hu := h.2.trans (inv_size_le hs)
  exact bounded_shift hs i (by linarith [h.1, pitch_large hs]) (by linarith [pitch_large hs])

private theorem horizontal_bound {s : ℕ} (hs : 2 ≤ s) {x y : ℝ}
    (hx : XBound s x) (hy : XBound s y) : (x - y)^2 ≤ height s / 4 := by
  have hb : 0 ≤ 2 * pitch s * (size s : ℝ) :=
    mul_nonneg (by linarith [pitch_large hs]) (by positivity)
  have hmin : -(2 * pitch s * (size s : ℝ)) ≤ x - y := by linarith [hx.1, hy.2]
  have hmax : x - y ≤ 2 * pitch s * (size s : ℝ) := by linarith [hx.2, hy.1]
  calc
    _ ≤ (2 * pitch s * (size s : ℝ))^2 := by
      rw [sq_le_sq]
      exact abs_le.mpr ⟨hmin, hmax⟩ |>.trans (le_abs_self _)
    _ = _ := by unfold height; ring

private theorem closer_of_higher {s : ℕ} (hs : 2 ≤ s) (i : Fin s) {p r : Point}
    (hp : XBound s p.1) (hp₂ : p.2 ≤ 2)
    (hgap : r.2 + 1 / 2 ≤ p.2) :
    sqDist (query i) p < sqDist (query i) r := by
  have hg := height_large hs
  have hr₂ : r.2 ≤ 2 := by linarith
  have h₀ : 0 ≤ 2 * height s - p.2 - r.2 := by linarith
  have hv := mul_le_mul_of_nonneg_right (show 1 / 2 ≤ p.2 - r.2 by linarith) h₀
  have hh := horizontal_bound hs (query_x_bound hs i) hp
  have hh₀ := sq_nonneg ((query i).1 - r.1)
  simp only [query] at hh hh₀
  simp only [sqDist, query]
  nlinarith

private def row {s : ℕ} (e y : ℝ) (j : Fin s) : Point := (origin j + e, y)

private theorem origin_step {s : ℕ} (hs : 2 ≤ s) {i j : Fin s} (h : i < j) :
    origin i + 3 * pitch s ≤ origin j := by
  have hij : (i.val : ℝ) + 1 ≤ j.val := by
    exact_mod_cast (show i.val + 1 ≤ j.val from h)
  have hm := mul_le_mul_of_nonneg_right hij (show 0 ≤ pitch s by linarith [pitch_large hs])
  unfold origin
  linarith

private theorem row_own_first {s : ℕ} (hs : 2 ≤ s) (i : Fin s) {e y : ℝ}
    (he : -pitch s < (query i).1 - (origin i + e) ∧
      (query i).1 - (origin i + e) < pitch s) (j : Fin s) (hj : j ≠ i) :
    sqDist (query i) (row e y i) < sqDist (query i) (row e y j) := by
  simp only [sqDist, row, add_lt_add_iff_right]
  rcases lt_or_gt_of_ne hj with hij | hij
  · have hg := origin_step hs hij
    exact sq_lt_sq' (by linarith [he.1, pitch_large hs]) (by linarith [he.2, pitch_large hs])
  · have hg := origin_step hs hij
    rw [← neg_sq ((query i).1 - (origin j + e))]
    exact sq_lt_sq' (by linarith [he.1, pitch_large hs]) (by linarith [he.2, pitch_large hs])

private theorem row_untied {s : ℕ} (hs : 2 ≤ s) (i : Fin s) {e y : ℝ}
    (he : -pitch s < (query i).1 - (origin i + e) ∧
      (query i).1 - (origin i + e) < pitch s)
    (hne : (query i).1 - (origin i + e) ≠ 0) :
    Function.Injective (fun j : Fin s => sqDist (query i) (row e y j)) := by
  intro j l hh
  have hp := (show (0 : ℝ) < 40 by norm_num).trans_le (pitch_large hs)
  simp only [sqDist, row, add_left_inj, sq_eq_sq_iff_eq_or_eq_neg] at hh
  rcases hh with h | h
  · have h₁ : 3 * (j.val : ℝ) = 3 * (l.val : ℝ) :=
      mul_right_cancel₀ hp.ne' (by simp only [origin] at h; linarith)
    have h₂ : (j.val : ℝ) = l.val := by linarith
    exact Fin.ext (by exact_mod_cast h₂)
  · have h₁ : 3 * pitch s * ((j.val : ℝ) + l.val - 2 * i.val) =
        2 * ((query i).1 - (origin i + e)) := by
      simp only [origin] at h ⊢
      linarith
    have hn : j.val + l.val = 2 * i.val := by
      rcases lt_trichotomy (j.val + l.val) (2 * i.val) with hlt | heq | hgt
      · have hcast : (j.val : ℝ) + l.val + 1 ≤ 2 * i.val := by
          exact_mod_cast Nat.succ_le_of_lt hlt
        have hmul := mul_le_mul_of_nonneg_left hcast (show 0 ≤ 3 * pitch s by positivity)
        linarith [he.1, he.2]
      · exact heq
      · have hcast : 2 * (i.val : ℝ) + 1 ≤ (j.val : ℝ) + l.val := by
          exact_mod_cast Nat.succ_le_of_lt hgt
        have hmul := mul_le_mul_of_nonneg_left hcast (show 0 ≤ 3 * pitch s by positivity)
        linarith [he.1, he.2]
    exfalso
    apply hne
    have hcast : (j.val : ℝ) + l.val = 2 * i.val := by exact_mod_cast hn
    rw [hcast, sub_self, mul_zero] at h₁
    linarith

private theorem paired_lt {s : ℕ} (hs : 2 ≤ s) (i : Fin s) {e f y z : ℝ}
    (he : 0 ≤ e ∧ e ≤ 2 * pitch s)
    (hy : y ≤ 2) (hz : z + 1 / 2 ≤ y) (j l : Fin s) :
    sqDist (query i) (row e y j) < sqDist (query i) (row f z l) :=
  closer_of_higher hs i (bounded_shift hs j he.1 he.2) hy hz

def firstSlot {s : ℕ} (hs : 2 ≤ s) : Slot s := ⟨0, by unfold size; omega⟩

@[simp] theorem firstSlot_val {s : ℕ} (hs : 2 ≤ s) : (firstSlot hs).val = 0 := rfl

@[simp] theorem site_tip {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    site (i, false) (firstSlot hs) = tip i := by
  simp [site, baseX, delta, tip]

@[simp] theorem swapped_tip {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    swapped (i, true) (firstSlot hs) = tip i := by
  simp [swapped, exchange]

/-- Among all stored points, the tip in the query's pair is uniquely nearest. -/
theorem uniqueNearest {s : ℕ} (hs : 2 ≤ s) (i : Fin s) (l : Label s) (t : Slot s)
    (h : (l, t) ≠ ((i, false), firstSlot hs)) :
    sqDist (query i) (tip i) < sqDist (query i) (site l t) := by
  obtain ⟨j, d⟩ := l
  have htip : XBound s (tip i).1 := by
    simpa [tip] using bounded_shift hs i (e := 0) (by norm_num) (by linarith [pitch_large hs])
  cases d
  · by_cases ht : t.val = 0
    · have he : t = firstSlot hs := Fin.ext ht
      subst t
      simp only [site_tip]
      have hj : j ≠ i := by
        intro he
        subst j
        exact h rfl
      have hd := drift_bounds hs
      have hu := hd.2.trans (inv_size_le hs)
      simpa [tip, row] using row_own_first hs i (e := 0) (y := 2)
        (by simp only [query, add_zero]; constructor <;> linarith [hd.1, pitch_large hs])
        j hj
    · apply closer_of_higher hs i htip (by simp [tip])
      norm_num [site, ht, tip]
  · apply closer_of_higher hs i htip (by simp [tip])
    norm_num [site, tip]

/-- The global nearest point moves to the higher row of the query's pair. -/
theorem swapped_uniqueNearest {s : ℕ} (hs : 2 ≤ s) (i : Fin s) (l : Label s) (t : Slot s)
    (h : (l, t) ≠ ((i, true), firstSlot hs)) :
    sqDist (query i) (tip i) < sqDist (query i) (swapped l t) := by
  unfold swapped
  apply uniqueNearest hs
  intro he
  apply h
  exact exchange.injective (by simpa [exchange] using he)

abbrev probeRank {s : ℕ} (r : Label s → Point) (q : Point) :=
  FlatScan.rank (fun j => sqDist q (r j))

/-- The front type entirely precedes the back type. Within either type the
query's own pair comes first, and all distances are distinct. -/
private structure RowOrder {s : ℕ} (r : Label s → Point) (i : Fin s) (front : Bool) : Prop where
  cross : ∀ j l, sqDist (query i) (r (j, front)) < sqDist (query i) (r (l, !front))
  front_first : ∀ j ≠ i, sqDist (query i) (r (i, front)) < sqDist (query i) (r (j, front))
  back_first : ∀ j ≠ i, sqDist (query i) (r (i, !front)) < sqDist (query i) (r (j, !front))
  untied : Function.Injective (fun l : Label s => sqDist (query i) (r l))

private theorem query_shift {s : ℕ} (i : Fin s) (e : ℝ) :
    (query i).1 - (origin i + e) = drift s + pitch s / 3 - e := by
  change origin i + drift s + pitch s / 3 - (origin i + e) = _
  ring

private theorem Bool.eq_or_eq_not (b c : Bool) : b = c ∨ b = !c := by
  cases b <;> cases c <;> decide

private theorem order_of_rows {s : ℕ} (hs : 2 ≤ s) (i : Fin s)
    (r : Label s → Point) (a : Bool) {e f y z : ℝ}
    (hf : ∀ j, r (j, a) = row e y j) (hb : ∀ j, r (j, !a) = row f z j)
    (he : 0 ≤ e ∧ e ≤ 2 * pitch s) (hy : y ≤ 2) (hyz : z + 1 / 2 ≤ y)
    (hfront : -pitch s < drift s + pitch s / 3 - e ∧
      drift s + pitch s / 3 - e < pitch s ∧ drift s + pitch s / 3 - e ≠ 0)
    (hback : -pitch s < drift s + pitch s / 3 - f ∧
      drift s + pitch s / 3 - f < pitch s ∧ drift s + pitch s / 3 - f ≠ 0) :
    RowOrder r i a := by
  have hc (j l : Fin s) := paired_lt hs i (f := f) he hy hyz j l
  have hfront' : -pitch s < (query i).1 - (origin i + e) ∧
      (query i).1 - (origin i + e) < pitch s :=
    by rw [query_shift]; exact ⟨hfront.1, hfront.2.1⟩
  have hback' : -pitch s < (query i).1 - (origin i + f) ∧
      (query i).1 - (origin i + f) < pitch s :=
    by rw [query_shift]; exact ⟨hback.1, hback.2.1⟩
  have hinj_f := row_untied hs i (y := y) hfront' (by rw [query_shift]; exact hfront.2.2)
  have hinj_b := row_untied hs i (y := z) hback' (by rw [query_shift]; exact hback.2.2)
  refine ⟨fun j l => by rw [hf, hb]; exact hc j l,
    fun j h => by rw [hf, hf]; exact row_own_first hs i hfront' j h,
    fun j h => by rw [hb, hb]; exact row_own_first hs i hback' j h, ?_⟩
  intro ⟨j, b⟩ ⟨l, c⟩ h
  dsimp only at h
  rcases Bool.eq_or_eq_not b a with hd | hd <;> rcases Bool.eq_or_eq_not c a with hd' | hd' <;>
    subst b <;> subst c
  · rw [hf, hf] at h
    rw [hinj_f h]
  · rw [hf, hb] at h
    exact False.elim ((hc j l).ne h)
  · rw [hb, hf] at h
    exact False.elim ((hc l j).ne h.symm)
  · rw [hb, hb] at h
    rw [hinj_b h]

private theorem baseline_order {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    RowOrder (mean site) i true := by
  have hp := pitch_large hs
  have hd := drift_bounds hs
  have hd' := hd.2.trans (inv_size_le hs)
  have hk : 2 / (size s : ℝ) ≤ 1 / 2 := by
    apply (div_le_iff₀ (by linarith [size_four hs])).2
    linarith [size_four hs]
  apply order_of_rows hs i (mean site) true
    (e := pitch s + drift s) (f := drift s) (y := 1) (z := 2 / (size s : ℝ))
  · intro j
    simp [row, mean_D hs, add_assoc]
  · intro j
    simp [row, mean_A hs]
  · constructor <;> linarith [hd.1]
  · norm_num
  · linarith
  · refine ⟨?_, ?_, ?_⟩ <;> linarith
  · refine ⟨?_, ?_, ?_⟩ <;> linarith

private theorem swapped_order {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    RowOrder (mean swapped) i true := by
  have hp := pitch_large hs
  have hd := drift_bounds hs
  have hd' := hd.2.trans (inv_size_le hs)
  have hdiv := pitch_div_bounds hs
  apply order_of_rows hs i (mean swapped) true
    (e := pitch s + drift s - pitch s / (size s : ℝ))
    (f := drift s + pitch s / (size s : ℝ))
    (y := 1 + 1 / (size s : ℝ)) (z := 1 / (size s : ℝ))
  · intro j
    rw [mean_swap_D hs]
    congr 1
    ring
  · intro j
    simp [row, mean_swap_A hs, add_assoc]
  · constructor <;> linarith [hd.1, hdiv.1, hdiv.2]
  · linarith [inv_size_le hs]
  · linarith
  · refine ⟨?_, ?_, ?_⟩ <;> linarith [hdiv.1, hdiv.2]
  · refine ⟨?_, ?_, ?_⟩ <;> linarith [hdiv.1, hdiv.2]

private theorem rerouted_order {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    RowOrder rerouted i false := by
  have hp := pitch_large hs
  have hd := drift_bounds hs
  have hd' := hd.2.trans (inv_size_le hs)
  apply order_of_rows hs i rerouted false (e := 0) (f := pitch s + drift s) (y := 3 / 2) (z := 1)
  · intro j
    simp [rerouted, row]
  · intro j
    simp [rerouted, mean_D hs, row, add_assoc]
  · constructor <;> linarith
  · norm_num
  · norm_num
  · refine ⟨?_, ?_, ?_⟩ <;> linarith [hd.1]
  · refine ⟨?_, ?_, ?_⟩ <;> linarith

private theorem front_is_first {s : ℕ} {r : Label s → Point} {i : Fin s} {a : Bool}
    (h : RowOrder r i a) (l : Label s) (hl : l ≠ (i, a)) :
    sqDist (query i) (r (i, a)) < sqDist (query i) (r l) := by
  obtain ⟨j, b⟩ := l
  rcases Bool.eq_or_eq_not b a with he | he
  · subst b
    exact h.front_first j (by intro hj; subst j; exact hl rfl)
  · subst b
    exact h.cross i j

theorem swapped_first {s : ℕ} (hs : 2 ≤ s) (i : Fin s) (l : Label s) (hl : l ≠ (i, true)) :
    sqDist (query i) (mean swapped (i, true)) < sqDist (query i) (mean swapped l) :=
  front_is_first (swapped_order hs i) l hl

theorem rerouted_first {s : ℕ} (hs : 2 ≤ s) (i : Fin s) (l : Label s) (hl : l ≠ (i, false)) :
    sqDist (query i) (rerouted (i, false)) < sqDist (query i) (rerouted l) :=
  front_is_first (rerouted_order hs i) l hl

/-- All higher rows precede the original tip's row, which comes first among lower rows. -/
theorem baseline_rank {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    probeRank (mean site) (query i) (i, false) = s + 1 := by
  have h := baseline_order hs i
  let e : Fin s ↪ Label s := ⟨fun j => (j, true), fun _ _ he => congrArg Prod.fst he⟩
  have hf : (Finset.univ.filter (fun j => sqDist (query i) (mean site j) <
        sqDist (query i) (mean site (i, false)))) = Finset.univ.map e := by
    ext ⟨j, b⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map]
    cases b
    · constructor
      · intro hj
        have hji : j ≠ i := fun he => by subst j; exact lt_irrefl _ hj
        exact False.elim ((h.back_first j hji).asymm hj)
      · rintro ⟨l, _, he⟩
    · exact ⟨fun _ => ⟨j, rfl⟩, fun _ => h.cross j i⟩
  rw [probeRank, FlatScan.rank, hf, Finset.card_map, Finset.card_univ, Fintype.card_fin, Nat.add_comm]

theorem swapped_rank {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    probeRank (mean swapped) (query i) (i, true) = 1 :=
  FlatScan.rank_first (swapped_first hs i)

theorem rerouted_rank {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    probeRank rerouted (query i) (i, false) = 1 :=
  FlatScan.rank_first (rerouted_first hs i)

theorem baseline_untied {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    Function.Injective (fun l : Label s => sqDist (query i) (mean site l)) :=
  (baseline_order hs i).untied

theorem swapped_untied {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    Function.Injective (fun l : Label s => sqDist (query i) (mean swapped l)) :=
  (swapped_order hs i).untied

theorem rerouted_untied {s : ℕ} (hs : 2 ≤ s) (i : Fin s) :
    Function.Injective (fun l : Label s => sqDist (query i) (rerouted l)) :=
  (rerouted_order hs i).untied

end IndexRecall.PlanarGap
