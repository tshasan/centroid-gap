import IndexRecall.EMeansUpdate
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.Perm

/-! # Counting document seeds sampled uniformly without replacement -/

noncomputable section
open scoped Classical

namespace IndexRecall.SphericalGap.EMeans

variable {k : ℕ}

/-- The closed form for one ordinary target seed and `k - 1` foreign seeds. -/
def seedProbability (k : ℕ) : ℝ :=
  (k - 1 : ℕ) * ((k ^ 2 - k).choose (k - 1) : ℝ) / ((k ^ 2).choose k : ℝ)

/-- The sample space before labeling the centers. Each size-`k` subset of document
addresses has the same probability when the addresses are drawn without replacement. -/
def seedSamples (k : ℕ) : Finset (Finset (Address k)) :=
  Finset.univ.powersetCard k

@[simp] theorem mem_seedSamples {s : Finset (Address k)} :
    s ∈ seedSamples k ↔ s.card = k := by
  simp [seedSamples]

theorem seedSamples_card (k : ℕ) : (seedSamples k).card = (k ^ 2).choose k := by
  simp [seedSamples, Address, sq]

/-- Exactly one selected address is in row `z`, and its column differs from `z`,
so this seed is ordinary. The unique winning document has address `(z,z)`. -/
def SeedEvent (z : Fin k) (s : Finset (Address k)) : Prop :=
  ∃ j : Fin k, j ≠ z ∧ (z, j) ∈ s ∧ ∀ p ∈ s, p.1 = z → p = (z, j)

/-- Probability of an event under the uniform law on the unordered seed sets. -/
def uniformSeedProbability (k : ℕ) (P : Finset (Address k) → Prop) : ℝ :=
  (((seedSamples k).filter P).card : ℝ) / (seedSamples k).card

/-- Counting ordered draws gives the same probability. Each admissible set of IDs
has exactly `k!` bijections from the labels, in the numerator and denominator alike. -/
theorem uniformSeedProbability_eq_ordered (k : ℕ) (P : Finset (Address k) → Prop) :
    uniformSeedProbability k P =
      ((∑ s ∈ (seedSamples k).filter P, Fintype.card (Fin k ≃ ↥s) : ℕ) : ℝ) /
      ((∑ s ∈ seedSamples k, Fintype.card (Fin k ≃ ↥s) : ℕ) : ℝ) := by
  have count (A : Finset (Finset (Address k))) (hA : A ⊆ seedSamples k) :
      (∑ s ∈ A, Fintype.card (Fin k ≃ ↥s)) = A.card * k.factorial := by
    calc
      _ = ∑ _s ∈ A, k.factorial := by
        apply Finset.sum_congr rfl
        intro s hs
        have hsize : Fintype.card (Fin k) = Fintype.card ↥s := by
          simpa using (mem_seedSamples.mp (hA hs)).symm
        simpa using Fintype.card_equiv (Fintype.equivOfCardEq hsize)
      _ = _ := by simp
  rw [count _ (Finset.filter_subset _ _), count _ (fun _ hs => hs), Nat.cast_mul, Nat.cast_mul]
  exact (mul_div_mul_right _ _ (Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero k))).symm

theorem uniformSeedProbability_univ (k : ℕ) : uniformSeedProbability k (fun _ => True) = 1 := by
  have h : 0 < (seedSamples k).card := by
    rw [seedSamples_card, sq]
    exact Nat.choose_pos (Nat.le_mul_self k)
  simp [uniformSeedProbability, h.ne']

private def foreignDocs (z : Fin k) : Finset (Address k) :=
  (Finset.univ.erase z) ×ˢ Finset.univ

@[simp] private theorem mem_foreignDocs {z : Fin k} {p : Address k} :
    p ∈ foreignDocs z ↔ p.1 ≠ z := by
  simp [foreignDocs]

/-- Inserting the ordinary target seed into a set of `k - 1` foreign addresses
is a bijection onto the seed event. Its inverse erases the unique target seed. -/
theorem seedEvent_count (z : Fin k) :
    ((seedSamples k).filter (SeedEvent z)).card =
      (k - 1) * (k ^ 2 - k).choose (k - 1) := by
  let choices := (foreignDocs z).powersetCard (k - 1)
  have foreign {t : Finset (Address k)} (ht : t ∈ choices) {p : Address k} (hp : p ∈ t) :
      p.1 ≠ z := mem_foreignDocs.mp ((Finset.mem_powersetCard.mp ht).1 hp)
  have hbij : ((Finset.univ.erase z) ×ˢ choices).card =
      ((seedSamples k).filter (SeedEvent z)).card := by
    apply Finset.card_bij (fun a _ => insert (z, a.1) a.2)
    · rintro ⟨j, t⟩ h
      obtain ⟨hj, ht⟩ := Finset.mem_product.mp h
      have hnot : (z, j) ∉ t := fun hm => foreign ht hm rfl
      refine Finset.mem_filter.mpr ⟨mem_seedSamples.mpr ?_, j, (Finset.mem_erase.mp hj).1,
        Finset.mem_insert_self _ _, ?_⟩
      · rw [Finset.card_insert_of_notMem hnot, (Finset.mem_powersetCard.mp ht).2,
          Nat.sub_add_cancel (Fin.pos z)]
      · intro p hp hz
        exact (Finset.mem_insert.mp hp).elim id (fun hm => (foreign ht hm hz).elim)
    · rintro ⟨j, t⟩ h ⟨j', t'⟩ h' heq
      obtain ⟨_, ht⟩ := Finset.mem_product.mp h
      obtain ⟨_, ht'⟩ := Finset.mem_product.mp h'
      change insert (z, j) t = insert (z, j') t' at heq
      have hm : (z, j) ∈ insert (z, j') t' := heq ▸ Finset.mem_insert_self _ _
      have hj : j = j' := (Finset.mem_insert.mp hm).elim (congrArg Prod.snd)
        (fun hbad => (foreign ht' hbad rfl).elim)
      subst j'
      refine Prod.ext rfl ?_
      have herase := congrArg (fun s : Finset (Address k) => s.erase (z, j)) heq
      simpa only [Finset.erase_insert (show (z, j) ∉ t from fun hm => foreign ht hm rfl),
        Finset.erase_insert (show (z, j) ∉ t' from fun hm => foreign ht' hm rfl)] using herase
    · intro s hs
      obtain ⟨hsize, j, hj, hmem, huniq⟩ := Finset.mem_filter.mp hs
      refine ⟨(j, s.erase (z, j)), Finset.mem_product.mpr ⟨by simpa using hj, ?_⟩,
        Finset.insert_erase hmem⟩
      refine Finset.mem_powersetCard.mpr ⟨?_, ?_⟩
      · intro p hp
        obtain ⟨hne, hps⟩ := Finset.mem_erase.mp hp
        exact mem_foreignDocs.mpr (fun hz => hne (huniq p hps hz))
      · rw [Finset.card_erase_of_mem hmem, mem_seedSamples.mp hsize]
  rw [← hbij]
  simp [choices, foreignDocs, Nat.sub_mul, sq]

theorem uniformSeedProbability_seedEvent (z : Fin k) :
    uniformSeedProbability k (SeedEvent z) = seedProbability k := by
  simp [uniformSeedProbability, seedEvent_count, seedSamples_card, seedProbability]

/-- Every event which holds on these seed sets inherits their exact probability
as a lower bound. The implication needs to hold only on size-`k` samples. -/
theorem seedProbability_le_of_event (z : Fin k) (P : Finset (Address k) → Prop)
    (h : ∀ s, s.card = k → SeedEvent z s → P s) :
    seedProbability k ≤ uniformSeedProbability k P := by
  rw [← uniformSeedProbability_seedEvent z]
  unfold uniformSeedProbability
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  apply Nat.cast_le.mpr
  apply Finset.card_le_card
  intro s hs
  obtain ⟨hs, he⟩ := Finset.mem_filter.mp hs
  exact Finset.mem_filter.mpr ⟨hs, h s (mem_seedSamples.mp hs) he⟩

end IndexRecall.SphericalGap.EMeans
