import IndexRecall.ManyQueryOptimality
import IndexRecall.SphericalIVF

/-! # A single table of tips routes an entire spherical IVF workload -/

noncomputable section
open scoped InnerProductSpace

namespace IndexRecall.SphericalGap
namespace ManyQuery

variable {k : ℕ}

/-- Negated cosine is the increasing score used by the flat router. -/
def routingScore (centers : Fin k → Space k) (i j : Fin k) : ℝ := -⟪query i, centers j⟫_ℝ

/-- Mean number of representative and document scores under the uniform law on the k queries. -/
def uniformWork (centers : Fin k → Space k) (probes : ℕ) : ℝ :=
  (∑ i, (FlatScan.work (Fin k) (routingScore centers i) probes : ℝ)) / k

private theorem uniformWork_of_constant (hk : 0 < k) (centers : Fin k → Space k) (p n : ℕ)
    (h : ∀ i, FlatScan.work (Fin k) (routingScore centers i) p = n) :
    uniformWork centers p = n := by
  simp only [uniformWork, h, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact mul_div_cancel_left₀ _ (by exact_mod_cast hk.ne')

/-- Both routers scan the same stored rows of x. The table `tip` depends only on
the list label. Every query has its own unique answer in a different list. -/
structure Gap (x : Fin k × Fin k → Space k) : Prop where
  dimension : Module.finrank ℝ (Space k) = 2 * k
  database_size : Fintype.card (Fin k × Fin k) = k^2
  list_size : ∀ i : Fin k, (Finset.univ.filter (fun p : Fin k × Fin k => p.1 = i)).card = k
  distinct : Function.Injective x
  documents_unit : ∀ p, ‖x p‖ = 1
  queries_unit : ∀ i : Fin k, ‖query i‖ = 1
  tips : ∀ i, x (i,i) = tip i
  optimum : UniqueOptimum x
  margin : ∀ i p, p ≠ (i,i) → ⟪query i, x p⟫_ℝ + 9 / 100 < ⟪query i, x (i,i)⟫_ℝ
  mean_last : ∀ i j, j ≠ i →
    ⟪query i, meanDirection x i⟫_ℝ < ⟪query i, meanDirection x j⟫_ℝ
  tips_first : ∀ i j : Fin k, j ≠ i → ⟪query i, tip j⟫_ℝ < ⟪query i, tip i⟫_ℝ
  scans : ∀ i, ScanGap (fun p => -⟪query i, x p⟫_ℝ)
    (routingScore (meanDirection x) i) (routingScore tip i) (i,i)
  uniform_mean_work : uniformWork (meanDirection x) k = ((k + k*k : ℕ) : ℝ)
  uniform_tip_work : uniformWork (tip (k := k)) 1 = ((k + k : ℕ) : ℝ)
  uniform_ratio : uniformWork (meanDirection x) k / uniformWork (tip (k := k)) 1 = ((k : ℝ) + 1) / 2

/-- Every positive perturbation in the stated interval works for all queries simultaneously. -/
theorem gap_of_perturbation (hk : 2 ≤ k) {t : ℝ} (ht : 0 < t) (hle : t ≤ perturbationRadius k) :
    Gap (site (k := k) t) := by
  have h1 := hle.trans (radius_bounds hk).2
  let x := site (k := k) t
  have hs (i : Fin k) : ScanGap (fun p => -⟪query i, x p⟫_ℝ)
      (routingScore (meanDirection x) i) (routingScore tip i) (i,i) := by
    apply scan_gap
    · intro b hb
      apply neg_lt_neg
      have hm := margin ht.le h1 i b hb
      change ⟪query i, x b⟫_ℝ < ⟪query i, x (i,i)⟫_ℝ
      linarith only [show ⟪query i, x b⟫_ℝ + 9 / 100 < ⟪query i, x (i,i)⟫_ℝ from hm]
    · exact fun j hj => neg_lt_neg (means_last hk ht.le h1 i j hj)
    · exact fun j hj => neg_lt_neg (tips_first i j hj)
  have hk0 : 0 < k := by omega
  have wm := uniformWork_of_constant hk0 (meanDirection x) k _ (fun i => (hs i).mean_work)
  have wr := uniformWork_of_constant hk0 tip 1 _ (fun i => (hs i).retuned_work)
  refine ⟨dimension _, by simp [sq], row_card, site_injective ht h1, site_unit ht.le h1,
    query_unit, site_tip t, unique_optimum hk ht.le hle, margin ht.le h1, means_last hk ht.le h1,
    tips_first, hs, wm, wr, ?_⟩
  let i : Fin k := ⟨0, hk0⟩
  change uniformWork (meanDirection x) k / uniformWork (tip (k := k)) 1 = _
  simpa only [wm, wr, (hs i).mean_work, (hs i).retuned_work] using (hs i).ratio

end ManyQuery

/-- For every k ≥ 2 this explicit unit dataset in ℝ^(2k) has a unique spherical
k-means optimum up to labels, over all assignments. Normalized means cost (k+1)/2
times as much work as the one fixed table of tips, per query and in uniform expectation. -/
theorem many_query_ivf_gap {k : ℕ} (hk : 2 ≤ k) :
    ManyQuery.Gap (ManyQuery.site (k := k) (ManyQuery.perturbationRadius k)) :=
  ManyQuery.gap_of_perturbation hk (ManyQuery.radius_bounds hk).1 le_rfl

end IndexRecall.SphericalGap

#print axioms IndexRecall.SphericalGap.many_query_ivf_gap
