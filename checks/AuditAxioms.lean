import IndexRecall
import Lloyd
import Lean.Util.CollectAxioms

/-! Audit every theorem declared in a project module, including private declarations. -/

open Lean Elab Command

run_cmd do
  let env ← getEnv
  let modules := env.allImportedModuleNames
  let declarations := env.constants.fold (fun selected name info =>
    match env.getModuleIdxFor? name with
    | some index =>
      let module := modules[index.toNat]!
      if module == `Lloyd || (`IndexRecall).isPrefixOf module then
        match info with
        | .thmInfo _ => selected.push (module, name, false)
        | .axiomInfo _ => selected.push (module, name, true)
        | _ => selected
      else selected
    | none => selected) (#[] : Array (Name × Name × Bool))
  let declarations := declarations.qsort (fun a b => a.2.1.toString < b.2.1.toString)
  -- Every theorem the docs cite must still exist (a deleted or renamed one fails the audit).
  for expected in [`IndexRecall.GapFamily.total_ratio, `IndexRecall.PlanarGap.total_ratio,
      `IndexRecall.GapFamily.optimal_partition, `IndexRecall.GapFamily.probe_thresholds,
      `IndexRecall.GapFamily.distance_counts,
      `IndexRecall.PlanarGap.strict_optimal, `IndexRecall.PlanarGap.probe_thresholds,
      `IndexRecall.PlanarGap.distance_counts,
      `IndexRecall.PlanarGap.exists_power_bias, `IndexRecall.PlanarGap.power_router_one_probe,
      `IndexRecall.PostingLists.miss_iff, `IndexRecall.PostingLists.recall_improves_iff,
      `IndexRecall.same_round_opposite_recall, `Lloyd.sharp_ratio,
      `IndexRecall.RobustPruneFailure.beam_search_robustPrune_failure,
      `IndexRecall.GraphConic.greedy_mips_exact_from_any_start,
      `IndexRecall.GraphConic.greedy_sharded_terminates_with_bound,
      `IndexRecall.FlatScan.balanced_work_cap, `IndexRecall.SphericalGap.cosine_gap,
      `IndexRecall.SphericalGap.many_query_ivf_gap,
      `IndexRecall.SphericalGap.EMeans.emeans_gap, `IndexRecall.SphericalGap.EMeans.random_start_training,
      `IndexRecall.SphericalGap.EMeans.uniformSeedProbability_seedEvent,
      `IndexRecall.SphericalGap.EMeans.tendsto_seedProbability,
      `IndexRecall.SphericalGap.EMeans.random_start_last_probability,
      `IndexRecall.RobustPruneFailure.bidirectional_robustPrune_failure,
      `IndexRecall.GraphConic.mem_finCone_iff_dual,
      `IndexRecall.GraphConic.universal_mips_navigable_iff,
      `IndexRecall.GraphConic.sharded_conic_navigable_iff,
      `IndexRecall.GraphConic.approximate_sharded_conic_bound,
      `IndexRecall.GraphConic.exact_shard_recovery_under_margin,
      `IndexRecall.OneVectorImpossibility.twoTip_returns_all,
      `IndexRecall.OneVectorImpossibility.affine_answered_le,
      `IndexRecall.OneVectorImpossibility.affine_fullRecall_budget_ge,
      `IndexRecall.OneVectorImpossibility.affine_sum_tieRank_ge,
      `IndexRecall.OneVectorImpossibility.affine_sum_work_ge,
      `IndexRecall.OneVectorImpossibility.powerDist_sum_work_ge,
      `IndexRecall.OneVectorImpossibility.balanced_affine_vs_twoTip_separation,
      `IndexRecall.OneVectorImpossibility.linear_recall_le_half,
      `IndexRecall.OneVectorImpossibility.linear_sum_work_ge,
      `IndexRecall.OneVectorImpossibility.witness_sum_tieRank_eq,
      `IndexRecall.OneVectorImpossibility.witness_sum_work_eq,
      `IndexRecall.OneVectorImpossibility.affine_min_fullRecall_budget,
      `IndexRecall.OneVectorCircle.antipodal_sum_tieRank_ge,
      `IndexRecall.OneVectorCircle.antipodal_fullRecall_budget_ge,
      `IndexRecall.OneVectorCircle.circle_doc_injective,
      `IndexRecall.OneVectorCircle.circle_query_margin,
      `IndexRecall.OneVectorCircle.circle_twoTip_returns_all,
      `IndexRecall.OneVectorCircle.circle_affine_sum_work_ge,
      `IndexRecall.OneVectorCircle.circle_powerDist_sum_work_ge,
      `IndexRecall.OneVectorCircle.circle_affine_vs_twoTip,
      `IndexRecall.OneVectorCircle.circle_witness_sum_tieRank_eq] do
    unless declarations.any (fun d => d.2.1 == expected) do
      throwError m!"expected proof missing from audit: {expected}"
  for (module, name, isAxiom) in declarations do
    if isAxiom then throwError m!"project axiom {module}.{name}"
    let axioms ← collectAxioms name
    IO.println s!"{module}\t{name}\t{axioms.map Name.toString}"
    for axiomName in axioms do
      unless [`propext, `Classical.choice, `Quot.sound].contains axiomName do
        throwError m!"{name} depends on unexpected axiom {axiomName}"
  -- Equation lemmas, structure `inj`/`injEq`/`sizeOf_spec`/`ctorIdx` lemmas, and auxiliary
  -- proofs (`_proof_*`, `_simp_*`, `match_*`), structure-field projections and `brecOn` that Lean creates on its own
  -- (private names also start with `_private`, but those are written by hand).
  let generated := declarations.filter (fun d => (env.getProjectionFnInfo? d.2.1).isSome ||
    d.2.1.components.any (fun c =>
    let t := c.toString
    t.startsWith "eq_" || t == "injEq" || t == "inj" || t == "sizeOf_spec" || t.endsWith "ctorIdx" ||
      t.startsWith "_simp" || t == "brecOn" || t == "binductionOn" || t == "congr_simp" || t.startsWith "_proof" || t.startsWith "proof_" ||
      t.startsWith "match_" || t.startsWith "_auxLemma" || t == "_sunfold" || t == "_unfold"))
  IO.println s!"Audited {declarations.size} theorem declarations ({generated.size} generated by Lean, {declarations.size - generated.size} others)."
