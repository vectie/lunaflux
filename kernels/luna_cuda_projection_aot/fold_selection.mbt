///|
/// Measurement-driven lowering. Workload and lowering identity are bound by
/// the generated source; unmeasured shapes keep their ordinary strategy.
pub fn lower_measured_projection_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  dispatch : @luna_projection_strategy.ProjectionRuntimeDispatch,
  limits : ProjectionAotLimits,
  records : ArrayView[@luna_projection_tuning.ProjectionFoldRecord],
  minimum_improvement_basis_points? : Int = 100,
) -> ProjectionCudaLowering raise ProjectionAotError {
  if minimum_improvement_basis_points < 0 ||
    minimum_improvement_basis_points > 10000 {
    raise Invalid(CompilerPolicy)
  }
  let baseline = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation, profile, target, compiler, operands, dispatch, limits,
  )
  let digest = baseline.source_digest().as_hex()
  let baseline_times = records.filter(r => {
    r.baseline_sha256() == digest && r.fold_choices().is_empty()
  })
  guard baseline_times is [measured] else { return baseline }
  // A tiny timing fluctuation is not a reason to change an AOT schedule.
  // Compare against the baseline, not against each intermediate winner, so
  // table order cannot prevent selecting the best eligible alternative.
  let baseline_cost = measured.median_latency_ns()
  let margin = minimum_improvement_basis_points.to_int64()
  let required_gain = baseline_cost / 10000L * margin +
    baseline_cost % 10000L * margin / 10000L
  let cutoff = baseline_cost - Int64::max(1L, required_gain)
  for
    record in records
    selected = baseline, cost = measured.median_latency_ns() {
    if record.baseline_sha256() != digest ||
      record.fold_choices().is_empty() ||
      record.median_latency_ns() > cutoff ||
      record.median_latency_ns() > cost ||
      (
        record.median_latency_ns() == cost &&
        (
          selected.source_digest() == baseline.source_digest() ||
          record.artifact_sha256() >= selected.source_digest().as_hex()
        )
      ) {
      continue selected, cost
    }
    let candidate = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      fold_choices=record.fold_choices()[:],
    ) catch {
      LimitExceeded(LaunchDimensions) => continue selected, cost
      error => raise error
    }
    guard candidate.source_digest().as_hex() == record.artifact_sha256() else {
      raise Invalid(CompilerPolicy)
    }
    continue candidate, record.median_latency_ns()
  } nobreak {
    selected
  }
}
