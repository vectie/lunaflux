///|
/// Publish a bounded executable's compiler-owned product cardinality. CUDA's
/// matrix-column width belongs here, never in the engine's bucket arithmetic.
/// The admitted launch remains the maximum envelope (including scalar-row
/// coverage), and any published persistent-launch ceiling remains authoritative.
/// A distinct selected-row/gather algorithm keeps its existing launch rule.
fn projection_bounded_work_grid_x(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Int? {
  let schedule = compilation.schedule()
  let (pipeline, output_columns) = if schedule.sibling_pipeline() is Some(plan) {
    (Some(plan), parameters.intermediate_width)
  } else {
    (schedule.matrix_pipeline(), parameters.output_width)
  }
  guard pipeline is Some(plan) else { return None }
  let rows = if schedule.demand() == QueryRowEnds {
    maximum_rows
  } else {
    maximum_tokens
  }
  guard plan.workgroup_count(
      maximum_rows=rows,
      output_columns~,
      matrix_columns=16,
    )
    is Some(count) else {
    return None
  }
  Some(if count < dimensions.grid_x() { count } else { dimensions.grid_x() })
}
