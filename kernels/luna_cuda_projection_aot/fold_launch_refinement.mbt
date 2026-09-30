///|
/// Refine an explicit offline alternative's admitted launch envelope from its
/// independent product. Both matrix work and the one-row scalar fallback must
/// fit; lowering must not derive launch cardinality from a different geometry.
fn projection_refine_fold_dimensions(
  parameters : ProjectionParameters,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  profile : @launch_contract.PagedKernelProfile,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> @launch_contract.AotLaunchDimensions raise ProjectionAotError {
  // Selected-row gathering owns a distinct launch/numerical implementation.
  if parameters.family == LanguageModelHead {
    return dimensions
  }
  let schedule = compilation.schedule()
  let (pipeline, columns) = if schedule.sibling_pipeline() is Some(pipeline) {
    (Some(pipeline), parameters.intermediate_width)
  } else {
    (schedule.matrix_pipeline(), parameters.output_width)
  }
  guard pipeline is Some(pipeline) else { return dimensions }
  guard pipeline.workgroup_count(
      maximum_rows=profile.max_query_tokens(),
      output_columns=columns,
      matrix_columns=16,
    )
    is Some(matrix_work) else {
    raise SizeOverflow(LaunchDimensions)
  }
  let groups = schedule.output_tiles_per_workgroup().to_int64()
  let scalar_work = (columns.to_int64() + groups - 1L) / groups
  let work = Int64::max(matrix_work.to_int64(), scalar_work)
  guard work > 0L && work <= 2147483647L else {
    raise SizeOverflow(LaunchDimensions)
  }
  // Matrix folds can stride through an admitted persistent launch ceiling;
  // the one-row scalar route cannot. A smaller consumer product therefore
  // needs a larger scalar floor even in a small bounded-row executable.
  let grid = Int::max(
    scalar_work.to_int(),
    Int::min(work.to_int(), dimensions.grid_x()),
  )
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid,
    grid_y=dimensions.grid_y(),
    grid_z=dimensions.grid_z(),
    block_x=schedule.output_tiles_per_workgroup() * 32,
    block_y=dimensions.block_y(),
    block_z=dimensions.block_z(),
    shared_memory_bytes=dimensions.shared_memory_bytes(),
  ) catch {
    _ => raise Invalid(LaunchDimensions)
  }
}
