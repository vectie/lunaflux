///|
fn projection_companion_dimensions(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  profile : @launch_contract.PagedKernelProfile,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  refine_fold_geometry? : Bool = false,
) -> @launch_contract.AotLaunchDimensions? raise ProjectionAotError {
  if !projection_tensor_core_mlp_eligible(parameters, dispatch) {
    return None
  }
  guard compilation.schedule().companion_output_tiles_per_workgroup()
    is Some(groups) else {
    return None
  }
  let width = parameters.output_width.to_int64()
  let rows = (profile.max_query_tokens().to_int64() + 15L) / 16L
  let columns = (width + groups.to_int64() * 16L - 1L) /
    (groups.to_int64() * 16L)
  let work = if refine_fold_geometry {
    guard compilation.schedule().intermediate_pipeline() is Some(pipeline) &&
      pipeline.workgroup_count(
        maximum_rows=profile.max_query_tokens(),
        output_columns=parameters.output_width,
        matrix_columns=16,
      )
      is Some(matrix_work) else {
      raise SizeOverflow(LaunchDimensions)
    }
    Int64::max(
      matrix_work.to_int64(),
      (width + groups.to_int64() * 4L - 1L) / (groups.to_int64() * 4L),
    )
  } else {
    Int64::max(
      rows * columns,
      (width + groups.to_int64() - 1L) / groups.to_int64(),
    )
  }
  if work <= 0L || work > 2147483647L {
    raise SizeOverflow(LaunchDimensions)
  }
  Some(
    @launch_contract.AotLaunchDimensions::new(
      grid_x=work.to_int(),
      grid_y=1,
      grid_z=1,
      block_x=groups * 32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ) catch {
      _ => raise Invalid(LaunchDimensions)
    },
  )
}

///|
pub fn ProjectionCudaLowering::companion_dimensions(
  self : ProjectionCudaLowering,
) -> @launch_contract.AotLaunchDimensions? {
  self.companion_dimensions
}
