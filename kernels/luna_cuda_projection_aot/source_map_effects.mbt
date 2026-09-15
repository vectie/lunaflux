///|
fn projection_map_fence_source(
  plan : @luna_projection_tile_compiler.ProjectionMapEffectPlan,
  boundary : @luna_projection_tile_compiler.ProjectionMapBoundary,
) -> String {
  projection_materialization_fence_source(plan.at(boundary))
}

///|
fn projection_materialization_fence_source(
  fence : @luna_projection_tile_compiler.ProjectionStorageFence?,
) -> String {
  match fence {
    None => ""
    Some(WorkgroupFence) => "__syncthreads();"
    Some(ConsumerGroupFence) => "__syncwarp();"
  }
}
