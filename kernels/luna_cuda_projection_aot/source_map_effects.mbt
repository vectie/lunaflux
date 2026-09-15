///|
fn projection_map_fence_source(
  plan : @luna_projection_tile_compiler.ProjectionMapEffectPlan,
  boundary : @luna_projection_tile_compiler.ProjectionMapBoundary,
) -> String {
  match plan.at(boundary) {
    None => ""
    Some(WorkgroupFence) => "__syncthreads();"
    Some(ConsumerGroupFence) => "__syncwarp();"
  }
}
