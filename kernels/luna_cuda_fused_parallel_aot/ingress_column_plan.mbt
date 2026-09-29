///|
/// Backend resource envelope for the existing four-WMMA-group ingress CTA.
/// Two live fragments cover its supported head width of at most 128 without
/// an unbounded register array. This is a static bound, not an autotune result.
/// The compiler owns the window partition, operand lifetime and fold order.
fn qwen_ingress_column_plan(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
) -> @luna_projection_tile_compiler.ProjectionColumnFoldPlan {
  guard compilation
    .schedule()
    .column_fold_plan(consumer_groups=4, maximum_live_accumulators=2)
    is Some(plan) else {
    abort("CUDA attention ingress requires a matrix column-fold plan")
  }
  plan
}
