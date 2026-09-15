///|
/// Static CUDA storage is separate from launch-time dynamic storage. The
/// current launch contract does not opt in to a larger per-block limit.
/// Reject offline choices which need that unavailable launch capability.
fn projection_validate_fold_resources(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  dynamic_bytes : Int,
) -> Unit raise ProjectionAotError {
  let schedule = compilation.schedule()
  for
    (plan, products, dynamic_bytes) in [
      (schedule.matrix_pipeline(), 1, dynamic_bytes),
      (schedule.sibling_pipeline(), 2, dynamic_bytes),
      (schedule.intermediate_pipeline(), 1, 0),
    ] {
    guard plan is Some(plan) else { continue }
    let rows = plan.row_width().to_int64()
    let columns = plan.output_tiles().to_int64() * 16L
    let ring_bytes = (rows + products.to_int64() * columns) *
      plan.transfer_width().to_int64() *
      plan.stage_count().to_int64() *
      2L
    // The sibling epilogue aliases one float result plane onto its dead ring.
    let static_bytes = if products == 2 {
      Int64::max(ring_bytes, rows * columns * 4L)
    } else {
      ring_bytes
    }
    if static_bytes > 49152L || static_bytes + dynamic_bytes.to_int64() > 49152L {
      raise LimitExceeded(LaunchDimensions)
    }
  }
}
