///|
/// Static CUDA storage is separate from launch-time dynamic storage. The
/// current launch contract does not opt in to a larger per-block limit.
/// Reject offline choices which need that unavailable launch capability.
fn projection_validate_fold_resources(
  compilation : @luna_projection_tile_compiler.PhysicalProjectionTileCompilation,
  dynamic_bytes : Int,
) -> Unit raise ProjectionAotError {
  for
    (plan, dynamic_bytes) in [
      (compilation.matrix, dynamic_bytes),
      (compilation.sibling, dynamic_bytes),
      (compilation.intermediate, 0),
    ] {
    guard plan is Some(plan) else { continue }
    let static_bytes = plan.static_storage_bytes(
      element_bytes=2,
      result_bytes=4,
    )
    if static_bytes > 49152L || static_bytes + dynamic_bytes.to_int64() > 49152L {
      raise LimitExceeded(LaunchDimensions)
    }
  }
}
