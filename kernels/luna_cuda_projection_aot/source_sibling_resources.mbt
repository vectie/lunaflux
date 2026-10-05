///|
/// CUDA schedule policy for the fully occupied sibling workgroup topology.
/// The launch width comes from the immutable schedule, not a model shape.
/// A two-resident-block register target recovered throughput for this topology;
/// it is not a promise of residency or a hardware-independent optimum.
/// Launch bounds can override the file-level register budget. Omit them when
/// the caller requests a budget below this topology's 64-register threshold,
/// preserving the ordinary compiler policy rather than silently relaxing it.
/// The companion down entry point has a different lifetime and resource plan
/// and deliberately does not consume this qualifier.
fn projection_sibling_launch_bounds_source(
  compilation : @luna_projection_tile_compiler.PhysicalProjectionTileCompilation,
  max_registers~ : Int,
) -> String {
  let groups = compilation.schedule().output_tiles_per_workgroup()
  let threads = groups * 32
  guard threads == 512 && max_registers >= 64 else { return "" }
  guard compilation.sibling is Some(physical) else { return "" }
  let distribution = physical.distribution
  // Bounding the row product can leave launched groups without consumers.
  // That is a different resource topology from the fully occupied reuse map.
  if physical.sibling_reuse {
    guard distribution.consumer_groups() * 2 == groups else { return "" }
  } else {
    // Two one-row products have the same live accumulator count as the
    // two-row split product. Ownership alone must not remove its residency
    // target. Larger live row products need their own measured resource plan.
    guard compilation.sibling_epilogue is Some(epilogue) &&
      epilogue.ownership() == CoownedSiblingValues &&
      distribution.consumer_groups() == groups &&
      distribution.rows_per_group() == 1 else {
      return ""
    }
  }
  "__launch_bounds__(\{threads}, 2) "
}
