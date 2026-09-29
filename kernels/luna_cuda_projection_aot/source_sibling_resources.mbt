///|
/// CUDA schedule policy for the measured sibling-reuse workgroup topology.
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
  guard compilation.sibling is Some(physical) && physical.sibling_reuse else {
    return ""
  }
  let reuse = physical.distribution
  // Bounding the row product can leave launched groups without consumers.
  // That is a different resource topology from the fully occupied reuse map.
  guard reuse.consumer_groups() * 2 == groups else { return "" }
  "__launch_bounds__(\{threads}, 2) "
}
