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
  schedule : @luna_projection_tile_compiler.ProjectionTileSchedule,
  max_registers~ : Int,
) -> String {
  let groups = schedule.output_tiles_per_workgroup()
  let threads = groups * 32
  guard threads == 512 && max_registers >= 64 else { return "" }
  guard schedule.sibling_pipeline() is Some(pipeline) else { return "" }
  guard @luna_projection_tile_compiler.partition_projection_sibling_reuse(
      row_tiles=pipeline.row_width() / 16,
      output_tiles=pipeline.output_tiles(),
      consumer_groups=groups,
    )
    is Some(_) else {
    return ""
  }
  "__launch_bounds__(\{threads}, 2) "
}
