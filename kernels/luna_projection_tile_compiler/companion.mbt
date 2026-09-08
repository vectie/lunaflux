///|
/// The consumer of a materialized intermediate is a separate parallel map.
/// Keep its distribution independent of the producer's two live accumulators.
/// This value counts output tiles, not hardware threads or warp lanes.
pub fn ProjectionTileSchedule::companion_output_tiles_per_workgroup(
  self : ProjectionTileSchedule,
) -> Int? {
  if self.reuse_intermediate_tile {
    Some(projection_companion_output_tiles(self.output_tiles_per_workgroup))
  } else {
    None
  }
}

///|
fn projection_companion_output_tiles(producer_tiles : Int) -> Int {
  Int::min(4, producer_tiles)
}
