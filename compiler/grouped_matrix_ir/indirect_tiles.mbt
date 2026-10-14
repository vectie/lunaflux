///|
/// Capacity bounds storage, not the number of idle workgroups to launch.
/// Each worker visits tile w, w+workers, ... below the device-published live
/// tile count. This is a partition of the domain, not a matrix reduction.
pub struct IndirectMatrixTraversal {
  priv capacity_groups : Int
  priv column_tiles : Int
  priv workers : Int
}

///|
pub fn IndirectMatrixTraversal::new(
  capacity_groups~ : Int,
  column_tiles~ : Int,
) -> Self raise @precision.PrecisionPlanError {
  guard capacity_groups > 0 &&
    column_tiles > 0 &&
    capacity_groups.to_int64() * column_tiles.to_int64() <= 2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  // One worker per bounded job, independent of output width. A worker may
  // execute multiple independent tiles; runtime group counts remain on device.
  { capacity_groups, column_tiles, workers: capacity_groups, }
}

///|
pub fn IndirectMatrixTraversal::workers(self : Self) -> Int {
  self.workers
}

///|
pub fn IndirectMatrixTraversal::tile_count(
  self : Self,
  live_groups : Int,
) -> Int raise @precision.PrecisionPlanError {
  guard live_groups >= 0 && live_groups <= self.capacity_groups else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  live_groups * self.column_tiles
}
