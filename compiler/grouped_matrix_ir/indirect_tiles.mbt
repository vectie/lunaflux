///|
/// Capacity bounds storage, not the number of idle workgroups to launch.
/// Each worker visits tile w, w+workers, ... below the device-published live
/// tile count. This is a partition of the domain, not a matrix reduction.
pub struct IndirectMatrixTraversal {
  priv capacity_groups : Int32
  priv column_tiles : Int32
  priv workers : Int32
}

///|
pub fn IndirectMatrixTraversal::new(
  capacity_groups~ : Int32,
  column_tiles~ : Int32,
  workers? : Int32,
) -> Self raise @precision.PrecisionPlanError {
  guard capacity_groups > 0 &&
    column_tiles > 0 &&
    capacity_groups.to_int64() * column_tiles.to_int64() <= 2147483647L else {
    raise ArithmeticOverflow
  }
  // One worker per bounded job, independent of output width. A worker may
  // execute multiple independent tiles; runtime group counts remain on device.
  let workers = workers.unwrap_or(capacity_groups)
  guard workers > 0 && workers <= 1048576 else { raise IllegalPlacement }
  {
    capacity_groups,
    column_tiles,
    workers: workers.min(capacity_groups * column_tiles),
  }
}

///|
pub fn IndirectMatrixTraversal::workers(self : Self) -> Int32 {
  self.workers
}

///|
pub fn IndirectMatrixTraversal::tile_count(
  self : Self,
  live_groups : Int32,
) -> Int32 raise @precision.PrecisionPlanError {
  guard live_groups >= 0 && live_groups <= self.capacity_groups else {
    raise IllegalPlacement
  }
  live_groups * self.column_tiles
}
