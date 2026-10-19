///|
/// Incremental materialization: a completed pool is immutable until one of its
/// source rows is written. The first writer in the current frame owns its one
/// recomputation. Physical addresses, not request order, identify the pool.
pub struct PoolUpdatePlan {
  priv rows : Int
  priv pool : Int
}

///|
pub fn PoolUpdatePlan::new(rows~ : Int, pool~ : Int) -> Self? {
  if rows <= 0 || pool <= 0 {
    None
  } else {
    Some({ rows, pool, })
  }
}

///|
pub fn PoolUpdatePlan::workgroups(self : Self) -> Int {
  self.rows
}

///|
/// Negative addresses are inactive. This pure oracle describes the device
/// first-writer ownership, including same-generation overwrites and reorders.
pub fn PoolUpdatePlan::owns_update(
  self : Self,
  addresses : ArrayView[Int],
  row : Int,
) -> Bool {
  guard row >= 0 &&
    row < self.rows &&
    row < addresses.length() &&
    addresses[row] >= 0 else {
    return false
  }
  let pool = addresses[row] / self.pool
  for earlier in 0..<row {
    if addresses[earlier] >= 0 && addresses[earlier] / self.pool == pool {
      return false
    }
  }
  true
}
