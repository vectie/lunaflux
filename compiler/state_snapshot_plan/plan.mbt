///|
pub(all) suberror StateSnapshotPlanError {
  InvalidStateSize
  SnapshotBudgetExceeded
} derive(Debug, Eq)

///|
/// Pure placement of mutable state into one reusable backup arena. No weights,
/// scratch, device names, pointer identities or authentication enter this plan.
pub struct StateSnapshotPlan {
  priv sizes : ReadOnlyArray[Int64]
  priv offsets : ReadOnlyArray[Int64]
  priv bytes : Int64
} derive(Debug, Eq)

///|
pub fn plan_state_snapshot(
  sizes : ArrayView[Int64],
  maximum_backup_bytes~ : Int64,
) -> StateSnapshotPlan raise StateSnapshotPlanError {
  guard sizes.length() > 0 && maximum_backup_bytes > 0L else {
    raise InvalidStateSize
  }
  let offsets = []
  let owned_sizes = []
  let mut bytes = 0L
  for size in sizes {
    guard size > 0L else { raise InvalidStateSize }
    guard size <= maximum_backup_bytes - bytes else {
      raise SnapshotBudgetExceeded
    }
    offsets.push(bytes)
    owned_sizes.push(size)
    bytes += size
  }
  {
    sizes: ReadOnlyArray::from_array(owned_sizes),
    offsets: ReadOnlyArray::from_array(offsets),
    bytes,
  }
}

///|
pub fn StateSnapshotPlan::backup_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn StateSnapshotPlan::sizes(self : Self) -> ReadOnlyArray[Int64] {
  self.sizes
}

///|
pub fn StateSnapshotPlan::offsets(self : Self) -> ReadOnlyArray[Int64] {
  self.offsets
}
