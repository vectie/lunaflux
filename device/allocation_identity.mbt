///|
/// Compare allocation-owner identity, never byte count or a recycled device
/// address. This startup join check is not a liveness or completion proof;
/// executor binding still validates that each allocation is live.
pub fn Allocation::same_storage(self : Self, other : Self) -> Bool {
  same_storage_owner(self.raw, other.raw)
}

///|
fn[T] same_storage_owner(left : T, right : T) -> Bool {
  physical_equal(left, right)
}
