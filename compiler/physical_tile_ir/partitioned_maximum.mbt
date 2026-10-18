///|
/// Disjoint partial publication for an exact finite nonnegative maximum.
/// Producers write only the appended partial plane. Consumers read that plane
/// until their ordered launch retires, never a concurrently published result.
pub struct PartitionedMaximum {
  priv partitions : Int
  priv group_elements : Int
  priv partial_offset : Int64
  priv allocation_bytes : Int64
} derive(Eq, Debug)

///|
pub extend PartitionedMaximum with Eq::{equal, not_equal}

///|
pub extend PartitionedMaximum with @debug.Debug::{to_repr}

///|
pub fn PartitionedMaximum::group_elements(self : Self) -> Int {
  self.group_elements
}

///|
pub fn PartitionedMaximum::new(
  partitions~ : Int,
  group_elements~ : Int,
  after_bytes~ : Int64,
) -> Self? {
  guard partitions > 0 &&
    partitions <= 256 &&
    group_elements > 0 &&
    after_bytes > 0L &&
    after_bytes <= 9223372036854774776L else {
    return None
  }
  let partial_offset = (after_bytes + 3L) / 4L * 4L
  Some({
    partitions,
    group_elements,
    partial_offset,
    allocation_bytes: partial_offset + partitions.to_int64() * 4L,
  })
}

///|
pub fn PartitionedMaximum::partitions(self : Self) -> Int {
  self.partitions
}

///|
pub fn PartitionedMaximum::partial_offset(self : Self) -> Int64 {
  self.partial_offset
}

///|
pub fn PartitionedMaximum::allocation_bytes(self : Self) -> Int64 {
  self.allocation_bytes
}

///|
/// Decode uses one producer; every other live row count uses the finite pool.
pub fn PartitionedMaximum::live_partitions(self : Self, live_rows : Int) -> Int {
  if live_rows <= 0 {
    0
  } else if live_rows == 1 {
    1
  } else {
    self.partitions
  }
}

///|
/// Strided ownership covers each input exactly once without inter-group waits.
pub fn PartitionedMaximum::owner(
  self : Self,
  index : Int64,
  live_rows : Int,
) -> Int? {
  let count = self.live_partitions(live_rows)
  if index < 0L || count == 0 {
    None
  } else {
    Some((index / self.group_elements.to_int64() % count.to_int64()).to_int())
  }
}
