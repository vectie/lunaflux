///|
pub fn CheckedI32Table::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn CheckedI32Table::column_count(self : Self) -> Int {
  self.column_count
}

///|
pub fn CheckedI32Table::minimum_value(self : Self) -> Int {
  self.minimum_value
}

///|
pub fn CheckedI32Table::exclusive_maximum_value(self : Self) -> Int {
  self.exclusive_maximum_value
}

///|
pub fn CheckedI32Table::byte_count(self : Self) -> Int {
  self.row_count * self.column_count * 4
}

///|
pub fn CheckedI32Table::is_released(self : Self) -> Bool {
  self.released
}

///|
/// Synchronously lends the derived table as arena zero. The caller cannot
/// retain a reusable borrow capability after the call returns.
pub fn CheckedI32Table::provide_arena(
  self : Self,
  consumer : @segmented_host_borrow.SegmentedHostArenaConsumer,
) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError {
  if self.released {
    raise SourceUnavailable(0)
  }
  guard self.bytes is Some(bytes) else { raise SourceUnavailable(0) }
  consumer.accept(bytes)
}

///|
/// Drops the derived I32 payload exactly once without scrubbing its bytes.
pub fn CheckedI32Table::release(self : Self) -> Bool {
  if self.released {
    return false
  }
  self.bytes = None
  self.released = true
  true
}
