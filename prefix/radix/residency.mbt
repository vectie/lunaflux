///|
/// Host residency changes payload placement only. Token roots, copied scope,
/// shared page anchors, and prefix identities remain in the same radix index.
pub fn PrefixLookupBuffer::host_at(
  self : PrefixLookupBuffer,
  index : Int,
) -> @streaming_ir.HostPage raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  self.hosts[index]
}

///|
pub fn PrefixLookupBuffer::cell_at(
  self : PrefixLookupBuffer,
  index : Int,
) -> Int raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  self.cells[index]
}

///|
pub fn PrefixLookupBuffer::fully_resident(self : PrefixLookupBuffer) -> Bool {
  for i in 0..<self.page_count {
    if self.hosts[i].slot >= 0 {
      return false
    }
  }
  true
}

///|
pub fn PrefixEvictionBuffer::host_at(
  self : PrefixEvictionBuffer,
  index : Int,
) -> @streaming_ir.HostPage raise PrefixIndexError {
  if index < 0 || index >= self.page_count {
    raise ResultIndexOutOfBounds
  }
  self.hosts[index]
}

///|
pub fn PrefixIndex::page_occupied(self : PrefixIndex, cell : Int) -> Bool {
  cell >= 0 &&
  cell < self.page_hosts.length() &&
  (try! self.page_values.is_occupied(cell))
}

///|
pub fn PrefixIndex::page_host(
  self : PrefixIndex,
  cell : Int,
) -> @streaming_ir.HostPage raise PrefixIndexError {
  if !self.page_occupied(cell) {
    raise ResultIndexOutOfBounds
  }
  self.page_hosts[cell]
}

///|
pub fn PrefixIndex::page_value(
  self : PrefixIndex,
  cell : Int,
) -> @page_allocator.PageId raise PrefixIndexError {
  if !self.page_occupied(cell) {
    raise ResultIndexOutOfBounds
  }
  self.page_values.read(cell) catch {
    _ => raise ResultIndexOutOfBounds
  }
}

///|
/// Called only by the page owner after successful DMA completion. Compare the
/// exact prior payload before committing; no token/key reconstruction occurs.
pub fn PrefixIndex::spill_page(
  self : PrefixIndex,
  cell : Int,
  expected : @page_allocator.PageId,
  host : @streaming_ir.HostPage,
) -> Bool {
  if self.publish_plan_active ||
    self.eviction_plan_active ||
    !self.page_occupied(cell) ||
    self.page_hosts[cell].slot >= 0 ||
    host.slot < 0 ||
    host.generation <= 0 ||
    (try! self.page_values.read(cell)) != expected {
    return false
  }
  self.page_hosts[cell] = host
  true
}

///|
pub fn PrefixIndex::restore_page(
  self : PrefixIndex,
  cell : Int,
  expected : @streaming_ir.HostPage,
  page : @page_allocator.PageId,
) -> Bool {
  if self.publish_plan_active ||
    self.eviction_plan_active ||
    !self.page_occupied(cell) ||
    !self.page_hosts[cell].same(expected) ||
    expected.slot < 0 {
    return false
  }
  try! self.page_values.write(cell, page)
  self.page_hosts[cell] = { slot: -1, generation: 0, }
  true
}
