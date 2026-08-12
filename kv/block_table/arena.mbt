///|
const TABLE_FREE : Int = 0

///|
const TABLE_ALLOCATED : Int = 1

///|
const TABLE_RETIRED : Int = 2

///|
/// Preallocates the complete dense mapping arena and ascending FIFO free list.
pub fn BlockTableArena::new(
  limits : BlockTableLimits,
) -> BlockTableArena raise BlockTableError {
  let capacity = limits.table_capacity()
  let max_pages = limits.max_pages_per_table()
  let free_next = FixedArray::make(capacity, -1)
  for index in 0..<capacity {
    free_next[index] = if index + 1 < capacity { index + 1 } else { -1 }
  }
  {
    limits,
    table_capacity: capacity,
    max_pages_per_table: max_pages,
    tokens_per_page: limits.tokens_per_page(),
    max_generation: limits.max_generation(),
    states: FixedArray::make(capacity, TABLE_FREE),
    generations: FixedArray::make(capacity, 0),
    page_counts: FixedArray::make(capacity, 0),
    pages: page_storage_new(capacity * max_pages, PageCellCapacity),
    free_next,
    free_head: 0,
    free_tail: capacity - 1,
    free_tables: capacity,
    allocated_tables: 0,
    retired_tables: 0,
  }
}

///|
pub fn BlockTableArena::limits(self : BlockTableArena) -> BlockTableLimits {
  self.limits
}

///|
pub fn BlockTableArena::free_table_count(self : BlockTableArena) -> Int {
  self.free_tables
}

///|
pub fn BlockTableArena::allocated_table_count(self : BlockTableArena) -> Int {
  self.allocated_tables
}

///|
pub fn BlockTableArena::retired_table_count(self : BlockTableArena) -> Int {
  self.retired_tables
}

///|
fn BlockTableArena::checked_index(
  self : BlockTableArena,
  table : BlockTableId,
) -> Int raise BlockTableError {
  let index = table.index()
  if index < 0 ||
    index >= self.table_capacity ||
    self.generations[index] != table.generation() {
    raise StaleTable(table)
  }
  if self.states[index] != TABLE_ALLOCATED {
    raise TableNotAllocated(table)
  }
  index
}

///|
/// Allocates the oldest free table as an empty mapping.
pub fn BlockTableArena::allocate(
  self : BlockTableArena,
) -> BlockTableId raise BlockTableError {
  if self.free_tables == 0 {
    raise Exhausted
  }
  let index = self.free_head
  let next = self.free_next[index]
  self.free_head = next
  if next == -1 {
    self.free_tail = -1
  }
  self.free_next[index] = -1
  self.states[index] = TABLE_ALLOCATED
  self.generations[index] += 1
  self.page_counts[index] = 0
  self.free_tables -= 1
  self.allocated_tables += 1
  BlockTableId(index, self.generations[index])
}

///|
fn BlockTableArena::recycle_empty(self : BlockTableArena, index : Int) -> Unit {
  self.allocated_tables -= 1
  if self.generations[index] == self.max_generation {
    self.states[index] = TABLE_RETIRED
    self.retired_tables += 1
    return
  }
  self.states[index] = TABLE_FREE
  self.free_next[index] = -1
  if self.free_tail == -1 {
    self.free_head = index
    self.free_tail = index
  } else {
    self.free_next[self.free_tail] = index
    self.free_tail = index
  }
  self.free_tables += 1
}

///|
/// Releases an empty mapping slot.
///
/// A non-empty table is rejected so the arena can never discard the caller's
/// outstanding physical-page release obligations.
pub fn BlockTableArena::release(
  self : BlockTableArena,
  table : BlockTableId,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  if self.page_counts[index] != 0 {
    raise TableNotEmpty(table)
  }
  self.recycle_empty(index)
}

///|
pub fn BlockTableArena::page_count(
  self : BlockTableArena,
  table : BlockTableId,
) -> Int raise BlockTableError {
  let index = self.checked_index(table)
  self.page_counts[index]
}

///|
fn BlockTableArena::checked_append_base(
  self : BlockTableArena,
  table : BlockTableId,
  added : Int,
) -> (Int, Int) raise BlockTableError {
  let index = self.checked_index(table)
  if added <= 0 || added > self.max_pages_per_table {
    raise InvalidPageCount
  }
  let current = self.page_counts[index]
  if added > self.max_pages_per_table - current {
    raise TableFull(table)
  }
  (index, index * self.max_pages_per_table + current)
}

///|
/// Appends one copied physical page identity without temporary collection
/// construction. The caller retains all physical-reference ownership.
pub fn BlockTableArena::append_page(
  self : BlockTableArena,
  table : BlockTableId,
  page : @page_allocator.PageId,
) -> Unit raise BlockTableError {
  let (index, base) = self.checked_append_base(table, 1)
  page_storage_write(self.pages, base, page)
  self.page_counts[index] += 1
}

///|
/// Appends an allocator run directly from its reusable caller-owned buffer.
/// Capacity is proven before mutation, and no intermediate array is created.
pub fn BlockTableArena::append_run(
  self : BlockTableArena,
  table : BlockTableId,
  run : @page_allocator.PageRunBuffer,
) -> Unit raise BlockTableError {
  let added = run.length()
  let (index, base) = self.checked_append_base(table, added)
  for offset in 0..<added {
    page_storage_write(self.pages, base + offset, page_run_read(run, offset))
  }
  self.page_counts[index] += added
}

///|
/// Appends an exact logical page run after proving identity and capacity.
/// Rejection leaves both the table and caller-owned input unchanged.
pub fn BlockTableArena::append_pages(
  self : BlockTableArena,
  table : BlockTableId,
  pages : ArrayView[@page_allocator.PageId],
) -> Unit raise BlockTableError {
  let added = pages.length()
  let (index, base) = self.checked_append_base(table, added)
  for offset, page in pages {
    page_storage_write(self.pages, base + offset, page)
  }
  self.page_counts[index] += added
}

///|
pub fn BlockTableArena::page_at(
  self : BlockTableArena,
  table : BlockTableId,
  logical_page_index : Int,
) -> @page_allocator.PageId raise BlockTableError {
  let index = self.checked_index(table)
  if logical_page_index < 0 || logical_page_index >= self.page_counts[index] {
    raise LogicalPageIndexOutOfBounds(table)
  }
  page_storage_read(
    self.pages,
    index * self.max_pages_per_table + logical_page_index,
  )
}

///|
/// Resolves a token position to its exact logical page, physical ID, and
/// within-page token offset. A mapped page is required.
pub fn BlockTableArena::resolve_token_position(
  self : BlockTableArena,
  table : BlockTableId,
  token_position : Int,
  output : BlockTablePositionBuffer,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  if token_position < 0 {
    raise TokenPositionOutOfBounds(table)
  }
  let logical_page = token_position / self.tokens_per_page
  if logical_page >= self.page_counts[index] {
    raise TokenPositionOutOfBounds(table)
  }
  let page = page_storage_read(
    self.pages,
    index * self.max_pages_per_table + logical_page,
  )
  output.replace(page, logical_page, token_position % self.tokens_per_page)
}

///|
/// Copies the current mapping into reusable storage without changing either
/// physical ownership or table lifecycle. Failure leaves `output` unchanged.
pub fn BlockTableArena::copy_pages(
  self : BlockTableArena,
  table : BlockTableId,
  output : BlockTablePageBuffer,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  let count = self.page_counts[index]
  if output.capacity() < count {
    raise PageBufferTooSmall
  }
  output.replace_from_table(self, index, count)
}

///|
/// Transactionally removes the logical tail `[new_length, page_count)`.
///
/// Removed identities are copied into reusable caller-owned storage in their
/// original order. The caller then owns every corresponding physical active or
/// cached reference transaction. Invalid cuts or insufficient output capacity
/// leave both the table and output unchanged.
pub fn BlockTableArena::truncate(
  self : BlockTableArena,
  table : BlockTableId,
  new_length : Int,
  output : BlockTablePageBuffer,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  let current = self.page_counts[index]
  if new_length < 0 || new_length > current {
    raise InvalidTruncateLength(table)
  }
  let removed = current - new_length
  if output.capacity() < removed {
    raise PageBufferTooSmall
  }
  let base = index * self.max_pages_per_table
  output.replace_from_range(self, base + new_length, removed)
  for offset in new_length..<current {
    page_storage_clear(self.pages, base + offset)
  }
  self.page_counts[index] = new_length
}

///|
/// Atomically drains all page identities into reusable caller storage and
/// resets the allocated table to empty.
///
/// The caller remains responsible for releasing every drained page through
/// the physical allocator. Failure mutates neither table nor output.
pub fn BlockTableArena::reset(
  self : BlockTableArena,
  table : BlockTableId,
  output : BlockTablePageBuffer,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  let count = self.page_counts[index]
  if output.capacity() < count {
    raise PageBufferTooSmall
  }
  output.replace_from_table(self, index, count)
  let base = index * self.max_pages_per_table
  for offset in 0..<count {
    page_storage_clear(self.pages, base + offset)
  }
  self.page_counts[index] = 0
}

///|
/// Drains and releases a table in one metadata transaction.
///
/// On success the returned identities in `output` are still caller-owned
/// physical release obligations, while `table` is no longer valid.
pub fn BlockTableArena::reset_and_release(
  self : BlockTableArena,
  table : BlockTableId,
  output : BlockTablePageBuffer,
) -> Unit raise BlockTableError {
  let index = self.checked_index(table)
  let count = self.page_counts[index]
  if output.capacity() < count {
    raise PageBufferTooSmall
  }
  output.replace_from_table(self, index, count)
  let base = index * self.max_pages_per_table
  for offset in 0..<count {
    page_storage_clear(self.pages, base + offset)
  }
  self.page_counts[index] = 0
  self.recycle_empty(index)
}
