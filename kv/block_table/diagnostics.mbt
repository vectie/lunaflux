///|
fn table_residency_from_code(code : Int) -> BlockTableResidency {
  match code {
    0 => Free
    1 => Allocated
    _ => Retired
  }
}

///|
/// Allocates a bounded metadata copy for diagnostics and tests only.
pub fn BlockTableArena::debug_snapshot(
  self : BlockTableArena,
) -> BlockTableSnapshot {
  let slots : Array[BlockTableSlotSnapshot] = []
  for index in 0..<self.table_capacity {
    slots.push({
      id: BlockTableId(index, self.generations[index]),
      residency: table_residency_from_code(self.states[index]),
      page_count: self.page_counts[index],
    })
  }
  {
    free_tables: self.free_tables,
    allocated_tables: self.allocated_tables,
    retired_tables: self.retired_tables,
    slots: ReadOnlyArray::from_array(slots),
  }
}

///|
pub fn BlockTableSlotSnapshot::id(
  self : BlockTableSlotSnapshot,
) -> BlockTableId {
  self.id
}

///|
pub fn BlockTableSlotSnapshot::residency(
  self : BlockTableSlotSnapshot,
) -> BlockTableResidency {
  self.residency
}

///|
pub fn BlockTableSlotSnapshot::page_count(self : BlockTableSlotSnapshot) -> Int {
  self.page_count
}

///|
pub fn BlockTableSnapshot::free_table_count(self : BlockTableSnapshot) -> Int {
  self.free_tables
}

///|
pub fn BlockTableSnapshot::allocated_table_count(
  self : BlockTableSnapshot,
) -> Int {
  self.allocated_tables
}

///|
pub fn BlockTableSnapshot::retired_table_count(
  self : BlockTableSnapshot,
) -> Int {
  self.retired_tables
}

///|
pub fn BlockTableSnapshot::slot_count(self : BlockTableSnapshot) -> Int {
  self.slots.length()
}

///|
pub fn BlockTableSnapshot::slot_at(
  self : BlockTableSnapshot,
  index : Int,
) -> BlockTableSlotSnapshot raise BlockTableError {
  if index < 0 || index >= self.slots.length() {
    raise SnapshotIndexOutOfBounds
  }
  self.slots[index]
}

///|
/// Checks the free-list topology, lifecycle partition, generations, dense
/// table shapes, and cleared unused page cells. This allocating O(capacity)
/// operation is intentionally limited to diagnostics and tests.
pub fn BlockTableArena::debug_check_invariants(
  self : BlockTableArena,
) -> Unit raise BlockTableError {
  if self.free_tables + self.allocated_tables + self.retired_tables !=
    self.table_capacity {
    raise InvariantViolation(CountPartition)
  }
  if (self.free_tables == 0 && (self.free_head != -1 || self.free_tail != -1)) ||
    (self.free_tables > 0 && (self.free_head == -1 || self.free_tail == -1)) {
    raise InvariantViolation(QueueEndpoint)
  }
  let seen = FixedArray::make(self.table_capacity, false)
  let mut cursor = self.free_head
  let mut count = 0
  let mut last = -1
  while cursor != -1 {
    if cursor < 0 || cursor >= self.table_capacity || seen[cursor] {
      raise InvariantViolation(QueueCycle)
    }
    if self.states[cursor] != TABLE_FREE {
      raise InvariantViolation(QueueMembership)
    }
    seen[cursor] = true
    count += 1
    last = cursor
    cursor = self.free_next[cursor]
  }
  if count != self.free_tables || last != self.free_tail {
    raise InvariantViolation(QueueCount)
  }
  for index in 0..<self.table_capacity {
    let generation = self.generations[index]
    if generation < 0 || generation > self.max_generation {
      raise InvariantViolation(GenerationRange)
    }
    let page_count = self.page_counts[index]
    if page_count < 0 || page_count > self.max_pages_per_table {
      raise InvariantViolation(TableShape)
    }
    match self.states[index] {
      0 =>
        if !seen[index] || page_count != 0 {
          raise InvariantViolation(TableShape)
        }
      1 => if seen[index] { raise InvariantViolation(TableShape) }
      2 =>
        if seen[index] || page_count != 0 || generation != self.max_generation {
          raise InvariantViolation(TableShape)
        }
      _ => raise InvariantViolation(LifecycleState)
    }
    let base = index * self.max_pages_per_table
    for offset in 0..<self.max_pages_per_table {
      let occupied = page_storage_is_occupied(self.pages, base + offset)
      if occupied !=
        (self.states[index] == TABLE_ALLOCATED && offset < page_count) {
        raise InvariantViolation(ClearedCells)
      }
    }
  }
}
