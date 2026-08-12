///|
fn residency_from_code(code : Int) -> PageResidency {
  match code {
    0 => Free
    1 => Allocated
    _ => Retired
  }
}

///|
/// Allocates an immutable metadata copy for startup diagnostics and tests.
/// Never call this from a scheduling token step.
pub fn PageAllocator::debug_snapshot(
  self : PageAllocator,
) -> PageAllocatorSnapshot {
  let pages : Array[PageSnapshot] = []
  for index in 0..<self.page_capacity {
    pages.push({
      id: PageId(index, self.generations[index]),
      residency: residency_from_code(self.residency[index]),
      active_references: self.active_references[index],
      cached_references: self.cached_references[index],
    })
  }
  {
    free_pages: self.free_pages,
    allocated_pages: self.allocated_pages,
    retired_pages: self.retired_pages,
    pages: ReadOnlyArray::from_array(pages),
  }
}

///|
pub fn PageSnapshot::id(self : PageSnapshot) -> PageId {
  self.id
}

///|
pub fn PageSnapshot::residency(self : PageSnapshot) -> PageResidency {
  self.residency
}

///|
pub fn PageSnapshot::active_references(self : PageSnapshot) -> Int {
  self.active_references
}

///|
pub fn PageSnapshot::cached_references(self : PageSnapshot) -> Int {
  self.cached_references
}

///|
pub fn PageAllocatorSnapshot::free_page_count(
  self : PageAllocatorSnapshot,
) -> Int {
  self.free_pages
}

///|
pub fn PageAllocatorSnapshot::allocated_page_count(
  self : PageAllocatorSnapshot,
) -> Int {
  self.allocated_pages
}

///|
pub fn PageAllocatorSnapshot::retired_page_count(
  self : PageAllocatorSnapshot,
) -> Int {
  self.retired_pages
}

///|
pub fn PageAllocatorSnapshot::page_count(self : PageAllocatorSnapshot) -> Int {
  self.pages.length()
}

///|
pub fn PageAllocatorSnapshot::page_at(
  self : PageAllocatorSnapshot,
  index : Int,
) -> PageSnapshot raise PageAllocatorError {
  if index < 0 || index >= self.pages.length() {
    raise RunIndexOutOfBounds
  }
  self.pages[index]
}

///|
/// Checks queue topology, count partitioning, generations, and reference-state
/// coherence. This explicit startup/test operation is linear in page capacity.
pub fn PageAllocator::debug_check_invariants(
  self : PageAllocator,
) -> Unit raise PageAllocatorError {
  let capacity = self.page_capacity
  if self.free_pages + self.allocated_pages + self.retired_pages != capacity {
    raise InvariantViolation(CountPartition)
  }
  if (self.free_pages == 0 && (self.free_head != -1 || self.free_tail != -1)) ||
    (self.free_pages > 0 && (self.free_head == -1 || self.free_tail == -1)) {
    raise InvariantViolation(QueueEndpoint)
  }
  let seen = FixedArray::make(capacity, false)
  let mut cursor = self.free_head
  let mut queue_count = 0
  let mut last = -1
  while cursor != -1 {
    if cursor < 0 || cursor >= capacity || seen[cursor] {
      raise InvariantViolation(QueueCycle)
    }
    if self.residency[cursor] != PAGE_FREE {
      raise InvariantViolation(QueueMembership)
    }
    seen[cursor] = true
    queue_count += 1
    last = cursor
    cursor = self.free_next[cursor]
  }
  if queue_count != self.free_pages || last != self.free_tail {
    raise InvariantViolation(QueueCount)
  }
  for index in 0..<capacity {
    let generation = self.generations[index]
    if generation < 0 || generation > self.max_generation {
      raise InvariantViolation(GenerationRange)
    }
    match self.residency[index] {
      0 =>
        if !seen[index] ||
          self.active_references[index] != 0 ||
          self.cached_references[index] != 0 {
          raise InvariantViolation(FreeReferences)
        }
      1 =>
        if seen[index] ||
          (
            self.active_references[index] == 0 &&
            self.cached_references[index] == 0
          ) {
          raise InvariantViolation(AllocatedReferences)
        }
      _ =>
        if seen[index] ||
          generation != self.max_generation ||
          self.active_references[index] != 0 ||
          self.cached_references[index] != 0 {
          raise InvariantViolation(RetiredReferences)
        }
    }
  }
}
