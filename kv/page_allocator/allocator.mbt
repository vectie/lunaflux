///|
const PAGE_FREE : Int = 0

///|
const PAGE_ALLOCATED : Int = 1

///|
const PAGE_RETIRED : Int = 2

///|
/// Preallocates every metadata array and initializes a deterministic FIFO of
/// physical page indices in ascending order.
pub fn PageAllocator::new(limits : PageAllocatorLimits) -> PageAllocator {
  let capacity = limits.page_capacity()
  let free_next = FixedArray::make(capacity, -1)
  for index in 0..<capacity {
    free_next[index] = if index + 1 < capacity { index + 1 } else { -1 }
  }
  {
    limits,
    page_capacity: capacity,
    max_run_pages: limits.max_run_pages(),
    max_active_references_per_page: limits.max_active_references_per_page(),
    max_cached_references_per_page: limits.max_cached_references_per_page(),
    max_generation: limits.max_generation(),
    generations: FixedArray::make(capacity, 0),
    active_references: FixedArray::make(capacity, 0),
    cached_references: FixedArray::make(capacity, 0),
    residency: FixedArray::make(capacity, PAGE_FREE),
    free_next,
    free_head: 0,
    free_tail: capacity - 1,
    free_pages: capacity,
    allocated_pages: 0,
    retired_pages: 0,
    allocation_journal: FixedArray::make(capacity, -1),
    allocation_journal_generations: FixedArray::make(capacity, 0),
    allocation_journal_next: FixedArray::make(capacity, -1),
    allocation_journal_length: 0,
    allocation_checkpoint_open: false,
    allocation_checkpoint_epoch: 0UL,
    checkpoint_free_head: -1,
    checkpoint_free_tail: -1,
    checkpoint_free_pages: 0,
    checkpoint_allocated_pages: 0,
    checkpoint_retired_pages: 0,
  }
}

///|
pub fn PageAllocator::limits(self : PageAllocator) -> PageAllocatorLimits {
  self.limits
}

///|
pub fn PageAllocator::free_page_count(self : PageAllocator) -> Int {
  self.free_pages
}

///|
pub fn PageAllocator::allocated_page_count(self : PageAllocator) -> Int {
  self.allocated_pages
}

///|
pub fn PageAllocator::retired_page_count(self : PageAllocator) -> Int {
  self.retired_pages
}

///|
fn PageAllocator::pop_free(self : PageAllocator) -> PageId {
  let index = self.free_head
  let next = self.free_next[index]
  if self.allocation_checkpoint_open {
    let journal_index = self.allocation_journal_length
    self.allocation_journal[journal_index] = index
    self.allocation_journal_generations[journal_index] = self.generations[index]
    self.allocation_journal_next[journal_index] = next
    self.allocation_journal_length = journal_index + 1
  }
  self.free_head = next
  if next == -1 {
    self.free_tail = -1
  }
  self.free_next[index] = -1
  self.free_pages -= 1
  self.allocated_pages += 1
  self.residency[index] = PAGE_ALLOCATED
  self.generations[index] += 1
  self.active_references[index] = 1
  self.cached_references[index] = 0
  PageId(index, self.generations[index])
}

///|
fn PageAllocator::preflight_allocate(
  self : PageAllocator,
  count : Int,
) -> Unit raise PageAllocatorError {
  if self.allocation_checkpoint_open &&
    self.allocation_journal_length + count > self.allocation_journal.length() {
    raise InvariantViolation(CountPartition)
  }
  let mut cursor = self.free_head
  for _ in 0..<count {
    if cursor < 0 || cursor >= self.page_capacity {
      raise InvariantViolation(QueueEndpoint)
    }
    if self.residency[cursor] != PAGE_FREE ||
      self.active_references[cursor] != 0 ||
      self.cached_references[cursor] != 0 {
      raise InvariantViolation(QueueMembership)
    }
    if self.generations[cursor] >= self.max_generation {
      raise IdentityOverflow(PageGenerationIdentity)
    }
    cursor = self.free_next[cursor]
  }
}

///|
/// Allocates the oldest free page with one active-request reference.
pub fn PageAllocator::allocate_one(
  self : PageAllocator,
) -> PageId raise PageAllocatorError {
  if self.free_pages == 0 {
    raise Exhausted
  }
  self.preflight_allocate(1)
  self.pop_free()
}

///|
/// Allocates exactly `count` FIFO pages into reusable caller-owned storage.
///
/// Capacity is proven before mutation. On any rejected request, allocator and
/// output buffer remain unchanged; success performs no collection growth.
pub fn PageAllocator::allocate_run(
  self : PageAllocator,
  count : Int,
  output : PageRunBuffer,
) -> Unit raise PageAllocatorError {
  if count <= 0 || count > self.max_run_pages {
    raise InvalidRunLength
  }
  if output.capacity() < count {
    raise RunBufferTooSmall
  }
  if self.free_pages < count {
    raise Exhausted
  }
  self.preflight_allocate(count)
  for offset in 0..<count {
    output.pages[offset] = self.pop_free()
  }
  output.length = count
}

///|
fn PageAllocator::require_reference_mutation_allowed(
  self : PageAllocator,
) -> Unit raise PageAllocatorError {
  if self.allocation_checkpoint_open {
    raise AllocationCheckpointReferenceMutationRejected
  }
}

///|
fn PageAllocator::checked_index(
  self : PageAllocator,
  page : PageId,
) -> Int raise PageAllocatorError {
  let index = page.index()
  if index < 0 ||
    index >= self.page_capacity ||
    self.generations[index] != page.generation() {
    raise StalePage(page)
  }
  if self.residency[index] != PAGE_ALLOCATED {
    raise PageNotAllocated(page)
  }
  index
}

///|
fn PageAllocator::enqueue_free(self : PageAllocator, index : Int) -> Unit {
  self.free_next[index] = -1
  if self.free_tail == -1 {
    self.free_head = index
    self.free_tail = index
  } else {
    self.free_next[self.free_tail] = index
    self.free_tail = index
  }
  self.free_pages += 1
}

///|
fn PageAllocator::recycle_if_unowned(self : PageAllocator, index : Int) -> Unit {
  if self.active_references[index] != 0 || self.cached_references[index] != 0 {
    return
  }
  self.allocated_pages -= 1
  if self.generations[index] == self.max_generation {
    self.residency[index] = PAGE_RETIRED
    self.retired_pages += 1
  } else {
    self.residency[index] = PAGE_FREE
    self.enqueue_free(index)
  }
}

///|
/// Adds one active-request owner, such as another request sharing cached KV.
pub fn PageAllocator::retain_active(
  self : PageAllocator,
  page : PageId,
) -> Unit raise PageAllocatorError {
  self.require_reference_mutation_allowed()
  let index = self.checked_index(page)
  if self.active_references[index] == self.max_active_references_per_page {
    raise ReferenceOverflow(page, Active)
  }
  self.active_references[index] += 1
}

///|
/// Releases one active-request owner.
pub fn PageAllocator::release_active(
  self : PageAllocator,
  page : PageId,
) -> Unit raise PageAllocatorError {
  self.require_reference_mutation_allowed()
  let index = self.checked_index(page)
  if self.active_references[index] == 0 {
    raise ReferenceUnderflow(page, Active)
  }
  self.active_references[index] -= 1
  self.recycle_if_unowned(index)
}

///|
/// Adds one cached-prefix owner.
///
/// Call this before releasing the last active reference when publishing a
/// request-private page into a prefix cache.
pub fn PageAllocator::retain_cached(
  self : PageAllocator,
  page : PageId,
) -> Unit raise PageAllocatorError {
  self.require_reference_mutation_allowed()
  let index = self.checked_index(page)
  if self.cached_references[index] == self.max_cached_references_per_page {
    raise ReferenceOverflow(page, Cached)
  }
  self.cached_references[index] += 1
}

///|
/// Releases one cached-prefix owner, typically during deterministic eviction.
pub fn PageAllocator::release_cached(
  self : PageAllocator,
  page : PageId,
) -> Unit raise PageAllocatorError {
  self.require_reference_mutation_allowed()
  let index = self.checked_index(page)
  if self.cached_references[index] == 0 {
    raise ReferenceUnderflow(page, Cached)
  }
  self.cached_references[index] -= 1
  self.recycle_if_unowned(index)
}

///|
/// Returns the current active-request reference count for a checked identity.
pub fn PageAllocator::active_reference_count(
  self : PageAllocator,
  page : PageId,
) -> Int raise PageAllocatorError {
  let index = self.checked_index(page)
  self.active_references[index]
}

///|
/// Authenticates that `page` is the allocator's current generation and still
/// has at least one active-request owner.
///
/// This read-only check is valid while an allocation checkpoint is open. Its
/// successful path performs only fixed-array reads and does not allocate.
pub fn PageAllocator::validate_active(
  self : PageAllocator,
  page : PageId,
) -> Unit raise PageAllocatorError {
  let index = self.checked_index(page)
  if self.active_references[index] <= 0 {
    raise PageNotActive(page)
  }
}

///|
/// Returns the current cached-prefix reference count for a checked identity.
pub fn PageAllocator::cached_reference_count(
  self : PageAllocator,
  page : PageId,
) -> Int raise PageAllocatorError {
  let index = self.checked_index(page)
  self.cached_references[index]
}
