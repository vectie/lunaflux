///|
/// Opens one exclusive provisional-allocation transaction.
///
/// While it is open, page allocation remains available and is journaled.
/// Reference-count mutation is rejected so provisional identities cannot be
/// retained, published into a cache, released, or recycled before commit.
pub fn PageAllocator::checkpoint(
  self : PageAllocator,
) -> PageAllocationCheckpoint raise PageAllocatorError {
  if self.allocation_checkpoint_open {
    raise AllocationCheckpointAlreadyOpen
  }
  if self.allocation_checkpoint_epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(AllocationCheckpointEpochIdentity)
  }
  self.allocation_checkpoint_epoch = self.allocation_checkpoint_epoch + 1UL
  self.allocation_journal_length = 0
  self.checkpoint_free_head = self.free_head
  self.checkpoint_free_tail = self.free_tail
  self.checkpoint_free_pages = self.free_pages
  self.checkpoint_allocated_pages = self.allocated_pages
  self.checkpoint_retired_pages = self.retired_pages
  self.allocation_checkpoint_open = true
  PageAllocationCheckpoint(self, self.allocation_checkpoint_epoch)
}

///|
fn PageAllocationCheckpoint::authenticate_owner(
  self : PageAllocationCheckpoint,
  owner : PageAllocator,
) -> Unit raise PageAllocatorError {
  if !physical_equal(self.0, owner) {
    raise PageAllocationCheckpointMismatch
  }
  if self.1 != owner.allocation_checkpoint_epoch ||
    !owner.allocation_checkpoint_open {
    raise StalePageAllocationCheckpointEpoch
  }
}

///|
fn PageAllocator::preflight_checkpoint_rollback(
  self : PageAllocator,
) -> Unit raise PageAllocatorError {
  let length = self.allocation_journal_length
  if length < 0 ||
    length > self.allocation_journal.length() ||
    self.checkpoint_free_pages - length != self.free_pages ||
    self.checkpoint_allocated_pages + length != self.allocated_pages ||
    self.checkpoint_retired_pages != self.retired_pages {
    raise PageAllocationCheckpointMismatch
  }
  if length == 0 {
    if self.free_head != self.checkpoint_free_head ||
      self.free_tail != self.checkpoint_free_tail {
      raise PageAllocationCheckpointMismatch
    }
  } else if self.allocation_journal[0] != self.checkpoint_free_head ||
    (self.free_pages == 0 && self.free_tail != -1) ||
    (self.free_pages > 0 && self.free_tail != self.checkpoint_free_tail) {
    raise PageAllocationCheckpointMismatch
  }
  for journal_index in 0..<length {
    let index = self.allocation_journal[journal_index]
    let previous_generation = self.allocation_journal_generations[journal_index]
    let previous_next = self.allocation_journal_next[journal_index]
    let expected_next = if journal_index + 1 < length {
      self.allocation_journal[journal_index + 1]
    } else {
      self.free_head
    }
    if index < 0 ||
      index >= self.page_capacity ||
      previous_generation < 0 ||
      previous_generation >= self.max_generation ||
      previous_next != expected_next ||
      self.generations[index] != previous_generation + 1 ||
      self.residency[index] != PAGE_ALLOCATED ||
      self.active_references[index] != 1 ||
      self.cached_references[index] != 0 ||
      self.free_next[index] != -1 {
      raise PageAllocationCheckpointMismatch
    }
  }
}

///|
/// Makes every allocation since `checkpoint` permanent and consumes the
/// handle. This mutates only bounded scalar transaction metadata.
pub fn PageAllocator::commit_checkpoint(
  self : PageAllocator,
  checkpoint : PageAllocationCheckpoint,
) -> Unit raise PageAllocatorError {
  checkpoint.authenticate_owner(self)
  self.allocation_journal_length = 0
  self.allocation_checkpoint_open = false
}

///|
/// Atomically restores the exact free FIFO, counters, page generations,
/// residency, and reference counts captured by `checkpoint`.
///
/// All journal entries are validated before mutation. A successful rollback
/// consumes the handle and performs no allocation. The caller must first
/// detach every copy of a provisional identity from external block tables or
/// journals; rollback authenticates allocator ownership, not external copies.
pub fn PageAllocator::rollback_to(
  self : PageAllocator,
  checkpoint : PageAllocationCheckpoint,
) -> Unit raise PageAllocatorError {
  checkpoint.authenticate_owner(self)
  self.preflight_checkpoint_rollback()
  let length = self.allocation_journal_length
  for journal_index in 0..<length {
    let index = self.allocation_journal[journal_index]
    self.generations[index] = self.allocation_journal_generations[journal_index]
    self.active_references[index] = 0
    self.cached_references[index] = 0
    self.residency[index] = PAGE_FREE
    self.free_next[index] = self.allocation_journal_next[journal_index]
  }
  self.free_head = self.checkpoint_free_head
  self.free_tail = self.checkpoint_free_tail
  self.free_pages = self.checkpoint_free_pages
  self.allocated_pages = self.checkpoint_allocated_pages
  self.retired_pages = self.checkpoint_retired_pages
  self.allocation_journal_length = 0
  self.allocation_checkpoint_open = false
}
