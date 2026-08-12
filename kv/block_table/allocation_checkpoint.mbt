///|
fn BlockTableArena::require_checkpoint_closed(
  self : BlockTableArena,
) -> Unit raise BlockTableError {
  if self.allocation_checkpoint_open {
    raise AllocationCheckpointMutationRejected
  }
}

///|
/// Opens one exclusive reversible table-allocation transaction.
///
/// The arena journals only tables first touched by an append, using fixed
/// startup storage. Appends may continue while open, but must be detached back
/// to the recorded baseline before rollback. Release, truncate, and reset
/// operations are rejected because they change lifecycle or discard it.
pub fn BlockTableArena::checkpoint(
  self : BlockTableArena,
) -> BlockTableAllocationCheckpoint raise BlockTableError {
  if self.allocation_checkpoint_open {
    raise AllocationCheckpointAlreadyOpen
  }
  if self.allocation_checkpoint_epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(AllocationCheckpointEpochIdentity)
  }
  self.allocation_checkpoint_epoch = self.allocation_checkpoint_epoch + 1UL
  self.allocation_journal_length = 0
  self.mapping_journal_length = 0
  self.checkpoint_free_head = self.free_head
  self.checkpoint_free_tail = self.free_tail
  self.checkpoint_free_tables = self.free_tables
  self.checkpoint_allocated_tables = self.allocated_tables
  self.checkpoint_retired_tables = self.retired_tables
  self.allocation_checkpoint_open = true
  BlockTableAllocationCheckpoint(self, self.allocation_checkpoint_epoch)
}

///|
fn BlockTableAllocationCheckpoint::authenticate_owner(
  self : BlockTableAllocationCheckpoint,
  owner : BlockTableArena,
) -> Unit raise BlockTableError {
  if !physical_equal(self.0, owner) {
    raise BlockTableAllocationCheckpointMismatch
  }
  if self.1 != owner.allocation_checkpoint_epoch ||
    !owner.allocation_checkpoint_open {
    raise StaleBlockTableAllocationCheckpointEpoch
  }
}

///|
/// Clears every mapping appended since this checkpoint's exact baseline.
///
/// This intentionally returns no page identities. The scheduler must already
/// own them in its page-allocation journal, call this detach before table
/// rollback, and then roll back physical page allocation. Passing a baseline
/// other than the checkpoint's recorded count is rejected.
pub fn BlockTableArena::detach_suffix_for_checkpoint(
  self : BlockTableArena,
  checkpoint : BlockTableAllocationCheckpoint,
  table : BlockTableId,
  old_page_count : Int,
) -> Unit raise BlockTableError {
  checkpoint.authenticate_owner(self)
  let index = self.checked_index(table)
  let known = self.mapping_touched[index]
  let baseline = self.mapping_journal_baselines[index]
  let current = self.page_counts[index]
  if !known ||
    old_page_count != baseline ||
    old_page_count < 0 ||
    old_page_count > current {
    raise InvalidCheckpointDetach(table)
  }
  let base = index * self.max_pages_per_table
  for offset in old_page_count..<current {
    page_storage_clear(self.pages, base + offset)
  }
  self.page_counts[index] = old_page_count
}

///|
fn BlockTableArena::preflight_checkpoint_rollback(
  self : BlockTableArena,
) -> Unit raise BlockTableError {
  let length = self.allocation_journal_length
  if length < 0 ||
    length > self.allocation_journal.length() ||
    self.checkpoint_free_tables - length != self.free_tables ||
    self.checkpoint_allocated_tables + length != self.allocated_tables ||
    self.checkpoint_retired_tables != self.retired_tables {
    raise BlockTableAllocationCheckpointMismatch
  }
  if length == 0 {
    if self.free_head != self.checkpoint_free_head ||
      self.free_tail != self.checkpoint_free_tail {
      raise BlockTableAllocationCheckpointMismatch
    }
  } else if self.allocation_journal[0] != self.checkpoint_free_head ||
    (self.free_tables == 0 && self.free_tail != -1) ||
    (self.free_tables > 0 && self.free_tail != self.checkpoint_free_tail) {
    raise BlockTableAllocationCheckpointMismatch
  }
  let mapping_length = self.mapping_journal_length
  if mapping_length < 0 ||
    mapping_length > self.mapping_journal_indices.length() {
    raise BlockTableAllocationCheckpointMismatch
  }
  for journal_index in 0..<mapping_length {
    let index = self.mapping_journal_indices[journal_index]
    if index < 0 ||
      index >= self.table_capacity ||
      !self.mapping_touched[index] ||
      self.states[index] != TABLE_ALLOCATED ||
      self.page_counts[index] != self.mapping_journal_baselines[index] {
      raise BlockTableAllocationCheckpointMismatch
    }
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
      index >= self.table_capacity ||
      previous_generation < 0 ||
      previous_generation >= self.max_generation ||
      previous_next != expected_next ||
      !self.provisional_tables[index] ||
      self.generations[index] != previous_generation + 1 ||
      self.states[index] != TABLE_ALLOCATED ||
      self.page_counts[index] != 0 ||
      self.free_next[index] != -1 {
      raise BlockTableAllocationCheckpointMismatch
    }
  }
}

///|
/// Keeps checkpoint allocations and appended mappings, then consumes the
/// authenticated handle.
pub fn BlockTableArena::commit_checkpoint(
  self : BlockTableArena,
  checkpoint : BlockTableAllocationCheckpoint,
) -> Unit raise BlockTableError {
  checkpoint.authenticate_owner(self)
  for journal_index in 0..<self.mapping_journal_length {
    self.mapping_touched[self.mapping_journal_indices[journal_index]] = false
  }
  for journal_index in 0..<self.allocation_journal_length {
    self.provisional_tables[self.allocation_journal[journal_index]] = false
  }
  self.allocation_journal_length = 0
  self.mapping_journal_length = 0
  self.allocation_checkpoint_open = false
}

///|
/// Restores exact allocation identities and free-FIFO state after every
/// appended mapping has been detached to its checkpoint baseline.
pub fn BlockTableArena::rollback_to(
  self : BlockTableArena,
  checkpoint : BlockTableAllocationCheckpoint,
) -> Unit raise BlockTableError {
  checkpoint.authenticate_owner(self)
  self.preflight_checkpoint_rollback()
  let length = self.allocation_journal_length
  for journal_index in 0..<self.mapping_journal_length {
    self.mapping_touched[self.mapping_journal_indices[journal_index]] = false
  }
  for journal_index in 0..<length {
    let index = self.allocation_journal[journal_index]
    self.generations[index] = self.allocation_journal_generations[journal_index]
    self.states[index] = TABLE_FREE
    self.page_counts[index] = 0
    self.free_next[index] = self.allocation_journal_next[journal_index]
    self.provisional_tables[index] = false
  }
  self.free_head = self.checkpoint_free_head
  self.free_tail = self.checkpoint_free_tail
  self.free_tables = self.checkpoint_free_tables
  self.allocated_tables = self.checkpoint_allocated_tables
  self.retired_tables = self.checkpoint_retired_tables
  self.allocation_journal_length = 0
  self.mapping_journal_length = 0
  self.allocation_checkpoint_open = false
}
