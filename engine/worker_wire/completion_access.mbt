///|
pub fn ValidatedCompletionFrame::sequence_value(
  self : ValidatedCompletionFrame,
) -> UInt64 raise WorkerWireError {
  let owner = self.owner()
  read_u64(owner.storage, 16)
}

///|
pub fn ValidatedCompletionFrame::model_generation_value(
  self : ValidatedCompletionFrame,
) -> UInt64 raise WorkerWireError {
  let owner = self.owner()
  read_u64(owner.storage, 24)
}

///|
pub fn ValidatedCompletionFrame::entry_count(
  self : ValidatedCompletionFrame,
) -> Int raise WorkerWireError {
  self.owner().entry_count
}

///|
pub fn ValidatedCompletionFrame::entry(
  self : ValidatedCompletionFrame,
  index : Int,
) -> CompletionFrameEntry raise WorkerWireError {
  let owner = self.owner()
  if index < 0 || index >= owner.entry_count {
    raise InvalidFrame(CompletionEntry)
  }
  CompletionFrameEntry(owner, self.1, index)
}

///|
pub fn ValidatedCompletionFrame::entry_for_slot(
  self : ValidatedCompletionFrame,
  slot : Int,
) -> CompletionFrameEntry raise WorkerWireError {
  let owner = self.owner()
  if slot < 0 || slot >= owner.slot_to_entry.length() {
    raise InvalidFrame(CompletionSlot)
  }
  let index = owner.slot_to_entry[slot]
  if index < 0 || index >= owner.entry_count {
    raise InvalidFrame(CompletionSlot)
  }
  CompletionFrameEntry(owner, self.1, index)
}

///|
fn CompletionFrameEntry::offset(
  self : CompletionFrameEntry,
) -> Int raise WorkerWireError {
  ignore(ValidatedCompletionFrame(self.0, self.1).owner())
  if self.2 < 0 || self.2 >= self.0.entry_count {
    raise InvalidFrame(CompletionEntry)
  }
  completion_entry_offset(self.2)
}

///|
pub fn CompletionFrameEntry::slot(
  self : CompletionFrameEntry,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset())
}

///|
pub fn CompletionFrameEntry::kind(
  self : CompletionFrameEntry,
) -> @worker_protocol.CompletionEntryKind raise WorkerWireError {
  completion_kind_from_wire(read_u32(self.0.storage, self.offset() + 4))
}

///|
pub fn CompletionFrameEntry::request_id_value(
  self : CompletionFrameEntry,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 8)
}

///|
pub fn CompletionFrameEntry::request_generation_value(
  self : CompletionFrameEntry,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 16)
}

///|
pub fn CompletionFrameEntry::processed_tokens(
  self : CompletionFrameEntry,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 24)
}

///|
pub fn CompletionFrameEntry::token_id(
  self : CompletionFrameEntry,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 28)
}

///|
pub fn CompletionFrameEntry::failure(
  self : CompletionFrameEntry,
) -> @worker_protocol.WorkerFailure raise WorkerWireError {
  let offset = self.offset()
  if completion_kind_from_wire(read_u32(self.0.storage, offset + 4)) !=
    FailedCompletion {
    raise InvalidFrame(CompletionEntry)
  }
  worker_failure_from_wire(read_u32(self.0.storage, offset + 32))
}
