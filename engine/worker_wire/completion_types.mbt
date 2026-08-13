///|
/// Startup-owned reusable storage for one canonical completion frame.
pub struct CompletionFrameBuffer {
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv storage : FixedArray[Byte]
  priv slot_to_entry : FixedArray[Int]
  priv validation_requests : FixedArray[UInt64]
  priv validation_slots : FixedArray[Int]
  priv mut epoch : UInt64
  priv mut length : Int
  priv mut entry_count : Int
}

///|
/// Authenticated immutable view of one exact completion-frame epoch.
#valtype
pub struct ValidatedCompletionFrame(CompletionFrameBuffer, UInt64)

///|
/// Opaque completion-entry view tied to one exact frame epoch.
#valtype
pub struct CompletionFrameEntry(CompletionFrameBuffer, UInt64, Int)

///|
pub fn CompletionFrameBuffer::new(
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
) -> CompletionFrameBuffer raise WorkerWireError {
  let capacity = completion_frame_byte_capacity(worker_limits)
  {
    worker_limits,
    inference_limits,
    storage: FixedArray::make(capacity, b'\x00'),
    slot_to_entry: FixedArray::make(worker_limits.max_completion_slots(), -1),
    validation_requests: FixedArray::make(
      worker_limits.max_completion_slots(),
      0UL,
    ),
    validation_slots: FixedArray::make(worker_limits.max_completion_slots(), -1),
    epoch: 0UL,
    length: 0,
    entry_count: 0,
  }
}

///|
pub fn CompletionFrameBuffer::capacity(self : CompletionFrameBuffer) -> Int {
  self.storage.length()
}

///|
fn ValidatedCompletionFrame::owner(
  self : ValidatedCompletionFrame,
) -> CompletionFrameBuffer raise WorkerWireError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise StaleFrame
  }
  self.0
}

///|
pub fn ValidatedCompletionFrame::length(
  self : ValidatedCompletionFrame,
) -> Int raise WorkerWireError {
  self.owner().length
}

///|
fn CompletionFrameBuffer::clear_slot_index(
  self : CompletionFrameBuffer,
) -> Unit {
  for index in 0..<self.entry_count {
    let slot = read_i32(self.storage, completion_entry_offset(index))
    if slot >= 0 && slot < self.slot_to_entry.length() {
      self.slot_to_entry[slot] = -1
    }
  }
}

///|
fn CompletionFrameBuffer::rebuild_slot_index(
  self : CompletionFrameBuffer,
  count : Int,
) -> Unit {
  for index in 0..<count {
    let slot = read_i32(self.storage, completion_entry_offset(index))
    self.slot_to_entry[slot] = index
  }
  self.entry_count = count
}
