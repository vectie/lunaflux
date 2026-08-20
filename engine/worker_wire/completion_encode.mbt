///|
fn validate_completion_entry_for_encode(
  owner : CompletionFrameBuffer,
  entry : @worker_protocol.SubmittedCompletionEntry,
) -> Unit raise WorkerWireError {
  let slot = entry.slot() catch { error => raise Protocol(error) }
  let request = entry.request_id() catch { error => raise Protocol(error) }
  ignore(entry.request_generation() catch { error => raise Protocol(error) })
  let kind = entry.kind() catch { error => raise Protocol(error) }
  if slot.value() < 0 ||
    slot.value() >= owner.worker_limits.max_completion_slots() ||
    request.value() == 0UL {
    raise InvalidFrame(CompletionEntry)
  }
  match kind {
    PrefillCompletion | FinalPrefillCompletion => {
      let processed = entry.processed_tokens() catch {
        error => raise Protocol(error)
      }
      if processed <= 0 || processed > owner.worker_limits.max_plan_tokens() {
        raise InvalidFrame(CompletionEntry)
      }
    }
    _ => ()
  }
  match kind {
    FinalPrefillCompletion | DecodeCompletion => {
      let token = entry.token_id() catch { error => raise Protocol(error) }
      let max_token = if owner.worker_limits.max_token_id() <
        owner.inference_limits.max_token_id() {
        owner.worker_limits.max_token_id()
      } else {
        owner.inference_limits.max_token_id()
      }
      if token < 0 || token > max_token {
        raise InvalidFrame(CompletionEntry)
      }
    }
    FailedCompletion =>
      ignore(entry.failure() catch { error => raise Protocol(error) })
    PrefillCompletion => ()
  }
}

///|
fn write_completion_header(
  bytes : FixedArray[Byte],
  length : Int,
  sequence : UInt64,
  generation : UInt64,
  count : Int,
) -> Unit {
  write_u32(bytes, 0, completion_magic)
  write_u32(bytes, 4, wire_version)
  write_u32(bytes, 8, completion_kind)
  write_i32(bytes, 12, length)
  write_u64(bytes, 16, sequence)
  write_u64(bytes, 24, generation)
  write_i32(bytes, 32, count)
  write_u32(bytes, 36, 0)
  write_u32(bytes, completion_checksum_offset, 0)
  write_u32(bytes, 44, 0)
}

///|
fn write_completion_entry(
  bytes : FixedArray[Byte],
  offset : Int,
  entry : @worker_protocol.SubmittedCompletionEntry,
) -> Unit raise WorkerWireError {
  let slot = entry.slot() catch { error => raise Protocol(error) }
  let request = entry.request_id() catch { error => raise Protocol(error) }
  let generation = entry.request_generation() catch {
    error => raise Protocol(error)
  }
  let kind = entry.kind() catch { error => raise Protocol(error) }
  let mut processed = 0
  let mut token = 0
  let mut failure : UInt = 0
  match kind {
    PrefillCompletion | FinalPrefillCompletion =>
      processed = entry.processed_tokens() catch {
        error => raise Protocol(error)
      }
    _ => ()
  }
  match kind {
    FinalPrefillCompletion | DecodeCompletion =>
      token = entry.token_id() catch { error => raise Protocol(error) }
    FailedCompletion =>
      failure = worker_failure_to_wire(
        entry.failure() catch {
          error => raise Protocol(error)
        },
      )
    PrefillCompletion => ()
  }
  write_i32(bytes, offset, slot.value())
  write_u32(bytes, offset + 4, completion_kind_to_wire(kind))
  write_u64(bytes, offset + 8, request.value())
  write_u64(bytes, offset + 16, generation.value())
  write_i32(bytes, offset + 24, processed)
  write_i32(bytes, offset + 28, token)
  write_u32(bytes, offset + 32, failure)
  write_u32(bytes, offset + 36, 0)
}

///|
/// Encodes one exact submitted completion into startup-owned fixed storage.
pub fn CompletionFrameBuffer::encode_completion(
  self : CompletionFrameBuffer,
  completion : @worker_protocol.SubmittedCompletion,
) -> ValidatedCompletionFrame raise WorkerWireError {
  self.require_no_writer()
  let sequence = completion.sequence_wire() catch {
    error => raise Protocol(error)
  }
  let generation = completion.model_plan_generation_wire() catch {
    error => raise Protocol(error)
  }
  let count = completion.entry_count() catch { error => raise Protocol(error) }
  if sequence == 0UL ||
    generation == 0UL ||
    count <= 0 ||
    count > self.worker_limits.max_completion_slots() {
    raise InvalidFrame(Counts)
  }
  let length = exact_completion_frame_bytes(count)
  if length > self.storage.length() {
    raise InvalidLimit(CompletionFrameBytes)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  for index in 0..<count {
    validate_completion_entry_for_encode(
      self,
      completion.entry(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  self.clear_slot_index()
  write_completion_header(self.storage, length, sequence, generation, count)
  for index in 0..<count {
    write_completion_entry(
      self.storage,
      completion_entry_offset(index),
      completion.entry(index) catch {
        error => raise Protocol(error)
      },
    )
  }
  write_u32(
    self.storage,
    completion_checksum_offset,
    completion_frame_checksum(self.storage, length),
  )
  self.epoch += 1UL
  self.length = length
  self.rebuild_slot_index(count)
  ValidatedCompletionFrame(self, self.epoch)
}
