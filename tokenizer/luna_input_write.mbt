///|
pub fn LunaTokenizerWorker::begin_luna_input(
  self : LunaTokenizerWorker,
  byte_count~ : Int,
  special_policy~ : SpecialTokenPolicy,
  output_limit~ : Int,
  truncation~ : TruncationPolicy,
) -> LunaTokenizerInputWrite raise TokenizerError {
  if self.phase != LUNA_WORKER_IDLE {
    raise LunaTokenizerWorkerBusy
  }
  let maximum_output = Int::min(
    self.output_capacity,
    self.spec.limits.max_output_tokens,
  )
  if output_limit <= 0 || output_limit > maximum_output {
    raise InvalidOutputLimit(requested=output_limit, maximum=maximum_output)
  }
  if byte_count < 0 {
    raise LunaInvalidInputWrite
  }
  let maximum_input = Int::min(
    self.input_capacity,
    self.spec.limits.max_input_bytes,
  )
  if byte_count > maximum_input {
    raise InputTooLong(actual=byte_count, maximum=maximum_input)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaTokenizerWorkerEpochExhausted
  }
  self.epoch += 1UL
  self.input_length = byte_count
  self.written_input_bytes = 0
  self.ordinary_ready = false
  self.segment_start = true
  self.post_processor_pending = self.spec.post_processor_token_id >= 0
  self.scalar_is_meta = false
  self.scalar_direct_token_id = -1
  self.scalar_byte_offset = 0
  self.scalar_byte_length = 0
  self.scalar_source_advance = 0
  self.normalized_length = 0
  self.qwen_segment_end = -1
  self.special_policy = special_policy
  self.truncation = truncation
  self.output_limit = output_limit
  self.input_offset = 0
  self.node_count = 0
  self.head = -1
  self.tail = -1
  self.special_index = 0
  self.special_piece_offset = 0
  self.special_matches = true
  self.best_special_id = -1
  self.best_special_bytes = 0
  self.scan_left = -1
  self.lookup_low = 0
  self.lookup_high = 0
  self.selected_rank = -1
  self.compact_node = -1
  self.compact_actual = 0
  self.output_count = 0
  self.truncated = false
  self.last_work_units = 0
  self.phase = LUNA_WORKER_INPUT_WRITE
  { worker: self, epoch: self.epoch }
}

///|
fn LunaTokenizerInputWrite::require_worker(
  self : LunaTokenizerInputWrite,
) -> LunaTokenizerWorker raise TokenizerError {
  if self.epoch == 0UL ||
    self.worker.epoch != self.epoch ||
    self.worker.phase != LUNA_WORKER_INPUT_WRITE {
    raise LunaTokenizerWorkerStale
  }
  self.worker
}

///|
/// Copies exactly one byte into worker-owned startup storage.
pub fn LunaTokenizerInputWrite::push_byte(
  self : LunaTokenizerInputWrite,
  byte : Byte,
) -> Unit raise TokenizerError {
  let worker = self.require_worker()
  if worker.written_input_bytes >= worker.input_length {
    raise LunaInvalidInputWrite
  }
  worker.input[worker.written_input_bytes] = byte
  worker.written_input_bytes += 1
}

///|
/// Transfers exact-generation authority to the cooperative BPE state machine.
pub fn LunaTokenizerInputWrite::finish(
  self : LunaTokenizerInputWrite,
) -> LunaTokenizerWork raise TokenizerError {
  let worker = self.require_worker()
  if worker.written_input_bytes != worker.input_length {
    raise LunaInvalidInputWrite
  }
  worker.phase = if worker.spec.profile == TOKENIZER_PROFILE_QWEN3_BPE {
    LUNA_WORKER_QWEN3_NORMALIZE
  } else {
    LUNA_WORKER_INITIALIZE
  }
  { worker, epoch: self.epoch }
}

///|
/// Invalidates this writer without clearing the fixed byte backing.
pub fn LunaTokenizerInputWrite::abort(
  self : LunaTokenizerInputWrite,
) -> Unit raise TokenizerError {
  self.require_worker().clear_request()
}
