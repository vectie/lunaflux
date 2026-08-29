///|
pub fn LunaTokenizerWorker::new(
  spec : TokenizerSpec,
  input_capacity~ : Int,
  output_capacity~ : Int,
  budget : LunaTokenizerStepBudget,
) -> LunaTokenizerWorker raise TokenizerError {
  if input_capacity <= 0 ||
    input_capacity > spec.limits.max_input_bytes ||
    output_capacity <= 0 ||
    output_capacity > spec.limits.max_output_tokens {
    raise LunaInvalidWorkerCapacity
  }
  let node_capacity = spec.worker_symbol_capacity(input_capacity)
  {
    spec,
    input_capacity,
    node_capacity,
    output_capacity,
    budget,
    tokens: FixedArray::make(node_capacity, 0),
    previous: FixedArray::make(node_capacity, LUNA_REMOVED_NODE),
    next: FixedArray::make(node_capacity, LUNA_REMOVED_NODE),
    input: FixedArray::make(
      LunaTokenizerWorker::required_byte_cells(input_capacity).to_int(),
      b'\x00',
    ),
    input_length: 0,
    written_input_bytes: 0,
    ordinary_ready: false,
    segment_start: true,
    post_processor_pending: false,
    scalar_is_meta: false,
    scalar_direct_token_id: -1,
    scalar_byte_offset: 0,
    scalar_byte_length: 0,
    scalar_source_advance: 0,
    normalized_length: 0,
    qwen_segment_end: -1,
    special_policy: AsOrdinary,
    truncation: RejectOverflow,
    output_limit: 0,
    phase: LUNA_WORKER_IDLE,
    epoch: 0UL,
    input_offset: 0,
    node_count: 0,
    head: -1,
    tail: -1,
    special_index: 0,
    special_piece_offset: 0,
    special_matches: true,
    best_special_id: -1,
    best_special_bytes: 0,
    scan_left: -1,
    lookup_low: 0,
    lookup_high: 0,
    selected_rank: -1,
    compact_node: -1,
    compact_actual: 0,
    output_count: 0,
    truncated: false,
    last_work_units: 0,
  }
}

///|
fn LunaTokenizerWorker::clear_request(self : LunaTokenizerWorker) -> Unit {
  self.input_length = 0
  self.written_input_bytes = 0
  self.ordinary_ready = false
  self.segment_start = true
  self.post_processor_pending = false
  self.scalar_is_meta = false
  self.scalar_direct_token_id = -1
  self.scalar_byte_offset = 0
  self.scalar_byte_length = 0
  self.scalar_source_advance = 0
  self.normalized_length = 0
  self.qwen_segment_end = -1
  self.output_limit = 0
  self.phase = LUNA_WORKER_IDLE
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
}

///|
pub fn LunaTokenizerWorker::begin_bytes(
  self : LunaTokenizerWorker,
  input : Bytes,
  special_policy~ : SpecialTokenPolicy,
  output_limit~ : Int,
  truncation~ : TruncationPolicy,
) -> LunaTokenizerWork raise TokenizerError {
  let write = self.begin_luna_input(
    byte_count=input.length(),
    special_policy~,
    output_limit~,
    truncation~,
  )
  for byte in input {
    write.push_byte(byte)
  }
  write.finish()
}

///|
fn LunaTokenizerWork::require_worker(
  self : LunaTokenizerWork,
) -> LunaTokenizerWorker raise TokenizerError {
  if self.epoch == 0UL ||
    self.worker.epoch != self.epoch ||
    self.worker.phase == LUNA_WORKER_IDLE ||
    self.worker.phase == LUNA_WORKER_INPUT_WRITE {
    raise LunaTokenizerWorkerStale
  }
  self.worker
}

///|
pub fn LunaTokenizerWork::last_work_units(
  self : LunaTokenizerWork,
) -> Int raise TokenizerError {
  self.require_worker().last_work_units
}

///|
pub fn LunaTokenizerWork::token_count(
  self : LunaTokenizerWork,
) -> Int raise TokenizerError {
  let worker = self.require_worker()
  if worker.phase != LUNA_WORKER_READY {
    raise LunaTokenizerWorkerNotReady
  }
  worker.output_count
}

///|
/// Reads one result token without exposing the worker's mutable fixed storage.
/// The status is non-ready for an invalid index or before result readiness;
/// stale work credits retain the existing typed error.
pub fn LunaTokenizerWork::token_status(
  self : LunaTokenizerWork,
  index : Int,
) -> LunaTokenizerTokenStatus raise TokenizerError {
  let worker = self.require_worker()
  if worker.phase != LUNA_WORKER_READY ||
    index < 0 ||
    index >= worker.output_count {
    return LunaTokenizerTokenStatus(1, 0)
  }
  LunaTokenizerTokenStatus(0, worker.tokens[index])
}

///|
pub fn LunaTokenizerWork::copy_tokens_to(
  self : LunaTokenizerWork,
  destination : FixedArray[Int],
  source_offset~ : Int,
  destination_offset~ : Int,
  length~ : Int,
) -> Unit raise TokenizerError {
  let worker = self.require_worker()
  if worker.phase != LUNA_WORKER_READY {
    raise LunaTokenizerWorkerNotReady
  }
  if source_offset < 0 ||
    destination_offset < 0 ||
    length < 0 ||
    source_offset > worker.output_count - length ||
    destination_offset > destination.length() - length ||
    length > worker.budget.work_units() {
    raise LunaInvalidTokenCopy
  }
  for offset in 0..<length {
    destination[destination_offset + offset] = worker.tokens[source_offset +
      offset]
  }
  worker.last_work_units = length
}

///|
pub fn LunaTokenizerWork::abort(
  self : LunaTokenizerWork,
) -> Unit raise TokenizerError {
  self.require_worker().clear_request()
}

///|
pub fn LunaTokenizerWork::was_truncated(
  self : LunaTokenizerWork,
) -> Bool raise TokenizerError {
  let worker = self.require_worker()
  if worker.phase != LUNA_WORKER_READY {
    raise LunaTokenizerWorkerNotReady
  }
  worker.truncated
}
