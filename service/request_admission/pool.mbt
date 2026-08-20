///|
fn checked_add_cells(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise RequestAdmissionError {
  if left > 0xffffffffffffffffUL - right {
    raise Invalid(LunaPreparationStorage)
  }
  left + right
}

///|
fn checked_multiply_cells(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise RequestAdmissionError {
  if right != 0UL && left > 0xffffffffffffffffUL / right {
    raise Invalid(LunaPreparationStorage)
  }
  left * right
}

///|
fn validate_pool_storage(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  lane_count : Int,
  budget : LunaRequestPreparationStorageBudget,
) -> (Int, Int) raise RequestAdmissionError {
  if lane_count <= 0 || lane_count > LUNA_PREPARATION_MAX_LANES {
    raise Invalid(LunaPreparationPool)
  }
  let input_capacity = Int::min(
    tokenizer.limits().max_input_bytes,
    inference.max_text_bytes(),
  )
  let output_capacity = Int::min(
    tokenizer.limits().max_output_tokens,
    inference.max_input_tokens(),
  )
  let tokenizer_cells = @tokenizer.LunaTokenizerWorker::required_int_cells(
    input_capacity,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let token_cells = @inference.LunaTokenBufferStorage::required_int_cells(
    output_capacity,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }
  let output_int_cells = (@incremental_output.LunaIncrementalOutputWorkspace::required_int_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }).to_uint64()
  let per_lane_int = checked_add_cells(
    checked_add_cells(tokenizer_cells, token_cells),
    output_int_cells,
  )
  let per_lane_byte = (@incremental_output.LunaIncrementalOutputWorkspace::required_byte_cells(
    inference,
  ) catch {
    _ => raise Invalid(LunaPreparationStorage)
  }).to_uint64()
  let lanes = lane_count.to_uint64()
  let total_int = checked_add_cells(
    checked_multiply_cells(per_lane_int, lanes),
    checked_multiply_cells(lanes, 2UL),
  )
  let total_byte = checked_multiply_cells(per_lane_byte, lanes)
  let total_references = lanes
  if total_int > budget.int_cells ||
    total_byte > budget.byte_cells ||
    total_references > budget.reference_cells {
    raise Invalid(LunaPreparationStorage)
  }
  (input_capacity, output_capacity)
}

///|
fn new_preparation_lane(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  input_capacity : Int,
  output_capacity : Int,
) -> LunaRequestPreparationLane raise RequestAdmissionError {
  let tokenizer_budget = @tokenizer.LunaTokenizerStepBudget::new(1) catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  let output_budget = @incremental_output.LunaIncrementalOutputStepBudget::new(
    1,
  ) catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  {
    tokenizer: @tokenizer.LunaTokenizerWorker::new(
      tokenizer,
      input_capacity~,
      output_capacity~,
      tokenizer_budget,
    ) catch {
      _ => raise Invalid(Tokenization)
    },
    token_storage: @inference.LunaTokenBufferStorage::new(
      capacity=output_capacity,
      limits=inference,
    ) catch {
      _ => raise Invalid(Input)
    },
    output_workspace: @incremental_output.LunaIncrementalOutputWorkspace::new(
      tokenizer, inference, output_budget,
    ) catch {
      _ => raise Invalid(OutputContract)
    },
    generation: 0UL,
    phase: LUNA_PREPARATION_IDLE,
    queued: false,
    receipt_at_millis: 0UL,
    request: None,
    deadline: None,
    effective: None,
    input_capacity: 0,
    tokenizer_work: None,
    token_write: None,
    token_lease: None,
    output_work: None,
    output_lease: None,
    tokens: None,
    copy_index: 0,
    claim: None,
    failure: None,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn new_pool_with_clock(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  clock_read : () -> UInt64 raise RequestAdmissionError,
  lane_count : Int,
  step_budget : LunaRequestPreparationStepBudget,
  work_limit : LunaRequestPreparationWorkLimit,
  storage_budget : LunaRequestPreparationStorageBudget,
) -> LunaRequestPreparationPool raise RequestAdmissionError {
  if tokenizer.digest() != expected_tokenizer {
    raise Invalid(TokenizerIdentity)
  }
  let (input_capacity, output_capacity) = validate_pool_storage(
    tokenizer, inference, lane_count, storage_budget,
  )
  let lanes = FixedArray::makei(lane_count, _ => {
    new_preparation_lane(tokenizer, inference, input_capacity, output_capacity)
  })
  {
    tokenizer,
    expected_tokenizer,
    expected_model,
    inference,
    clock_read,
    step_budget,
    work_limit,
    lanes,
    free_ring: FixedArray::makei(lane_count, index => index),
    active_ring: FixedArray::make(lane_count, -1),
    free_head: 0,
    free_count: lane_count,
    active_head: 0,
    active_count: 0,
    draining: false,
  }
}

///|
pub fn LunaRequestPreparationPool::new(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer~ : @tokenizer.TokenizerDigest,
  expected_model~ : @spec.ModelIdentity,
  inference~ : @inference.InferenceLimits,
  clock~ : @monotonic_clock.MonotonicClock,
  lane_count~ : Int,
  step_budget~ : LunaRequestPreparationStepBudget,
  work_limit~ : LunaRequestPreparationWorkLimit,
  storage_budget~ : LunaRequestPreparationStorageBudget,
) -> LunaRequestPreparationPool raise RequestAdmissionError {
  new_pool_with_clock(
    tokenizer,
    expected_tokenizer,
    expected_model,
    inference,
    () => clock.now_millis() catch { _ => raise ClockUnavailable },
    lane_count,
    step_budget,
    work_limit,
    storage_budget,
  )
}

///|
pub fn LunaRequestPreparationPool::begin_drain(
  self : LunaRequestPreparationPool,
) -> Unit {
  self.draining = true
}

///|
pub fn LunaRequestPreparationPool::is_drained(
  self : LunaRequestPreparationPool,
) -> Bool {
  self.draining && self.free_count == self.lanes.length()
}

///|
fn LunaRequestPreparationPool::pop_free_lane(
  self : LunaRequestPreparationPool,
) -> Int {
  if self.free_count <= 0 {
    abort("Luna preparation free-ring invariant")
  }
  let lane = self.free_ring[self.free_head]
  self.free_head = (self.free_head + 1) % self.lanes.length()
  self.free_count -= 1
  lane
}

///|
fn LunaRequestPreparationPool::push_free_lane(
  self : LunaRequestPreparationPool,
  lane : Int,
) -> Unit {
  if self.free_count >= self.lanes.length() {
    abort("Luna preparation free-ring overflow invariant")
  }
  let tail = (self.free_head + self.free_count) % self.lanes.length()
  self.free_ring[tail] = lane
  self.free_count += 1
}

///|
fn LunaRequestPreparationPool::enqueue_active(
  self : LunaRequestPreparationPool,
  lane_index : Int,
) -> Unit {
  if self.active_count >= self.active_ring.length() {
    abort("Luna preparation active-ring overflow invariant")
  }
  let tail = (self.active_head + self.active_count) % self.active_ring.length()
  self.active_ring[tail] = lane_index
  self.active_count += 1
  self.lanes[lane_index].queued = true
}

///|
fn LunaRequestPreparationPool::dequeue_active(
  self : LunaRequestPreparationPool,
) -> Int {
  if self.active_count <= 0 {
    abort("Luna preparation active-ring invariant")
  }
  let lane = self.active_ring[self.active_head]
  self.active_ring[self.active_head] = -1
  self.active_head = (self.active_head + 1) % self.active_ring.length()
  self.active_count -= 1
  self.lanes[lane].queued = false
  lane
}

///|
fn validate_pool_time(
  receipt_at_millis : UInt64,
  deadline : @inference.AdmissionDeadline,
  observed_at_millis : UInt64,
) -> Unit raise RequestAdmissionError {
  if observed_at_millis < receipt_at_millis {
    raise Invalid(ReceiptTime)
  }
  if deadline.is_expired(observed_at_millis) {
    raise Invalid(Deadline)
  }
}

///|
fn LunaRequestPreparationPool::reset_lane_to_idle(
  self : LunaRequestPreparationPool,
  lane_index : Int,
) -> Unit {
  let lane = self.lanes[lane_index]
  lane.phase = LUNA_PREPARATION_IDLE
  lane.request = None
  lane.deadline = None
  lane.effective = None
  lane.tokenizer_work = None
  lane.token_write = None
  lane.token_lease = None
  lane.output_work = None
  lane.output_lease = None
  lane.tokens = None
  lane.claim = None
  lane.failure = None
  lane.copy_index = 0
  lane.last_work_units = 0
  lane.total_work_units = 0UL
  self.push_free_lane(lane_index)
}

///|
fn LunaRequestPreparationPool::release_lane(
  self : LunaRequestPreparationPool,
  lane_index : Int,
  generation : UInt64,
) -> Unit raise RequestAdmissionError {
  if lane_index < 0 || lane_index >= self.lanes.length() {
    raise Invalid(LunaPreparedLifecycle)
  }
  let lane = self.lanes[lane_index]
  if generation == 0UL || lane.generation != generation || lane.queued {
    raise Invalid(LunaPreparedLifecycle)
  }
  self.reset_lane_to_idle(lane_index)
}

///|
fn LunaRequestPreparationPool::start_lane(
  self : LunaRequestPreparationPool,
  lane_index : Int,
  received : ReceivedRequest,
) -> Unit raise RequestAdmissionError {
  let lane = self.lanes[lane_index]
  if received.inference != self.inference {
    raise Invalid(RequestContract)
  }
  if received.request.model_identity() != self.expected_model {
    raise Invalid(ModelIdentity)
  }
  if self.tokenizer.digest() != self.expected_tokenizer {
    raise Invalid(TokenizerIdentity)
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=received.request.max_new_tokens(),
    context_ceiling=received.request.context_ceiling(),
    self.inference,
  ) catch {
    _ => raise Invalid(Input)
  }
  let input_capacity = Int::min(
    effective.context_ceiling() - effective.max_new_tokens(),
    Int::min(
      self.inference.max_input_tokens(),
      self.tokenizer.limits().max_output_tokens,
    ),
  )
  if input_capacity <= 0 {
    raise Invalid(Input)
  }
  validate_pool_time(
    received.receipt.at_millis,
    received.deadline,
    (self.clock_read)(),
  )
  if lane.generation == 0xffffffffffffffffUL {
    self.draining = true
    raise Invalid(LunaPreparationPool)
  }
  lane.generation += 1UL
  lane.receipt_at_millis = received.receipt.at_millis
  lane.request = Some(received.request)
  lane.deadline = Some(received.deadline)
  lane.effective = Some(effective)
  lane.input_capacity = input_capacity
  lane.copy_index = 0
  lane.failure = None
  lane.last_work_units = 0
  lane.total_work_units = 0UL
  lane.claim = None
  lane.output_work = Some(
    lane.output_workspace.begin(received.request.stops(), special_policy=Skip) catch {
      _ => raise Invalid(OutputContract)
    },
  )
  match received.request.input() {
    Text(text) => {
      lane.tokenizer_work = Some(
        lane.tokenizer.begin_bytes(
          text.utf8(),
          special_policy=Reject,
          output_limit=input_capacity,
          truncation=RejectOverflow,
        ) catch {
          _ => raise Invalid(Tokenization)
        },
      )
      lane.tokens = None
      lane.phase = LUNA_PREPARATION_TEXT_TOKENIZE
    }
    TokenIds(tokens) => {
      if !tokens.is_live() || tokens.length() > input_capacity {
        raise Invalid(Input)
      }
      lane.tokenizer_work = None
      lane.tokens = Some(tokens)
      lane.phase = LUNA_PREPARATION_OUTPUT_SETUP
    }
  }
}

///|
fn preparation_admission(
  pool : LunaRequestPreparationPool,
  kind : LunaRequestPreparationAdmissionKind,
  lane : Int,
  generation : UInt64,
) -> LunaRequestPreparationAdmission {
  {
    pool,
    lane,
    generation,
    disposition: match kind {
      LunaRequestPreparationAdmitted => 0
      LunaRequestPreparationSaturated => 1
      LunaRequestPreparationDraining => 2
    },
  }
}

///|
pub fn LunaRequestPreparationPool::try_submit(
  self : LunaRequestPreparationPool,
  received : ReceivedRequest,
) -> LunaRequestPreparationAdmission raise RequestAdmissionError {
  if self.draining {
    return preparation_admission(self, LunaRequestPreparationDraining, -1, 0UL)
  }
  if self.free_count == 0 {
    return preparation_admission(self, LunaRequestPreparationSaturated, -1, 0UL)
  }
  if received.consumed {
    raise Invalid(FrameContract)
  }
  let next_lane = self.free_ring[self.free_head]
  if self.lanes[next_lane].generation == 0xffffffffffffffffUL {
    self.draining = true
    return preparation_admission(self, LunaRequestPreparationDraining, -1, 0UL)
  }
  let lane_index = self.pop_free_lane()
  received.consumed = true
  self.start_lane(lane_index, received) catch {
    error => {
      let lane = self.lanes[lane_index]
      lane.queued = false
      self.abort_lane_resources(lane)
      self.reset_lane_to_idle(lane_index)
      raise error
    }
  }
  self.enqueue_active(lane_index)
  preparation_admission(
    self,
    LunaRequestPreparationAdmitted,
    lane_index,
    self.lanes[lane_index].generation,
  )
}

///|
pub fn LunaRequestPreparationAdmission::kind(
  self : LunaRequestPreparationAdmission,
) -> LunaRequestPreparationAdmissionKind {
  match self.disposition {
    0 => LunaRequestPreparationAdmitted
    1 => LunaRequestPreparationSaturated
    2 => LunaRequestPreparationDraining
    _ => abort("Luna preparation admission invariant")
  }
}

///|
pub fn LunaRequestPreparationAdmission::work(
  self : LunaRequestPreparationAdmission,
) -> LunaRequestPreparationWork raise RequestAdmissionError {
  if self.disposition != 0 || self.lane < 0 || self.generation == 0UL {
    raise Invalid(LunaPreparedLifecycle)
  }
  { pool: self.pool, lane: self.lane, generation: self.generation }
}
