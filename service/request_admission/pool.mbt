///|
fn new_pool_with_clock(
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  clock_read : () -> UInt64 raise RequestAdmissionError,
  lane_count : Int,
  step_budget : LunaRequestPreparationStepBudget,
  work_limit : LunaRequestPreparationWorkLimit,
  storage_budget : LunaRequestPreparationStorageBudget,
) -> LunaRequestPreparationPool raise RequestAdmissionError {
  if framed_limits.inference_limits() != inference {
    raise Invalid(RequestContract)
  }
  if tokenizer.digest() != expected_tokenizer {
    raise Invalid(TokenizerIdentity)
  }
  let (input_capacity, output_capacity) = validate_pool_storage(
    tokenizer, inference, framed_limits, lane_count, storage_budget,
  )
  let lanes = FixedArray::makei(lane_count, _ => {
    new_preparation_lane(
      tokenizer, inference, framed_limits, input_capacity, output_capacity,
    )
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
  framed_limits~ : @framed_wire.FramedWireLimits,
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
    framed_limits,
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
  lane.hard_deadline = None
  lane.effective = None
  lane.tokenizer_input_write = None
  lane.tokenizer_work = None
  lane.token_write = None
  lane.token_lease = None
  lane.semantic_write.clear()
  lane.semantic_work.clear()
  lane.semantic_lease.clear()
  lane.framed_work.clear()
  lane.framed_view.clear()
  lane.typed_handoff.clear()
  lane.output_work = None
  lane.output_lease = None
  lane.tokens = None
  lane.claim = None
  lane.failure = None
  lane.copy_index = 0
  lane.semantic_item_index = 0
  lane.semantic_unit_offset = 0
  lane.semantic_byte_length = 0
  lane.semantic_codepoint = 0
  lane.semantic_codepoint_width = 0
  lane.semantic_codepoint_byte_index = 0
  lane.framed_direct = false
  lane.typed_direct = false
  lane.typed_sampling = @inference.LunaValidatedSampling::greedy()
  lane.framed_receipt_complete = false
  lane.framed_model_index = 0
  lane.framed_input_kind = 0
  lane.framed_input_count = 0
  lane.framed_stop_token_count = 0
  lane.framed_stop_string_count = 0
  lane.framed_cache_scope_length = 0
  lane.framed_request_id = 0UL
  lane.framed_sampling_mode = 0
  lane.framed_temperature = 1.0
  lane.framed_has_top_k = false
  lane.framed_top_k = 0
  lane.framed_has_top_p = false
  lane.framed_top_p = 0.0
  lane.framed_sampling_seed = 0UL
  lane.framed_streaming = false
  lane.framed_token_ids_only = false
  lane.trusted_receipt_bound = false
  lane.trusted_deadline_millis = 0UL
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
  lane.output_work = None
  match received.request.input() {
    Text(text) => {
      lane.tokenizer_input_write = Some(
        lane.tokenizer.begin_luna_input(
          byte_count=text.utf8_bytes(),
          special_policy=Reject,
          output_limit=input_capacity,
          truncation=RejectOverflow,
        ) catch {
          _ => raise Invalid(Tokenization)
        },
      )
      lane.tokenizer_work = None
      lane.tokens = None
      lane.phase = LUNA_PREPARATION_TEXT_INPUT_COPY
    }
    TokenIds(tokens) => {
      if !tokens.is_live() || tokens.length() > input_capacity {
        raise Invalid(Input)
      }
      lane.tokenizer_input_write = None
      lane.tokenizer_work = None
      lane.tokens = Some(tokens)
      lane.phase = LUNA_PREPARATION_SEMANTIC_CACHE_MEASURE
    }
  }
}

///|
fn preparation_admission(
  pool : LunaRequestPreparationPool,
  kind : LunaRequestPreparationAdmissionKind,
  lane : Int,
  generation : UInt64,
  consumed : Int,
) -> LunaRequestPreparationAdmission {
  {
    pool,
    lane,
    generation,
    consumed,
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
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  if self.free_count == 0 {
    return preparation_admission(
      self,
      LunaRequestPreparationSaturated,
      -1,
      0UL,
      0,
    )
  }
  if received.consumed {
    raise Invalid(FrameContract)
  }
  let next_lane = self.free_ring[self.free_head]
  if self.lanes[next_lane].generation == 0xffffffffffffffffUL {
    self.draining = true
    return preparation_admission(
      self,
      LunaRequestPreparationDraining,
      -1,
      0UL,
      0,
    )
  }
  let lane_index = self.pop_free_lane()
  received.consumed = true
  let start_result : Result[Unit, RequestAdmissionError] = Ok(
    self.start_lane(lane_index, received),
  ) catch {
    error => Err(error)
  }
  match start_result {
    Err(error) => {
      let lane = self.lanes[lane_index]
      lane.queued = false
      self.abort_lane_resources(lane)
      self.reset_lane_to_idle(lane_index)
      raise error
    }
    Ok(_) => ()
  }
  self.enqueue_active(lane_index)
  preparation_admission(
    self,
    LunaRequestPreparationAdmitted,
    lane_index,
    self.lanes[lane_index].generation,
    0,
  )
}
