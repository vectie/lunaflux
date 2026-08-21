///|
fn LunaRequestPreparationLane::require_request(
  self : LunaRequestPreparationLane,
) -> @inference.GenerateRequest {
  match self.request {
    Some(value) => value
    None => abort("Luna preparation request invariant")
  }
}

///|
fn LunaRequestPreparationLane::require_deadline(
  self : LunaRequestPreparationLane,
) -> @inference.AdmissionDeadline {
  match self.deadline {
    Some(value) => value
    None => abort("Luna preparation deadline invariant")
  }
}

///|
fn LunaRequestPreparationLane::require_effective(
  self : LunaRequestPreparationLane,
) -> @inference.EffectiveLimits {
  match self.effective {
    Some(value) => value
    None => abort("Luna preparation effective-limit invariant")
  }
}

///|
fn LunaRequestPreparationPool::abort_lane_resources(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit {
  ignore(self)
  match lane.tokenizer_input_write {
    Some(write) => {
      try! write.abort()
      lane.tokenizer_input_write = None
    }
    None => ()
  }
  match lane.tokenizer_work {
    Some(work) => {
      try! work.abort()
      lane.tokenizer_work = None
    }
    None => ()
  }
  match lane.token_write {
    Some(write) => {
      try! write.abort()
      lane.token_write = None
    }
    None => ()
  }
  match lane.token_lease {
    Some(lease) => {
      try! lease.release()
      lane.token_lease = None
    }
    None => ()
  }
  match lane.output_work {
    Some(work) => {
      try! work.abort()
      lane.output_work = None
    }
    None => ()
  }
  match lane.output_lease {
    Some(lease) => {
      try! lease.release()
      lane.output_lease = None
    }
    None => ()
  }
  match lane.claim {
    Some(claim) => {
      try! claim.release_resources_only()
      return
    }
    None => ()
  }
  if !lane.semantic_write.is_empty() {
    try! lane.semantic_write[0].abort()
    lane.semantic_write.clear()
  }
  if !lane.semantic_work.is_empty() {
    try! lane.semantic_work[0].abort()
    lane.semantic_work.clear()
  }
  if !lane.semantic_lease.is_empty() {
    try! lane.semantic_lease[0].release()
    lane.semantic_lease.clear()
  }
}

///|
fn LunaRequestPreparationPool::fail_lane(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
  failure : RequestAdmissionError,
) -> Unit {
  self.abort_lane_resources(lane)
  lane.claim = None
  lane.failure = Some(failure)
  lane.phase = LUNA_PREPARATION_FAILED
}

///|
fn LunaRequestPreparationPool::progress_text_input_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let write = match lane.tokenizer_input_write {
    Some(value) => value
    None => abort("Luna tokenizer-input invariant")
  }
  let text = match lane.require_request().input() {
    Text(value) => value
    TokenIds(_) => abort("Luna tokenizer-input source invariant")
  }
  if lane.copy_index < text.utf8_bytes() {
    write.push_byte(text.utf8()[lane.copy_index]) catch {
      _ => raise Invalid(Tokenization)
    }
    lane.copy_index += 1
    return
  }
  lane.tokenizer_work = Some(
    write.finish() catch {
      _ => raise Invalid(Tokenization)
    },
  )
  lane.tokenizer_input_write = None
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TEXT_TOKENIZE
}

///|
fn LunaRequestPreparationPool::progress_text_tokenize(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-work invariant")
  }
  let progress = work.progress() catch {
    SpecialTokenRejected(token_id=_) | OutputTooLong(actual=_, maximum=_) => {
      // These terminal tokenizer failures clear the worker before raising, so
      // the retained Work alias is already stale and must not be aborted.
      lane.tokenizer_work = None
      raise Invalid(Tokenization)
    }
    _ => raise Invalid(Tokenization)
  }
  match progress {
    LunaTokenizerPending => ()
    LunaTokenizerReady => lane.phase = LUNA_PREPARATION_TEXT_BEGIN_COPY
  }
}

///|
fn LunaRequestPreparationPool::progress_text_begin_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-copy invariant")
  }
  let count = work.token_count() catch { _ => raise Invalid(Tokenization) }
  if count <= 0 || count > lane.input_capacity {
    raise Invalid(Input)
  }
  lane.token_write = Some(
    lane.token_storage.begin(token_count=count) catch {
      _ => raise Invalid(Input)
    },
  )
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TEXT_COPY
}

///|
fn LunaRequestPreparationPool::progress_text_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let tokenizer_work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-copy work invariant")
  }
  let write = match lane.token_write {
    Some(value) => value
    None => abort("Luna token-write invariant")
  }
  let count = tokenizer_work.token_count() catch {
    _ => raise Invalid(Tokenization)
  }
  if lane.copy_index < count {
    let status = tokenizer_work.token_status(lane.copy_index) catch {
      _ => raise Invalid(Tokenization)
    }
    if !status.is_ready() {
      raise Invalid(Tokenization)
    }
    write.push(status.token_id()) catch {
      _ => raise Invalid(Input)
    }
    lane.copy_index += 1
    return
  }
  let lease = write.seal() catch { _ => raise Invalid(Input) }
  lane.token_write = None
  lane.token_lease = Some(lease)
  lane.tokens = Some(try! lease.token_buffer())
  tokenizer_work.abort() catch {
    _ => raise Invalid(Tokenization)
  }
  lane.tokenizer_work = None
  lane.begin_semantic_import()
}

///|
fn LunaRequestPreparationPool::progress_output_setup(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.output_work {
    Some(value) => value
    None => abort("Luna output-work invariant")
  }
  let progress = work.progress() catch { _ => raise Invalid(OutputContract) }
  match progress {
    LunaIncrementalOutputPending => ()
    LunaIncrementalOutputReady => lane.phase = LUNA_PREPARATION_ASSEMBLE
  }
}

///|
fn LunaRequestPreparationPool::progress_assemble(
  self : LunaRequestPreparationPool,
  lane_index : Int,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  let request = lane.require_request()
  let tokens = match lane.tokens {
    Some(value) => value
    None => abort("Luna prepared-token invariant")
  }
  if !tokens.is_live() {
    raise Invalid(Input)
  }
  let output_work = match lane.output_work {
    Some(value) => value
    None => abort("Luna output-assembly invariant")
  }
  let output_lease = output_work.take_lease() catch {
    _ => raise Invalid(OutputContract)
  }
  lane.output_work = None
  lane.output_lease = Some(output_lease)
  validate_pool_time(
    lane.receipt_at_millis,
    lane.require_deadline(),
    (self.clock_read)(),
  )
  let scheduler_request = @core.TokenizedRequest::new(
    request_id=request.request_id(),
    model_identity=request.model_identity(),
    input_tokens=tokens,
    sampling=request.sampling(),
    sampling_seed=request.sampling_seed(),
    stop_tokens=lane.require_semantic_lease().stop_token_view() catch {
        _ => raise Invalid(SchedulerContract)
      },
    stream_preference=request.stream_preference(),
    effective_limits=lane.require_effective(),
    admission_deadline=lane.require_deadline(),
  ) catch {
    _ => raise Invalid(SchedulerContract)
  }
  let claim : LunaPreparedRequestClaim = {
    scheduler_request: Some(scheduler_request),
    pooled_output: Some(output_lease),
    token_lease: lane.token_lease,
    semantic_lease: lane.semantic_lease,
    stop_tokens: lane.require_semantic_lease().stop_token_view() catch {
      _ => raise Invalid(SchedulerContract)
    },
    pool: Some(self),
    lane: lane_index,
    generation: lane.generation,
    tokenizer_digest: self.tokenizer.digest(),
    inference_limits: self.inference,
    input_tokens: tokens.length(),
  }
  lane.token_lease = None
  lane.output_lease = None
  lane.claim = Some(claim)
  lane.phase = LUNA_PREPARATION_READY
}

///|
fn LunaRequestPreparationPool::progress_lane_one(
  self : LunaRequestPreparationPool,
  lane_index : Int,
) -> Unit raise RequestAdmissionError {
  let lane = self.lanes[lane_index]
  match lane.phase {
    LUNA_PREPARATION_TEXT_INPUT_COPY => self.progress_text_input_copy(lane)
    LUNA_PREPARATION_TEXT_TOKENIZE => self.progress_text_tokenize(lane)
    LUNA_PREPARATION_TEXT_BEGIN_COPY => self.progress_text_begin_copy(lane)
    LUNA_PREPARATION_TEXT_COPY => self.progress_text_copy(lane)
    LUNA_PREPARATION_SEMANTIC_CACHE_MEASURE =>
      self.progress_semantic_cache_measure(lane)
    LUNA_PREPARATION_SEMANTIC_BEGIN => self.progress_semantic_begin(lane)
    LUNA_PREPARATION_SEMANTIC_TOKEN_COPY =>
      self.progress_semantic_token_copy(lane)
    LUNA_PREPARATION_SEMANTIC_STRING_MEASURE =>
      self.progress_semantic_string_measure(lane)
    LUNA_PREPARATION_SEMANTIC_STRING_HEADER =>
      self.progress_semantic_string_header(lane)
    LUNA_PREPARATION_SEMANTIC_STRING_EMIT =>
      self.progress_semantic_string_emit(lane)
    LUNA_PREPARATION_SEMANTIC_CACHE_EMIT =>
      self.progress_semantic_cache_emit(lane)
    LUNA_PREPARATION_SEMANTIC_FINISH => self.progress_semantic_finish(lane)
    LUNA_PREPARATION_SEMANTIC_VALIDATE => self.progress_semantic_validate(lane)
    LUNA_PREPARATION_SEMANTIC_TAKE => self.progress_semantic_take(lane)
    LUNA_PREPARATION_OUTPUT_SETUP => self.progress_output_setup(lane)
    LUNA_PREPARATION_ASSEMBLE => self.progress_assemble(lane_index, lane)
    _ => abort("Luna preparation progress phase invariant")
  }
}

///|
pub fn LunaRequestPreparationPool::progress(
  self : LunaRequestPreparationPool,
) -> LunaRequestPreparationPoolProgress {
  if self.active_count == 0 {
    return LunaRequestPreparationPoolIdle
  }
  let lane_index = self.dequeue_active()
  let lane = self.lanes[lane_index]
  lane.last_work_units = 0
  if lane.phase == LUNA_PREPARATION_CANCEL_REQUESTED {
    self.abort_lane_resources(lane)
    try! self.release_lane(lane_index, lane.generation)
    return LunaRequestPreparationPoolAdvanced
  }
  try
    validate_pool_time(
      lane.receipt_at_millis,
      lane.require_deadline(),
      (self.clock_read)(),
    )
  catch {
    error => self.fail_lane(lane, error)
  } noraise {
    _ => {
      let maximum = self.step_budget.work_units()
      let mut used = 0
      while used < maximum &&
            lane.phase != LUNA_PREPARATION_READY &&
            lane.phase != LUNA_PREPARATION_FAILED {
        if lane.total_work_units >= self.work_limit.work_units() {
          self.fail_lane(lane, Invalid(LunaPreparationWork))
          break
        }
        lane.total_work_units += 1UL
        used += 1
        self.progress_lane_one(lane_index) catch {
          error => self.fail_lane(lane, error)
        }
      }
      lane.last_work_units = used
    }
  }
  if lane.phase != LUNA_PREPARATION_READY &&
    lane.phase != LUNA_PREPARATION_FAILED &&
    lane.phase != LUNA_PREPARATION_CANCEL_REQUESTED {
    self.enqueue_active(lane_index)
  }
  LunaRequestPreparationPoolAdvanced
}
