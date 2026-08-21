///|
struct FoundationSession {
  instance : @online_session.LunaOnlineInstance
  ticket : @online_session.LunaOnlineRequestTicket
  adapter : @framed_wire.LunaFramedEventAdapter
  mut credit : @online_session.LunaOnlineEventCredit?
}

///|
fn prepare_foundation_instance(
  executable : Bytes,
  expected_tokenizer : @tokenizer.TokenizerDigest,
) -> @online_session.LunaOnlineInstance raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_instance(
    expected_tokenizer,
    identity(),
    inference_limits(),
    scheduler_blueprint(),
    service_binding(),
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), LunaOnlineInstanceReady)
  prepared.take_ready()
}

///|
fn prepare_default_foundation_instance(
  executable : Bytes,
) -> @online_session.LunaOnlineInstance raise {
  prepare_foundation_instance(executable, session_tokenizer().digest())
}

///|
fn lower_reject_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=1,
    max_plan_tokens=1,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=1,
    max_sequence_tokens=4,
    max_token_id=1,
  )
}

///|
fn lower_reject_blueprint() -> @scheduler.SchedulerBlueprint raise {
  let scheduling = @scheduler_config.new(
    step_token_budget=1,
    max_active_requests=2,
    max_waiting_requests=4,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=1,
  )
  let cache = @cache.new(
    tokens_per_page=2,
    total_page_count=8,
    block_table_pages_per_request=2,
    prefix_enabled=false,
    max_prefix_entries=0,
    max_prefix_nodes=0,
    max_prefix_tokens_per_entry=0,
    max_prefix_pages=0,
    max_active_references_per_page=4,
    max_cached_references_per_page=4,
    layout_version=1,
  )
  let shapes = @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=4,
    vocabulary_size=2,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let limits = lower_reject_worker_limits()
  let capacity = @runtime_resolved.resolve(scheduling, cache, shapes, limits)
  let generation = @worker_protocol.ModelPlanGeneration::first()
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity=identity(),
    model_plan_generation=generation,
    [@plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  @scheduler.SchedulerBlueprint::new(
    capacity,
    identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
fn lower_reject_startup() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source().digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=lower_reject_worker_limits(),
    inference_limits=inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn prepare_lower_reject_foundation_instance(
  executable : Bytes,
  expected_tokenizer : @tokenizer.TokenizerDigest,
) -> @online_session.LunaOnlineInstance raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_instance(
    expected_tokenizer,
    identity(),
    inference_limits(),
    lower_reject_blueprint(),
    service_binding(),
    executable,
    lower_reject_startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), LunaOnlineInstanceReady)
  prepared.take_ready()
}

///|
fn prepare_foundation_request(
  tokenizer : @tokenizer.TokenizerSpec,
  received : @request_admission.ReceivedRequest,
) -> @request_admission.LunaPreparedRequest raise {
  @request_admission.prepare_luna_request(
    received,
    tokenizer,
    tokenizer.digest(),
    identity(),
    inference_limits(),
    @monotonic_clock.MonotonicClock::system(),
  )
}

///|
fn prepare_foundation_pool(
  tokenizer : @tokenizer.TokenizerSpec,
  lane_count : Int,
) -> @request_admission.LunaRequestPreparationPool raise {
  @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer=tokenizer.digest(),
    expected_model=identity(),
    inference=inference_limits(),
    framed_limits=@framed_wire.FramedWireLimits::new(
      inference_limits(),
      max_frame_bytes=4096,
    ),
    clock=@monotonic_clock.MonotonicClock::system(),
    lane_count~,
    step_budget=@request_admission.LunaRequestPreparationStepBudget::new(17),
    work_limit=@request_admission.LunaRequestPreparationWorkLimit::new(
      1000000UL,
    ),
    storage_budget=@request_admission.LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=1024UL,
    ),
  )
}

///|
fn submit_foundation_pool_work(
  pool : @request_admission.LunaRequestPreparationPool,
  received : @request_admission.ReceivedRequest,
) -> @request_admission.LunaRequestPreparationWork raise {
  let admission = pool.try_submit(received)
  match admission.kind() {
    LunaRequestPreparationAdmitted => admission.work()
    LunaRequestPreparationSaturated =>
      raise Failure("idle Luna preparation lane reported saturation")
    LunaRequestPreparationDraining =>
      raise Failure("live Luna preparation pool reported draining")
  }
}

///|
fn finish_foundation_pool_work(
  pool : @request_admission.LunaRequestPreparationPool,
  work : @request_admission.LunaRequestPreparationWork,
) -> @request_admission.LunaPreparedRequest raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match work.state() {
      LunaRequestPreparationReady => return work.take_prepared()
      LunaRequestPreparationFailed => raise work.failure()
      LunaRequestPreparationCancelRequested =>
        raise Failure("Luna preparation work was unexpectedly cancelled")
      LunaRequestPreparationPending => {
        assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
        assert_true(work.last_work_units() <= 17)
        assert_true(work.last_work_units() >= 0)
      }
    }
    attempts += 1
  }
  raise Failure("Luna preparation work did not finish within its work limit")
}

///|
fn prepare_foundation_pool_request(
  pool : @request_admission.LunaRequestPreparationPool,
  received : @request_admission.ReceivedRequest,
) -> @request_admission.LunaPreparedRequest raise {
  finish_foundation_pool_work(pool, submit_foundation_pool_work(pool, received))
}

///|
fn begin_foundation_session(
  instance : @online_session.LunaOnlineInstance,
  prepared : @request_admission.LunaPreparedRequest,
) -> FoundationSession raise {
  let adapter = @framed_wire.LunaFramedEventAdapter::new(
    @framed_wire.FramedWireLimits::new(inference_limits(), max_frame_bytes=4096),
  )
  let admission = instance.begin(prepared)
  match admission.kind() {
    LunaOnlineInstanceAdmitted =>
      { instance, ticket: admission.ticket(), adapter, credit: None }
    LunaOnlineInstanceBusy =>
      raise Failure("idle online instance rejected a request as busy")
    LunaOnlineInstanceDraining =>
      raise Failure("live online instance rejected a request as draining")
  }
}

///|
fn begin_default_foundation_session(
  instance : @online_session.LunaOnlineInstance,
  received : @request_admission.ReceivedRequest,
) -> FoundationSession raise {
  begin_foundation_session(
    instance,
    prepare_foundation_request(session_tokenizer(), received),
  )
}

///|
fn FoundationSession::take_credit(
  self : FoundationSession,
) -> @online_session.LunaOnlineEventCredit raise {
  match self.credit {
    Some(credit) => credit
    None => {
      let credit = self.instance.take_event(self.ticket)
      self.credit = Some(credit)
      credit
    }
  }
}

///|
fn FoundationSession::ack_event(self : FoundationSession) -> Unit raise {
  self.take_credit().ack()
  self.credit = None
}

///|
fn FoundationSession::progress(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceProgress raise {
  self.instance.progress(self.ticket)
}

///|
fn FoundationSession::progress_terminalization(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceProgress raise {
  self.instance.progress_terminalization(self.ticket)
}

///|
fn FoundationSession::request_cancel(self : FoundationSession) -> Unit raise {
  self.instance.request_cancel(self.ticket)
}

///|
fn FoundationSession::check_deadline(self : FoundationSession) -> Unit raise {
  self.instance.check_deadline(self.ticket)
}

///|
fn FoundationSession::begin_abort(self : FoundationSession) -> Unit raise {
  self.instance.begin_abort(self.ticket)
  self.credit = None
}

///|
fn FoundationSession::progress_retirement(
  self : FoundationSession,
) -> @online_session.LunaOnlineInstanceRetirementProgress raise {
  self.instance.progress_request_retirement(self.ticket)
}

///|
fn drain_foundation_instance(
  instance : @online_session.LunaOnlineInstance,
) -> Unit raise {
  instance.begin_drain()
  let mut attempts = 0
  while attempts < 1000000 {
    match instance.progress_shutdown() {
      LunaOnlineInstanceShutdownClosed => return
      LunaOnlineInstanceShutdownPending => attempts += 1
    }
  }
  raise Failure("online instance did not complete bounded shutdown")
}

///|
fn progress_foundation_to_idle(session : FoundationSession) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match session.progress_retirement() {
      LunaOnlineInstanceIdle => return
      LunaOnlineInstanceRetirementPending => attempts += 1
      LunaOnlineInstanceClosed =>
        raise Failure("healthy online request closed its persistent instance")
    }
  }
  raise Failure("online request did not complete bounded retirement")
}
