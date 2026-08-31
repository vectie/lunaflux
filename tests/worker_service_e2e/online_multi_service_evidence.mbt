///|
fn multi_worker_limits_for_rows(
  max_rows : Int,
) -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=max_rows,
    max_decode_rows=max_rows,
    max_plan_rows=max_rows,
    max_plan_tokens=2,
    max_plan_pages=4,
    max_capabilities=2,
    max_completion_slots=2,
    max_sequence_tokens=4,
    max_token_id=31,
  )
}

///|
fn multi_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  multi_worker_limits_for_rows(2)
}

///|
fn multi_scheduler_blueprint_for_rows(
  max_batch_rows : Int,
) -> @scheduler.SchedulerBlueprint raise {
  let scheduling = @scheduler_config.new(
    step_token_budget=max_batch_rows,
    max_active_requests=2,
    max_waiting_requests=4,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=2,
    output_event_capacity=2,
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
    max_batch_rows~,
    max_sequence_tokens=4,
    vocabulary_size=32,
    hidden_size=8,
    intermediate_size=16,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=4,
  )
  let limits = multi_worker_limits_for_rows(max_batch_rows)
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
fn multi_scheduler_blueprint() -> @scheduler.SchedulerBlueprint raise {
  multi_scheduler_blueprint_for_rows(2)
}

///|
fn multi_startup() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source().digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=multi_worker_limits(),
    inference_limits=inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn prepare_multi_framed_service(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> @online_session.LunaOnlineFramedService raise {
  let blueprint = multi_scheduler_blueprint()
  let startup_contract = multi_startup()
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_framed_service_approved(
    tokenizer,
    tokenizer.digest(),
    identity(),
    inference_limits(),
    outbound_framed_limits(),
    2,
    @request_admission.LunaRequestPreparationStepBudget::new(1),
    @request_admission.LunaRequestPreparationWorkLimit::new(1000000UL),
    @request_admission.LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=1024UL,
    ),
    @framed_wire.LunaFramedEventStepBudget::new(17),
    blueprint,
    service_binding(),
    @worker_executable_fixture.admit(executable),
    startup_contract,
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
    restart_policy(),
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), LunaOnlineInstanceReady)
  prepared.take_ready()
}

///|
fn offer_complete_stream_frame(
  stream : @online_session.LunaOnlineFramedStream,
  frame : FixedArray[Byte],
) -> Unit raise {
  let mut cursor = 0
  while cursor < frame.length() {
    let ingress = stream.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    match ingress.kind() {
      LunaOnlineFramedIngressAccepted => {
        let consumed = ingress.consumed_bytes()
        assert_true(consumed > 0)
        cursor += consumed
      }
      LunaOnlineFramedIngressBackpressured => {
        assert_eq(ingress.consumed_bytes(), 0)
        ignore(stream.progress())
      }
      LunaOnlineFramedIngressDraining =>
        raise Failure("multi stream drained during framed receipt")
    }
  }
}

///|
fn next_multi_stream_event(
  stream : @online_session.LunaOnlineFramedStream,
) -> @inference.StreamEvent raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match stream.progress() {
      LunaOnlineFramedCoordinatorEventReady => {
        let length = stream.framed_event_remaining()
        let output = FixedArray::make(length, b'\x00')
        let mut cursor = 0
        while cursor < length {
          let offer = stream.copy_framed_event_chunk_to(
            output,
            destination_offset=cursor,
            maximum_length=17,
          )
          let amount = offer.length()
          assert_true(amount > 0)
          ignore(offer.confirm(length=amount))
          cursor += amount
        }
        return coordinator_event(output)
      }
      LunaOnlineFramedCoordinatorObservationReady => {
        let observation = stream.take_observation()
        observation.ack()
      }
      LunaOnlineFramedCoordinatorRejectionReady =>
        raise Failure("valid multi request was rejected")
      LunaOnlineFramedCoordinatorSemanticEventReady =>
        raise Failure("native multi stream exposed semantic-only credit")
      LunaOnlineFramedCoordinatorClosed =>
        raise Failure("multi stream closed before terminal events")
      LunaOnlineFramedCoordinatorIdle
      | LunaOnlineFramedCoordinatorWaiting
      | LunaOnlineFramedCoordinatorAwaitingInput
      | LunaOnlineFramedCoordinatorAdvanced
      | LunaOnlineFramedCoordinatorEventPreparing
      | LunaOnlineFramedCoordinatorMaintenanceRequired
      | LunaOnlineFramedCoordinatorDraining => ()
    }
    attempts += 1
  }
  raise Failure("multi stream did not publish an event")
}

///|
fn run_online_multi_service_evidence(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let service = prepare_multi_framed_service(executable, tokenizer)
  let opened = service.open_stream()
  assert_eq(opened.kind(), LunaOnlineFramedOpened)
  let stream = service.take_open_stream(opened)

  // Both receipts are complete before the worker can finish either request.
  offer_complete_stream_frame(stream, coordinator_request_frame(970UL))
  offer_complete_stream_frame(stream, coordinator_request_frame(971UL))

  guard next_multi_stream_event(stream) is Accepted(_) else {
    raise Failure("first multi event was not Accepted")
  }
  guard next_multi_stream_event(stream) is Accepted(_) else {
    raise Failure("second request did not admit before token output")
  }
  let admitted = service.luna_telemetry()
  assert_eq(admitted.active_requests() + admitted.queue_depth(), 2)

  let mut saw_two_row_plan = false
  let mut tokens = 0
  let mut usages = 0
  let mut completed = 0
  let mut attempts = 0
  while completed < 2 && attempts < 1000000 {
    let plan = service.luna_plan_telemetry()
    if plan.last_plan_rows() == 2 && plan.last_plan_tokens() == 2 {
      saw_two_row_plan = true
    }
    match next_multi_stream_event(stream) {
      Token(_) => tokens += 1
      Usage(_) => usages += 1
      Completed(_) => completed += 1
      _ => raise Failure("unexpected event in multi terminal sequence")
    }
    attempts += 1
  }
  assert_true(saw_two_row_plan)
  assert_eq(tokens, 2)
  assert_eq(usages, 2)
  assert_eq(completed, 2)
  assert_eq(service.luna_telemetry().active_requests(), 0)

  stream.disconnect()
  service.begin_drain()
  let mut drain_attempts = 0
  while !service.is_closed() && drain_attempts < 1000000 {
    ignore(service.progress())
    drain_attempts += 1
  }
  assert_true(service.is_closed())
}
