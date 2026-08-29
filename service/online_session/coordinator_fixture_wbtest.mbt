///|
fn coordinator_fixture_inference() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=16,
    max_context_tokens=64,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=1000UL,
    max_top_k=256,
    max_temperature=4.0,
  )
}

///|
fn coordinator_fixture_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256("0".repeat(64)),
    @spec.PlanDigest::from_sha256("1".repeat(64)),
  )
}

///|
fn coordinator_fixture_tokenizer() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 64,
      max_output_tokens: 64,
      max_decoded_bytes: 256,
      max_vocab_entries: 256,
      max_merge_rules: 4,
      max_token_bytes: 1,
      max_special_tokens: 1,
    },
  )
}

///|
fn coordinator_fixture_request() -> @inference.GenerateRequest raise {
  let limits = coordinator_fixture_inference()
  @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(7UL),
    model_identity=coordinator_fixture_identity(),
    input=@inference.Input::from_utf8(b"abc", limits),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(9UL),
    stops=@inference.StopConditions::new([9], [b"END"], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(1000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
}

///|
fn coordinator_fixture_frame() -> FixedArray[Byte] raise {
  let limits = @framed_wire.FramedWireLimits::new(
    coordinator_fixture_inference(),
    max_frame_bytes=4096,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(limits).encode(
    coordinator_fixture_request(),
  )
  let output = FixedArray::make(frame.length(), b'\x00')
  ignore(frame.copy_to(output, destination_offset=0))
  output
}

///|
fn coordinator_fixture_instance(
  multi_lane_count? : Int = 1,
) -> LunaOnlineInstance raise {
  let inference = coordinator_fixture_inference()
  {
    expected_tokenizer: coordinator_fixture_tokenizer().digest(),
    expected_model: coordinator_fixture_identity(),
    inference,
    claim: None,
    lease: None,
    events: @luna_event.LunaEventOwner::new(inference),
    output: FixedArray::make(inference.max_decoded_delta_bytes(), b'\x00'),
    input_tokens: 0,
    cached_input_tokens: 0,
    request_epoch: 0UL,
    event_epoch: 0UL,
    event_kind: NoSessionEvent,
    event_taken: false,
    accepted_acked: false,
    output_tokens: 0,
    final_tail_length: 0,
    finish_kind: FinishPending,
    string_cut: NoStringCut,
    terminal_cut: NoTerminalCut,
    deferred_cancel: false,
    deadline_code: deadline_failure_code(),
    output_code: output_failure_code(),
    worker_code: worker_failure_code(),
    failure_kind: NoLatchedFailure,
    phase: LunaInstanceIdlePhase,
    maintenance_state: 0,
    restart_observation_pending: false,
    restarted_failure_terminal_pending: false,
    drain_requested: false,
    recovery_close: false,
    multi: new_luna_online_multi_state(multi_lane_count, inference),
  }
}

///|
fn coordinator_fixture(
  multi_lane_count? : Int = 1,
) -> LunaOnlineFramedCoordinator raise {
  let inference = coordinator_fixture_inference()
  let tokenizer = coordinator_fixture_tokenizer()
  let limits = @framed_wire.FramedWireLimits::new(
    inference,
    max_frame_bytes=4096,
  )
  let clock = @monotonic_clock.MonotonicClock::system()
  let pool = @request_admission.LunaRequestPreparationPool::new(
    tokenizer,
    expected_tokenizer=tokenizer.digest(),
    expected_model=coordinator_fixture_identity(),
    inference~,
    framed_limits=limits,
    clock~,
    lane_count=2,
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
  let event_budget = @framed_wire.LunaFramedEventStepBudget::new(17)
  let service = new_luna_online_framed_coordinator(
    pool,
    coordinator_fixture_instance(multi_lane_count~),
    @framed_wire.LunaFramedEventWorkspace::new(limits, event_budget),
    event_budget.work_units(),
    inference,
    limits,
    coordinator_fixture_identity(),
    clock,
  )
  let opened = service.open_stream()
  { service, stream: service.take_open_stream(opened) }
}
