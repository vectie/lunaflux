///|
fn coordinator_request_frame_for_identity(
  request_id : UInt64,
  model_identity : @spec.ModelIdentity,
  max_new_tokens? : Int = 1,
) -> FixedArray[Byte] raise {
  let limits = inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity~,
    input=@inference.Input::from_token_ids([1, 2], limits),
    max_new_tokens~,
    context_ceiling=4,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(request_id),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(1000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(outbound_framed_limits()).encode(
    request,
  )
  let bytes = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(bytes, destination_offset=0)
  bytes
}

///|
fn coordinator_request_frame(
  request_id : UInt64,
  max_new_tokens? : Int = 1,
) -> FixedArray[Byte] raise {
  coordinator_request_frame_for_identity(
    request_id,
    identity(),
    max_new_tokens~,
  )
}

///|
fn coordinator_foreign_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "bbcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn prepare_framed_coordinator(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> @online_session.LunaOnlineFramedCoordinator raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_framed_coordinator(
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
fn offer_complete_coordinator_frame(
  coordinator : @online_session.LunaOnlineFramedCoordinator,
  frame : FixedArray[Byte],
) -> Unit raise {
  let mut cursor = 0
  while cursor < frame.length() {
    let ingress = coordinator.offer_luna_framed(
      frame,
      source_offset=cursor,
      length=frame.length() - cursor,
    )
    match ingress.kind() {
      LunaOnlineFramedIngressAccepted => {
        let consumed = ingress.consumed_bytes()
        assert_true(consumed > 0)
        assert_true(consumed <= frame.length() - cursor)
        cursor += consumed
      }
      LunaOnlineFramedIngressBackpressured => {
        assert_eq(ingress.consumed_bytes(), 0)
        ignore(coordinator.progress())
      }
      LunaOnlineFramedIngressDraining =>
        raise Failure("live coordinator drained while receiving a frame")
    }
  }
}

///|
fn progress_coordinator_to_event(
  coordinator : @online_session.LunaOnlineFramedCoordinator,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match coordinator.progress() {
      LunaOnlineFramedCoordinatorEventReady => return
      LunaOnlineFramedCoordinatorMaintenanceRequired =>
        ignore(coordinator.progress_off_reactor_maintenance())
      LunaOnlineFramedCoordinatorRejectionReady =>
        raise Failure("valid framed request was rejected")
      LunaOnlineFramedCoordinatorClosed =>
        raise Failure("coordinator closed before publishing an event")
      LunaOnlineFramedCoordinatorAwaitingInput =>
        raise Failure("complete framed request unexpectedly awaited input")
      LunaOnlineFramedCoordinatorIdle
      | LunaOnlineFramedCoordinatorAdvanced
      | LunaOnlineFramedCoordinatorEventPreparing
      | LunaOnlineFramedCoordinatorDraining => ()
    }
    attempts += 1
  }
  raise Failure("coordinator did not publish bounded framed output")
}

///|
fn copy_and_confirm_coordinator_event(
  coordinator : @online_session.LunaOnlineFramedCoordinator,
  prove_partial? : Bool = false,
) -> (FixedArray[Byte], @online_session.LunaOnlineFramedCoordinatorProgress) raise {
  let length = coordinator.framed_event_remaining()
  assert_true(length > 0)
  let output = FixedArray::make(length, b'\x00')
  let mut cursor = 0
  let mut first = true
  let mut final_progress = @online_session.LunaOnlineFramedCoordinatorEventReady
  while cursor < length {
    let offer = coordinator.copy_framed_event_chunk_to(
      output,
      destination_offset=cursor,
      maximum_length=17,
    )
    let offered = offer.length()
    assert_true(offered > 0)
    assert_true(offered <= 17)
    assert_eq(coordinator.progress(), LunaOnlineFramedCoordinatorEventReady)
    let confirmed = if prove_partial && first && offered > 1 {
      1
    } else {
      offered
    }
    final_progress = offer.confirm(length=confirmed)
    cursor += confirmed
    try ignore(offer.confirm(length=confirmed)) catch {
      LunaOnlineFramedCoordinatorFailed(LunaOnlineFramedEvent) => ()
      _ => raise Failure("stale framed offer raised the wrong error")
    } noraise {
      _ => raise Failure("a confirmed framed offer remained live")
    }
    first = false
  }
  (output, final_progress)
}

///|
fn coordinator_event(bytes : FixedArray[Byte]) -> @inference.StreamEvent raise {
  @framed_wire.EventFrameBuffer::new(outbound_framed_limits())
  .load(bytes, bytes.length())
  .event()
}

///|
fn finish_coordinator_disconnect(
  coordinator : @online_session.LunaOnlineFramedCoordinator,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    if coordinator.is_closed() {
      return
    }
    match coordinator.progress() {
      LunaOnlineFramedCoordinatorMaintenanceRequired =>
        ignore(coordinator.progress_off_reactor_maintenance())
      LunaOnlineFramedCoordinatorClosed => return
      _ => ()
    }
    attempts += 1
  }
  raise Failure("disconnected coordinator did not release all owners")
}

///|
fn run_online_framed_coordinator_evidence(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let coordinator = prepare_framed_coordinator(executable, tokenizer)
  offer_complete_coordinator_frame(
    coordinator,
    coordinator_request_frame(960UL),
  )
  // Queue a structurally valid frame with the wrong bound identity behind the
  // live request. Its preparation may fail early, but the single ordered
  // stream must not publish that later rejection before request 960 retires.
  offer_complete_coordinator_frame(
    coordinator,
    coordinator_request_frame_for_identity(
      961UL,
      coordinator_foreign_identity(),
    ),
  )

  progress_coordinator_to_event(coordinator)
  let (accepted_bytes, accepted_progress) = copy_and_confirm_coordinator_event(
    coordinator,
    prove_partial=true,
  )
  assert_eq(accepted_progress, LunaOnlineFramedCoordinatorAdvanced)
  match coordinator_event(accepted_bytes) {
    Accepted(value) => {
      assert_eq(value.model_identity(), identity())
      assert_eq(value.effective_limits().max_new_tokens(), 1)
      assert_eq(value.effective_limits().context_ceiling(), 4)
    }
    _ => raise Failure("coordinator first event was not Accepted")
  }

  progress_coordinator_to_event(coordinator)
  let (token_bytes, token_progress) = copy_and_confirm_coordinator_event(
    coordinator,
  )
  assert_eq(token_progress, LunaOnlineFramedCoordinatorAdvanced)
  match coordinator_event(token_bytes) {
    Token(value) => {
      assert_eq(value.token_id(), 0)
      assert_eq(value.position(), 0)
    }
    _ => raise Failure("coordinator second event was not Token")
  }

  progress_coordinator_to_event(coordinator)
  let (usage_bytes, usage_progress) = copy_and_confirm_coordinator_event(
    coordinator,
  )
  assert_eq(usage_progress, LunaOnlineFramedCoordinatorMaintenanceRequired)
  match coordinator_event(usage_bytes) {
    Usage(value) => {
      assert_eq(value.input_tokens(), 2)
      assert_eq(value.cached_input_tokens(), 0)
      assert_eq(value.output_tokens(), 1)
    }
    _ => raise Failure("coordinator third event was not Usage")
  }
  // The exact byte confirmation consumes only the transport Offer. Usage
  // semantic ACK remains pinned until the off-reactor maintenance boundary.
  assert_eq(
    coordinator.progress(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_eq(
    coordinator.progress(),
    LunaOnlineFramedCoordinatorMaintenanceRequired,
  )
  assert_eq(
    coordinator.progress_off_reactor_maintenance(),
    LunaOnlineFramedCoordinatorAdvanced,
  )

  progress_coordinator_to_event(coordinator)
  let (completed_bytes, completed_progress) = copy_and_confirm_coordinator_event(
    coordinator,
  )
  assert_eq(completed_progress, LunaOnlineFramedCoordinatorMaintenanceRequired)
  match coordinator_event(completed_bytes) {
    Completed(value) => {
      assert_eq(value.finish_reason(), TokenLimit)
      assert_eq(value.final_usage().output_tokens(), 1)
    }
    _ => raise Failure("coordinator terminal event was not Completed")
  }
  while coordinator.progress_off_reactor_maintenance() ==
        LunaOnlineFramedCoordinatorMaintenanceRequired {
    ()
  }

  let mut rejection_attempts = 0
  while rejection_attempts < 1000000 {
    match coordinator.progress() {
      LunaOnlineFramedCoordinatorRejectionReady => break
      LunaOnlineFramedCoordinatorMaintenanceRequired =>
        ignore(coordinator.progress_off_reactor_maintenance())
      LunaOnlineFramedCoordinatorEventReady =>
        raise Failure("later invalid request displaced the first request")
      _ => ()
    }
    rejection_attempts += 1
  }
  assert_true(rejection_attempts < 1000000)
  let rejection = coordinator.take_rejection()
  assert_eq(rejection.sequence(), 2UL)
  assert_eq(rejection.rule(), LunaOnlineFramedRejectedIdentity)
  rejection.ack()

  // Reuse the live owner, then cut a request after its first accepted byte.
  // Reactor progress drains that partial receipt before shutdown is allowed
  // onto the off-reactor maintenance path.
  let partial = coordinator_request_frame(962UL)
  let ingress = coordinator.offer_luna_framed(
    partial,
    source_offset=0,
    length=1,
  )
  assert_eq(ingress.kind(), LunaOnlineFramedIngressAccepted)
  assert_eq(ingress.consumed_bytes(), 1)
  coordinator.disconnect()
  let after_cut = coordinator.offer_luna_framed(
    partial,
    source_offset=1,
    length=partial.length() - 1,
  )
  assert_eq(after_cut.kind(), LunaOnlineFramedIngressDraining)
  assert_eq(after_cut.consumed_bytes(), 0)
  finish_coordinator_disconnect(coordinator)
  assert_true(coordinator.is_closed())

  // A second real owner cuts after Accepted bytes and their semantic credit
  // are confirmed, while the active request is awaiting its Token. Disconnect
  // must abort the exact ticket and close only through off-reactor retirement.
  let active = prepare_framed_coordinator(executable, tokenizer)
  offer_complete_coordinator_frame(active, coordinator_request_frame(963UL))
  progress_coordinator_to_event(active)
  let (_, active_accepted_progress) = copy_and_confirm_coordinator_event(active)
  assert_eq(active_accepted_progress, LunaOnlineFramedCoordinatorAdvanced)
  ignore(active.progress())
  active.disconnect()
  finish_coordinator_disconnect(active)
  assert_true(active.is_closed())
}
