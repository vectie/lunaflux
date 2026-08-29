///|
fn tcp_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=4,
    max_new_tokens=2,
    max_context_tokens=4,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=2,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=60000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn tcp_framed_limits() -> @framed_wire.FramedWireLimits raise {
  @framed_wire.FramedWireLimits::new(
    tcp_inference_limits(),
    max_frame_bytes=4096,
  )
}

///|
fn tcp_startup(
  fixture_root? : String = "/private/tmp",
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source_for_root(fixture_root).digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=worker_limits(),
    inference_limits=tcp_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn tcp_service_binding(
  fixture_root? : String = "/private/tmp",
) -> @worker_service.WorkerServiceBinding raise {
  @worker_service.WorkerServiceBinding::new(
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=bootstrap_source_for_root(fixture_root).digest(),
    device_ordinal=0,
    inference_limits=tcp_inference_limits(),
  )
}

///|
fn tcp_request_frame_for_identity(
  request_id : UInt64,
  model_identity : @spec.ModelIdentity,
  max_new_tokens? : Int = 1,
) -> FixedArray[Byte] raise {
  let limits = tcp_inference_limits()
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
    deadline=@inference.DeadlineBudget::from_milliseconds(10000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(tcp_framed_limits()).encode(
    request,
  )
  let bytes = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(bytes, destination_offset=0)
  bytes
}

///|
fn tcp_request_frame(
  request_id : UInt64,
  max_new_tokens? : Int = 1,
) -> FixedArray[Byte] raise {
  tcp_request_frame_for_identity(request_id, identity(), max_new_tokens~)
}

///|
fn prepare_tcp_coordinator(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> @online_session.LunaOnlineFramedCoordinatorPreparation raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_luna_online_framed_coordinator_approved(
    tokenizer,
    tokenizer.digest(),
    identity(),
    tcp_inference_limits(),
    tcp_framed_limits(),
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
    tcp_service_binding(),
    @worker_executable_fixture.admit(executable),
    tcp_startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
    restart_policy(),
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), LunaOnlineInstanceReady)
  prepared
}

///|
fn tcp_limits(
  read_chunk_bytes : Int,
  accept_timeout_millis? : Int = 1000,
) -> @online_tcp.LunaOnlineTcpLimits raise {
  @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes~,
    write_chunk_bytes=17,
    accept_timeout_millis~,
    input_idle_timeout_millis=5000,
    write_timeout_millis=5000,
    reactor_transition_budget=1,
  )
}

///|
async fn tcp_write_fixed(
  connection : @socket.Tcp,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Unit {
  let bytes = Bytes::makei(length, index => source[source_offset + index])
  let mut cursor = 0
  while cursor < length {
    let written = connection.write_once(
      bytes,
      offset=cursor,
      len=length - cursor,
    )
    if written <= 0 || written > length - cursor {
      raise Failure("loopback client failed to write an exact framed range")
    }
    cursor += written
  }
}

///|
fn endpoint_read_u32(source : FixedArray[Byte], offset : Int) -> Int {
  source[offset].to_int() |
  (source[offset + 1].to_int() << 8) |
  (source[offset + 2].to_int() << 16) |
  (source[offset + 3].to_int() << 24)
}

///|
async fn tcp_read_event(connection : @socket.Tcp) -> @inference.StreamEvent {
  let storage = FixedArray::make(4096, b'\x00')
  let mut cursor = 0
  let mut declared = 16
  while cursor < declared {
    let count = connection.read(storage, offset=cursor, max_len=1)
    if count != 1 {
      raise Failure("loopback endpoint did not publish a complete event")
    }
    cursor += 1
    if cursor == 16 {
      declared = endpoint_read_u32(storage, 12)
      if declared < 16 || declared > storage.length() {
        raise Failure("loopback endpoint published an invalid frame length")
      }
    }
  }
  @framed_wire.EventFrameBuffer::new(tcp_framed_limits())
  .load(storage, declared)
  .event()
}

///|
async fn endpoint_to_maintenance(
  endpoint : @online_tcp.LunaOnlineTcpEndpoint,
) -> Unit {
  let mut attempts = 0
  while attempts < 1000000 {
    match endpoint.progress_on_reactor() {
      LunaOnlineTcpMaintenanceRequired => return
      LunaOnlineTcpConnectedProgress
      | LunaOnlineTcpAdvanced
      | LunaOnlineTcpBackpressured => ()
      LunaOnlineTcpRejected =>
        raise Failure("loopback endpoint rejected an active request")
      LunaOnlineTcpCleanupRequired | LunaOnlineTcpClosedProgress =>
        raise Failure("loopback endpoint closed before event maintenance")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("loopback endpoint did not reach bounded maintenance")
}

///|
async fn endpoint_maintenance_to_reactor(
  endpoint : @online_tcp.LunaOnlineTcpEndpoint,
) -> Unit {
  let mut attempts = 0
  while attempts < 1000000 {
    match endpoint.progress_off_reactor_maintenance() {
      LunaOnlineTcpMaintenanceRequired => ()
      LunaOnlineTcpAdvanced | LunaOnlineTcpBackpressured => return
      LunaOnlineTcpCleanupRequired | LunaOnlineTcpRejected => return
      LunaOnlineTcpClosedProgress => return
      LunaOnlineTcpConnectedProgress =>
        raise Failure("unexpected endpoint maintenance disposition")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("loopback endpoint maintenance did not converge")
}

///|
async fn endpoint_to_rejection(
  endpoint : @online_tcp.LunaOnlineTcpEndpoint,
) -> Unit {
  let mut attempts = 0
  while attempts < 1000000 {
    match endpoint.progress_on_reactor() {
      LunaOnlineTcpRejected => return
      LunaOnlineTcpMaintenanceRequired =>
        endpoint_maintenance_to_reactor(endpoint)
      LunaOnlineTcpConnectedProgress
      | LunaOnlineTcpAdvanced
      | LunaOnlineTcpBackpressured
      | LunaOnlineTcpCleanupRequired => ()
      LunaOnlineTcpClosedProgress =>
        raise Failure("endpoint closed without publishing rejection credit")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("loopback endpoint did not publish queued rejection")
}

///|
async fn close_endpoint(endpoint : @online_tcp.LunaOnlineTcpEndpoint) -> Unit {
  let mut attempts = 0
  while endpoint.state() != LunaOnlineTcpClosed && attempts < 1000000 {
    match endpoint.progress_on_reactor() {
      LunaOnlineTcpMaintenanceRequired =>
        endpoint_maintenance_to_reactor(endpoint)
      _ => ()
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  if endpoint.state() != LunaOnlineTcpClosed {
    raise Failure("loopback endpoint did not release all retained owners")
  }
}

///|
async fn run_online_tcp_stream_evidence(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> Unit {
  let first = tcp_request_frame(970UL)
  let second = tcp_request_frame_for_identity(
    971UL,
    coordinator_foreign_identity(),
  )
  let remainder = FixedArray::make(
    first.length() - 1 + second.length(),
    b'\x00',
  )
  for index in 1..<first.length() {
    remainder[index - 1] = first[index]
  }
  for index in 0..<second.length() {
    remainder[first.length() - 1 + index] = second[index]
  }
  let endpoint = @online_tcp.bind_luna_online_tcp_single_endpoint(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_tcp_coordinator(executable, tokenizer),
    tcp_limits(remainder.length()),
  )
  let client = @socket.Tcp::connect(endpoint.local_addr())
  defer client.close()
  assert_eq(endpoint.progress_on_reactor(), LunaOnlineTcpConnectedProgress)

  tcp_write_fixed(client, first, source_offset=0, length=1)
  assert_eq(endpoint.progress_on_reactor(), LunaOnlineTcpAdvanced)
  tcp_write_fixed(client, remainder, source_offset=0, length=remainder.length())

  endpoint_to_maintenance(endpoint)
  match tcp_read_event(client) {
    Accepted(value) => assert_eq(value.model_identity(), identity())
    _ => raise Failure("endpoint first frame was not Accepted")
  }
  match tcp_read_event(client) {
    Token(value) => assert_eq(value.position(), 0)
    _ => raise Failure("endpoint second frame was not Token")
  }
  match tcp_read_event(client) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("endpoint third frame was not Usage")
  }
  endpoint_maintenance_to_reactor(endpoint)

  endpoint_to_maintenance(endpoint)
  match tcp_read_event(client) {
    Completed(value) => assert_eq(value.finish_reason(), TokenLimit)
    _ => raise Failure("endpoint fourth frame was not Completed")
  }
  endpoint_maintenance_to_reactor(endpoint)

  endpoint_to_rejection(endpoint)
  assert_eq(endpoint.rejection_sequence(), 2UL)
  assert_eq(endpoint.rejection_rule(), LunaOnlineTcpRejectedIdentity)
  let eof = FixedArray::make(1, b'\x00')
  assert_eq(client.read(eof, offset=0, max_len=1), 0)
  close_endpoint(endpoint)
}

///|
async fn run_online_tcp_disconnect_evidence(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> Unit {
  let frame = tcp_request_frame(972UL, max_new_tokens=2)
  let endpoint = @online_tcp.bind_luna_online_tcp_single_endpoint(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_tcp_coordinator(executable, tokenizer),
    tcp_limits(frame.length()),
  )
  let client = @socket.Tcp::connect(endpoint.local_addr())
  defer client.close()
  assert_eq(endpoint.progress_on_reactor(), LunaOnlineTcpConnectedProgress)
  tcp_write_fixed(client, frame, source_offset=0, length=frame.length())
  endpoint_to_maintenance(endpoint)
  match tcp_read_event(client) {
    Accepted(_) => ()
    _ => raise Failure("disconnect fixture did not publish Accepted")
  }
  endpoint.disconnect()
  close_endpoint(endpoint)
}

///|
async fn run_online_tcp_accept_timeout_evidence(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
) -> Unit {
  let endpoint = @online_tcp.bind_luna_online_tcp_single_endpoint(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_tcp_coordinator(executable, tokenizer),
    tcp_limits(4096, accept_timeout_millis=1),
  )
  assert_eq(endpoint.progress_on_reactor(), LunaOnlineTcpCleanupRequired)
  assert_eq(endpoint.failure_rule(), LunaOnlineTcpTimeout)
  close_endpoint(endpoint)
}

///|
async fn run_online_tcp_endpoint_evidence(executable : Bytes) -> Unit {
  let tokenizer = session_tokenizer()
  run_online_tcp_stream_evidence(executable, tokenizer)
  run_online_tcp_disconnect_evidence(executable, tokenizer)
  run_online_tcp_accept_timeout_evidence(executable, tokenizer)
}
