///|
fn pipeline_startup_with_limits(
  worker_limits : @worker_protocol.WorkerProtocolLimits,
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
    worker_limits~,
    inference_limits=tcp_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn pipeline_startup(
  fixture_root? : String = "/private/tmp",
) -> @worker_wire.WorkerStartupContract raise {
  pipeline_startup_with_limits(multi_worker_limits(), fixture_root~)
}

///|
fn prepare_pipeline_service_with_blueprint(
  executable : Bytes,
  blueprint : @scheduler.SchedulerBlueprint,
  startup : @worker_wire.WorkerStartupContract,
  fixture_root? : String = "/private/tmp",
) -> @online_session.LunaOnlineFramedServicePreparation raise {
  let tokenizer = session_tokenizer()
  let model = @approved_fs.ApprovedRoot::open_absolute(fixture_root)
  let kernel = @approved_fs.ApprovedRoot::open_absolute(fixture_root)
  let prepared = @online_session.prepare_owned_luna_online_framed_service_approved(
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
    blueprint,
    tcp_service_binding(fixture_root~),
    @worker_executable_fixture.admit(executable),
    startup,
    bootstrap_source_for_root(fixture_root),
    reusable_server_process_limits(),
    model,
    kernel,
    restart_policy(),
  )
  model.close()
  kernel.close()
  match prepared.state() {
    LunaOnlineInstanceReady => ()
    LunaOnlineInstanceCleanupRequired =>
      raise Failure("pipeline service preparation required cleanup")
  }
  prepared
}

///|
fn prepare_pipeline_service(
  executable : Bytes,
  fixture_root? : String = "/private/tmp",
) -> @online_session.LunaOnlineFramedServicePreparation raise {
  prepare_pipeline_service_with_blueprint(
    executable,
    multi_scheduler_blueprint(),
    pipeline_startup(fixture_root~),
    fixture_root~,
  )
}

///|
fn pipeline_two_frames(
  first : FixedArray[Byte],
  second : FixedArray[Byte],
) -> FixedArray[Byte] {
  let combined = FixedArray::make(first.length() + second.length(), b'\x00')
  for index in 0..<first.length() {
    combined[index] = first[index]
  }
  for index in 0..<second.length() {
    combined[first.length() + index] = second[index]
  }
  combined
}

///|
async fn progress_pipeline_pair(
  server : @online_tcp.LunaOnlineTcpServer,
) -> Unit {
  let mut cancelled = false
  let mut saw_two_live = false
  let mut saw_two_row_plan = false
  let mut attempts = 0
  while attempts < 1000000 {
    let progress = server.progress_on_reactor()
    match progress {
      LunaOnlineTcpServerObservationRecorded
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      LunaOnlineTcpServerConnectionRetired =>
        raise Failure("pipeline retired before both requests completed")
      LunaOnlineTcpServerListeningProgress
      | LunaOnlineTcpServerAcceptTimedOut =>
        raise Failure("pipeline reopened accept while the peer was live")
      LunaOnlineTcpServerDrainingProgress | LunaOnlineTcpServerClosedProgress =>
        raise Failure("pipeline drained while valid requests were live")
    }
    let metrics = server.metrics_snapshot()
    let admissions = metrics.counter_value(Admissions)
    let live = metrics.gauge_value(ActiveRequests) +
      metrics.gauge_value(QueueDepth)
    if admissions == 2UL && live == 2 {
      saw_two_live = true
    }
    if metrics.histogram_bucket_value(BatchRows, 1) > 0UL &&
      metrics.histogram_bucket_value(BatchTokens, 1) > 0UL {
      saw_two_row_plan = true
    }
    if saw_two_live && saw_two_row_plan && !cancelled {
      server.request_cancel()
      cancelled = true
    }
    if progress == LunaOnlineTcpServerConnectedProgress &&
      cancelled &&
      metrics.counter_value(Cancellations) == 1UL &&
      metrics.counter_value(Completions) == 1UL &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 {
      assert_true(saw_two_live)
      assert_true(saw_two_row_plan)
      return
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("pipeline pair did not reach balanced terminal state")
}

///|
async fn progress_pipeline_single(
  server : @online_tcp.LunaOnlineTcpServer,
  expected_admissions : UInt64,
  expected_completions : UInt64,
) -> Unit {
  let mut attempts = 0
  while attempts < 1000000 {
    let progress = server.progress_on_reactor()
    match progress {
      LunaOnlineTcpServerObservationRecorded
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      LunaOnlineTcpServerConnectionRetired =>
        raise Failure("valid scanned frame retired before completion")
      LunaOnlineTcpServerListeningProgress
      | LunaOnlineTcpServerAcceptTimedOut =>
        raise Failure("valid scanned frame reopened accept before completion")
      LunaOnlineTcpServerDrainingProgress | LunaOnlineTcpServerClosedProgress =>
        raise Failure("valid scanned frame poisoned the reusable server")
    }
    let metrics = server.metrics_snapshot()
    if progress == LunaOnlineTcpServerConnectedProgress &&
      metrics.counter_value(Admissions) == expected_admissions &&
      metrics.counter_value(Completions) == expected_completions &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 {
      return
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("valid scanned frame did not reach its balanced boundary")
}

///|
async fn run_luna_pipeline_server_evidence(executable : Bytes) -> Unit {
  let first = tcp_request_frame(990UL)
  let second = tcp_request_frame(991UL)
  let combined = pipeline_two_frames(first, second)
  let server = @online_tcp.bind_luna_online_tcp_pipeline_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_pipeline_service(executable),
    reusable_server_limits(combined.length()),
    log_capacity=64,
  )
  let control_preparation = @online_tcp.prepare_luna_online_http_control_for_native(
    server,
    @http1.LunaHttp1ControlLimits::new(max_head_bytes=256),
  )
  let control = @online_tcp.bind_luna_online_http_control_server(
    control_preparation,
    reusable_server_limits(256),
  )
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlReady)
  let client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlNotReady)
  tcp_write_fixed(client, combined, source_offset=0, length=combined.length())
  progress_pipeline_pair(server)

  // The globally ordered stream admits both requests before generated output.
  guard tcp_read_event(client) is Accepted(_) else {
    raise Failure("pipeline first event was not Accepted")
  }
  guard tcp_read_event(client) is Accepted(_) else {
    raise Failure("pipeline second event was not Accepted")
  }
  guard tcp_read_event(client) is Usage(first_usage) else {
    raise Failure("cancelled pipeline request lost Usage")
  }
  assert_eq(first_usage.output_tokens(), 0)
  guard tcp_read_event(client) is Completed(first_completed) else {
    raise Failure("cancelled pipeline request lost Completed")
  }
  assert_eq(first_completed.finish_reason(), Cancelled)
  guard tcp_read_event(client) is Token(second_token) else {
    raise Failure("surviving pipeline request lost Token")
  }
  assert_eq(second_token.position(), 0)
  guard tcp_read_event(client) is Usage(second_usage) else {
    raise Failure("surviving pipeline request lost Usage")
  }
  assert_eq(second_usage.output_tokens(), 1)
  guard tcp_read_event(client) is Completed(second_completed) else {
    raise Failure("surviving pipeline request lost Completed")
  }
  assert_eq(second_completed.finish_reason(), TokenLimit)

  // Peer close retires only this Stream; the same listener remains healthy.
  client.close()
  assert_eq(progress_server_to_retired(server), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlReady)
  control.begin_drain()
  assert_eq(
    control.progress_on_reactor(),
    LunaOnlineHttpControlDrainingProgress,
  )
  assert_eq(control.progress_on_reactor(), LunaOnlineHttpControlClosedProgress)

  // A full malformed frame leaves bytes behind its 16-byte prefix while the
  // lower scanner detects invalid magic. The server must preserve tail
  // backpressure until the coordinator publishes one typed rejection, retire
  // only this connection, and keep the listener/worker reusable.
  let malformed = tcp_request_frame(992UL)
  malformed[0] = malformed[0] ^ b'\x01'
  let rejected = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(
    rejected,
    malformed,
    source_offset=0,
    length=malformed.length(),
  )
  assert_eq(progress_server_to_retired(server), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  assert_eq(server.health(), LunaOnlineTcpServerHealthy)
  let rejected_metrics = server.metrics_snapshot()
  assert_eq(rejected_metrics.counter_value(NetworkRejections), 1UL)
  assert_eq(rejected_metrics.counter_value(NetworkAccepts), 2UL)
  assert_eq(rejected_metrics.counter_value(NetworkDisconnects), 2UL)
  let rejected_eof = FixedArray::make(1, b'\x00')
  assert_eq(rejected.read(rejected_eof, offset=0, max_len=1), 0)
  rejected.close()

  // A complete valid frame on the replacement connection crosses the same
  // prefix-scanner backpressure boundary and proves listener/service reuse.
  let valid = tcp_request_frame(993UL)
  let replacement = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(replacement, valid, source_offset=0, length=valid.length())
  progress_pipeline_single(server, 3UL, 2UL)
  guard tcp_read_event(replacement) is Accepted(_) else {
    raise Failure("replacement valid frame lost Accepted")
  }
  guard tcp_read_event(replacement) is Token(token) else {
    raise Failure("replacement valid frame lost Token")
  }
  assert_eq(token.position(), 0)
  guard tcp_read_event(replacement) is Usage(usage) else {
    raise Failure("replacement valid frame lost Usage")
  }
  assert_eq(usage.output_tokens(), 1)
  guard tcp_read_event(replacement) is Completed(completed) else {
    raise Failure("replacement valid frame lost Completed")
  }
  assert_eq(completed.finish_reason(), TokenLimit)
  replacement.close()
  assert_eq(progress_server_to_retired(server), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  assert_eq(server.health(), LunaOnlineTcpServerHealthy)

  drain_reusable_server(server)
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(NetworkAccepts), 3UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 3UL)
  assert_eq(metrics.counter_value(NetworkRejections), 1UL)
  assert_eq(metrics.counter_value(Admissions), 3UL)
  assert_eq(metrics.counter_value(Cancellations), 1UL)
  assert_eq(metrics.counter_value(Completions), 2UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(metrics.gauge_value(KvPagesFree), 8)
}
