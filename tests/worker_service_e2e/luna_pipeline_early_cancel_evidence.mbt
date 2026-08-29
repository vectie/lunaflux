///|
async fn progress_single_row_early_cancel_pair(
  server : @online_tcp.LunaOnlineTcpServer,
) -> Bool {
  let mut attempted = false
  let mut cancelled = false
  let mut attempts = 0
  while attempts < 1000000 {
    let progress = server.progress_on_reactor()
    match progress {
      LunaOnlineTcpServerObservationRecorded
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      _ => raise Failure("single-row early cancel left its connected stream")
    }
    let metrics = server.metrics_snapshot()
    let live = metrics.gauge_value(ActiveRequests) +
      metrics.gauge_value(QueueDepth)
    if !attempted && metrics.counter_value(Admissions) == 2UL && live == 2 {
      assert_eq(metrics.histogram_bucket_value(BatchRows, 1), 0UL)
      attempted = true
      cancelled = try {
        server.request_cancel()
        true
      } catch {
        LunaOnlineTcpFailed(LunaOnlineTcpCoordinator) => false
        error => raise error
      }
    }
    let expected_completions = if cancelled { 1UL } else { 2UL }
    let expected_cancellations = if cancelled { 1UL } else { 0UL }
    if attempted &&
      progress == LunaOnlineTcpServerConnectedProgress &&
      metrics.counter_value(Completions) == expected_completions &&
      metrics.counter_value(Cancellations) == expected_cancellations &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 {
      assert_eq(metrics.histogram_bucket_value(BatchRows, 1), 0UL)
      return cancelled
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("single-row early cancel did not reach terminal balance")
}

///|
fn pipeline_request_frame_with_deadline(
  request_id : UInt64,
  deadline_millis : UInt64,
) -> FixedArray[Byte] raise {
  let limits = tcp_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=identity(),
    input=@inference.Input::from_token_ids([1, 2], limits),
    max_new_tokens=1,
    context_ceiling=4,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(request_id),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(
      deadline_millis, limits,
    ),
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
async fn progress_single_row_deadline_pair(
  server : @online_tcp.LunaOnlineTcpServer,
) -> Unit {
  let mut delayed = false
  let mut attempts = 0
  while attempts < 1000000 {
    let progress = server.progress_on_reactor()
    match progress {
      LunaOnlineTcpServerObservationRecorded
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      _ => raise Failure("single-row deadline left its connected stream")
    }
    let metrics = server.metrics_snapshot()
    let live = metrics.gauge_value(ActiveRequests) +
      metrics.gauge_value(QueueDepth)
    if !delayed && metrics.counter_value(Admissions) == 2UL && live == 2 {
      delayed = true
      @async.sleep(75)
    }
    if progress == LunaOnlineTcpServerConnectedProgress &&
      metrics.counter_value(Admissions) == 2UL &&
      metrics.counter_value(Deadlines) == 1UL &&
      metrics.counter_value(Completions) == 1UL &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 {
      assert_eq(metrics.counter_value(Cancellations), 0UL)
      assert_eq(metrics.counter_value(WorkerFailures), 0UL)
      assert_eq(metrics.counter_value(Failures), 0UL)
      return
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("single-row deadline did not reach terminal balance")
}

///|
async fn assert_single_row_deadline_evidence(executable : Bytes) -> Unit {
  let first_id = 994UL
  let second_id = 995UL
  let combined = pipeline_two_frames(
    pipeline_request_frame_with_deadline(first_id, 50UL),
    pipeline_request_frame_with_deadline(second_id, 10000UL),
  )
  let server = @online_tcp.bind_luna_online_tcp_pipeline_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_pipeline_service_with_blueprint(
      executable,
      multi_scheduler_blueprint_for_rows(1),
      pipeline_startup_with_limits(multi_worker_limits_for_rows(1)),
    ),
    reusable_server_limits(combined.length()),
    log_capacity=64,
  )
  let client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client, combined, source_offset=0, length=combined.length())
  progress_single_row_deadline_pair(server)
  guard tcp_read_event(client) is Accepted(_) else {
    raise Failure("deadline first event was not Accepted")
  }
  guard tcp_read_event(client) is Accepted(_) else {
    raise Failure("deadline second event was not Accepted")
  }
  guard tcp_read_event(client) is Usage(first_usage) else {
    raise Failure("deadline request lost Usage")
  }
  assert_eq(first_usage.output_tokens(), 0)
  guard tcp_read_event(client) is Failed(first_failed) else {
    raise Failure("deadline request lost Failed")
  }
  assert_eq(first_failed.code().as_string(), "deadline_exceeded")
  guard tcp_read_event(client) is Token(_) else {
    raise Failure("deadline survivor lost Token")
  }
  guard tcp_read_event(client) is Usage(second_usage) else {
    raise Failure("deadline survivor lost Usage")
  }
  assert_eq(second_usage.output_tokens(), 1)
  guard tcp_read_event(client) is Completed(_) else {
    raise Failure("deadline survivor lost Completed")
  }
  client.close()
  assert_eq(progress_server_to_retired(server), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  drain_reusable_server(server)
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(WorkerFailures), 0UL)
  assert_eq(metrics.counter_value(Failures), 0UL)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(metrics.gauge_value(KvPagesFree), 8)
}

///|
async fn run_luna_pipeline_early_cancel_evidence(executable : Bytes) -> Bool {
  let first_id = 992UL
  let second_id = 993UL
  let combined = pipeline_two_frames(
    tcp_request_frame(first_id),
    tcp_request_frame(second_id),
  )
  let server = @online_tcp.bind_luna_online_tcp_pipeline_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_pipeline_service_with_blueprint(
      executable,
      multi_scheduler_blueprint_for_rows(1),
      pipeline_startup_with_limits(multi_worker_limits_for_rows(1)),
    ),
    reusable_server_limits(combined.length()),
    log_capacity=64,
  )
  let client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client, combined, source_offset=0, length=combined.length())
  let cancelled = progress_single_row_early_cancel_pair(server)
  assert_true(cancelled)
  let terminal_rejected = try {
    server.request_cancel()
    false
  } catch {
    LunaOnlineTcpFailed(LunaOnlineTcpCoordinator) => true
    error => raise error
  }
  assert_true(terminal_rejected)
  let ledger = Phase23SoakLedger::new(8)
  phase23_soak_read_wave(client, cancelled, first_id, second_id, ledger)
  client.close()
  assert_eq(progress_server_to_retired(server), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  drain_reusable_server(server)
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(Admissions), 2UL)
  assert_eq(metrics.counter_value(WorkerFailures), 0UL)
  assert_eq(metrics.counter_value(Failures), 0UL)
  assert_eq(
    metrics.counter_value(Cancellations),
    if cancelled {
      1UL
    } else {
      0UL
    },
  )
  assert_eq(
    metrics.counter_value(Completions),
    if cancelled {
      1UL
    } else {
      2UL
    },
  )
  assert_eq(metrics.histogram_bucket_value(BatchRows, 1), 0UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(metrics.gauge_value(KvPagesFree), 8)
  assert_single_row_deadline_evidence(executable)
  cancelled
}
