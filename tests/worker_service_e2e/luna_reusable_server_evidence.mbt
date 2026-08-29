///|
fn reusable_server_process_limits() -> @worker_process.WorkerProcessLimits raise {
  let limits = worker_limits()
  let required_frame_bytes = @worker_wire.bootstrap_source_capacity()
    .max(@worker_wire.plan_frame_byte_capacity(limits))
    .max(@worker_wire.completion_frame_byte_capacity(limits))
  @worker_process.WorkerProcessLimits::new(
    max_frame_bytes=required_frame_bytes,
    startup_io_timeout_millis=5000,
    io_timeout_millis=5000,
    shutdown_timeout_millis=100,
  )
}

///|
fn prepare_tcp_service(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
  fixture_root? : String = "/private/tmp",
) -> @online_session.LunaOnlineFramedServicePreparation raise {
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
      reference_cells=1000000UL,
    ),
    @framed_wire.LunaFramedEventStepBudget::new(17),
    scheduler_blueprint(),
    tcp_service_binding(fixture_root~),
    @worker_executable_fixture.admit(executable),
    tcp_startup(fixture_root~),
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
      raise Failure("reusable service preparation required cleanup")
  }
  prepared
}

///|
fn reusable_server_limits(
  read_chunk_bytes : Int,
) -> @online_tcp.LunaOnlineTcpLimits raise {
  @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes~,
    write_chunk_bytes=17,
    accept_timeout_millis=25,
    input_idle_timeout_millis=5000,
    write_timeout_millis=5000,
    reactor_transition_budget=1,
  )
}

///|
async fn progress_server_to_retired(
  server : @online_tcp.LunaOnlineTcpServer,
  cancel_after_admission? : Bool = false,
) -> Int {
  let mut observations = 0
  let mut cancelled = false
  let mut attempts = 0
  while attempts < 1000000 {
    match server.progress_on_reactor() {
      LunaOnlineTcpServerObservationRecorded => {
        observations += 1
        if cancel_after_admission && !cancelled && observations == 1 {
          server.request_cancel()
          cancelled = true
        }
      }
      LunaOnlineTcpServerConnectionRetired => return observations
      LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      LunaOnlineTcpServerListeningProgress
      | LunaOnlineTcpServerAcceptTimedOut =>
        raise Failure("reusable server returned to accept before retirement")
      LunaOnlineTcpServerDrainingProgress | LunaOnlineTcpServerClosedProgress =>
        raise Failure("reusable server drained before connection retirement")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("reusable server connection did not retire")
}

///|
async fn drain_reusable_server(
  server : @online_tcp.LunaOnlineTcpServer,
) -> Unit {
  server.begin_drain()
  let first = server.progress_on_reactor()
  assert_eq(first, LunaOnlineTcpServerDrainingProgress)
  let mut attempts = 0
  while server.state() != LunaOnlineTcpServerClosed && attempts < 1000000 {
    match server.progress_on_reactor() {
      LunaOnlineTcpServerDrainingProgress
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerClosedProgress => ()
      _ => raise Failure("draining reusable server reopened transport work")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  assert_eq(server.state(), LunaOnlineTcpServerClosed)
}

///|
fn metrics_histogram_total(
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  histogram : @metrics.LunaInstanceHistogram,
) -> UInt64 raise {
  let mut total = 0UL
  for bucket in 0..<@metrics.luna_instance_histogram_bucket_count() {
    total += snapshot.histogram_bucket_value(histogram, bucket)
  }
  total
}

///|
fn log_event_count(
  snapshot : @instance_log.LunaInstanceLogSnapshot,
  event : @instance_log.LunaInstanceLogEvent,
) -> Int raise {
  let mut count = 0
  for index in 0..<snapshot.length() {
    if snapshot.event_at(index) == event {
      count += 1
    }
  }
  count
}

///|
async fn assert_healthy_server_sequence(executable : Bytes) -> Unit {
  let tokenizer = session_tokenizer()
  let first = tcp_request_frame(980UL)
  let foreign = tcp_request_frame_for_identity(
    981UL,
    coordinator_foreign_identity(),
  )
  let coalesced = FixedArray::make(
    first.length() - 1 + foreign.length(),
    b'\x00',
  )
  for index in 1..<first.length() {
    coalesced[index - 1] = first[index]
  }
  for index in 0..<foreign.length() {
    coalesced[first.length() - 1 + index] = foreign[index]
  }
  let preparation = prepare_tcp_service(executable, tokenizer)
  let server = @online_tcp.bind_luna_online_tcp_server(
    @socket.Addr::parse("127.0.0.1:0"),
    preparation,
    reusable_server_limits(coalesced.length()),
    log_capacity=64,
  )

  // A timed-out accept leaves the same listener and Service reusable.
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerAcceptTimedOut)
  assert_eq(server.state(), LunaOnlineTcpServerListening)

  // Connection one: one-byte prefix fragmentation plus a coalesced foreign
  // second frame. Only request 980 may be admitted on this connection.
  let client1 = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client1, first, source_offset=0, length=1)
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerAdvanced)
  tcp_write_fixed(
    client1,
    coalesced,
    source_offset=0,
    length=coalesced.length(),
  )
  // Admission, first-token latency, usage, and terminal latency are four
  // independently ACKed observations for the one-token request.
  assert_eq(progress_server_to_retired(server), 4)
  match tcp_read_event(client1) {
    Accepted(value) => assert_eq(value.model_identity(), identity())
    _ => raise Failure("reusable server did not publish Accepted")
  }
  match tcp_read_event(client1) {
    Token(value) => assert_eq(value.position(), 0)
    _ => raise Failure("reusable server did not publish Token")
  }
  match tcp_read_event(client1) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("reusable server did not publish Usage")
  }
  match tcp_read_event(client1) {
    Completed(value) => assert_eq(value.finish_reason(), TokenLimit)
    _ => raise Failure("reusable server did not publish Completed")
  }
  let eof1 = FixedArray::make(1, b'\x00')
  assert_eq(client1.read(eof1, offset=0, max_len=1), 0)
  client1.close()

  // Connection two: protocol-terminal identity rejection writes no bytes and
  // returns the same listener/worker to Listening.
  let client2 = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client2, foreign, source_offset=0, length=foreign.length())
  assert_eq(progress_server_to_retired(server), 0)
  let eof2 = FixedArray::make(1, b'\x00')
  assert_eq(client2.read(eof2, offset=0, max_len=1), 0)
  client2.close()

  // Connection three: cancellation is requested only after the Admission
  // observation has been durably recorded and ACKed by the server.
  let third = tcp_request_frame(982UL, max_new_tokens=2)
  let client3 = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client3, third, source_offset=0, length=third.length())
  let cancelled_observations = progress_server_to_retired(
    server,
    cancel_after_admission=true,
  )
  // The child may have completed zero, one, or two tokens before the admitted
  // cancellation is committed. All such token observations are bounded; the
  // client-visible cut below remains exact.
  assert_true(cancelled_observations >= 3 && cancelled_observations <= 5)
  match tcp_read_event(client3) {
    Accepted(_) => ()
    _ => raise Failure("cancelled reusable request lost Accepted")
  }
  match tcp_read_event(client3) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("cancelled reusable request lost Usage")
  }
  match tcp_read_event(client3) {
    Completed(value) => assert_eq(value.finish_reason(), Cancelled)
    _ => raise Failure("cancelled reusable request lost Completed")
  }
  let eof3 = FixedArray::make(1, b'\x00')
  assert_eq(client3.read(eof3, offset=0, max_len=1), 0)
  client3.close()

  drain_reusable_server(server)
  let stale_metrics = server.metrics_snapshot()
  let metrics = server.metrics_snapshot()
  assert_false(stale_metrics.is_live())
  assert_eq(metrics.counter_value(NetworkAccepts), 3UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 3UL)
  assert_eq(metrics.counter_value(NetworkRejections), 1UL)
  assert_eq(metrics.counter_value(Admissions), 2UL)
  assert_eq(metrics.counter_value(Completions), 1UL)
  assert_eq(metrics.counter_value(Cancellations), 1UL)
  assert_eq(metrics.counter_value(PromptTokens), 4UL)
  assert_eq(metrics.counter_value(GeneratedTokens), 1UL)
  assert_eq(metrics.counter_value(Failures), 0UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(metrics.gauge_value(KvPagesFree), 8)
  assert_eq(metrics_histogram_total(metrics, BatchRows), 3UL)
  assert_eq(metrics_histogram_total(metrics, BatchTokens), 3UL)

  let stale_log = server.log_snapshot()
  let log = server.log_snapshot()
  assert_false(stale_log.is_live())
  assert_eq(log_event_count(log, LunaInstanceReady), 1)
  assert_eq(log_event_count(log, LunaNetworkAccepted), 3)
  assert_eq(log_event_count(log, LunaNetworkDisconnected), 3)
  assert_eq(log_event_count(log, LunaAdmissionAccepted), 2)
  assert_eq(log_event_count(log, LunaAdmissionRejected), 1)
  assert_eq(log_event_count(log, LunaNetworkRejected), 1)
  assert_eq(log_event_count(log, LunaRequestCompleted), 1)
  assert_eq(log_event_count(log, LunaRequestCancelled), 1)
  assert_eq(log_event_count(log, LunaDrainStarted), 1)
  assert_eq(log_event_count(log, LunaInstanceClosed), 1)
}

///|
async fn progress_failed_connection(
  server : @online_tcp.LunaOnlineTcpServer,
  client : @socket.Tcp,
  ordinal : Int,
  disconnect_after_usage : Bool,
) -> Int {
  let mut observations = 0
  let mut disconnected = false
  let mut attempts = 0
  while attempts < 1000000 {
    match server.progress_on_reactor() {
      LunaOnlineTcpServerObservationRecorded => {
        observations += 1
        let metrics = server.metrics_snapshot()
        let log = server.log_snapshot()
        let last = log.length() - 1
        match observations {
          1 => {
            assert_eq(metrics.counter_value(Admissions), ordinal.to_uint64())
            assert_eq(
              metrics.counter_value(WorkerFailures),
              (ordinal - 1).to_uint64(),
            )
            assert_eq(log_event_count(log, LunaAdmissionAccepted), ordinal)
            assert_true(log.event_at(last) == LunaAdmissionAccepted)
            assert_true(log.reason_at(last) == LunaLogNoReason)
          }
          2 => {
            assert_eq(
              metrics.counter_value(WorkerFailures),
              ordinal.to_uint64(),
            )
            assert_eq(metrics.counter_value(Failures), ordinal.to_uint64())
            assert_eq(log_event_count(log, LunaWorkerFailed), ordinal)
            assert_eq(log_event_count(log, LunaRequestFailed), ordinal)
            assert_true(log.event_at(last - 1) == LunaWorkerFailed)
            assert_true(log.reason_at(last - 1) == LunaLogWorkerExit)
            assert_true(log.event_at(last) == LunaRequestFailed)
            assert_true(log.reason_at(last) == LunaLogInternal)
          }
          3 => {
            assert_eq(
              metrics.counter_value(WorkerRestarts),
              ordinal.to_uint64(),
            )
            assert_eq(log_event_count(log, LunaWorkerRestarted), ordinal)
            assert_true(log.event_at(last) == LunaWorkerRestarted)
            assert_true(log.reason_at(last) == LunaLogWorkerExit)
          }
          4 => {
            assert_eq(
              metrics.counter_value(PromptTokens),
              (ordinal * 2).to_uint64(),
            )
            assert_eq(metrics.counter_value(GeneratedTokens), 0UL)
            assert_true(log.event_at(last) == LunaWorkerRestarted)
            if disconnect_after_usage {
              client.close()
              disconnected = true
            }
          }
          5 => {
            assert_eq(metrics.counter_value(Failures), ordinal.to_uint64())
            assert_eq(log_event_count(log, LunaRequestFailed), ordinal)
            assert_eq(
              metrics.counter_value(WorkerRestarts),
              ordinal.to_uint64(),
            )
            assert_true(log.event_at(last) == LunaWorkerRestarted)
            assert_true(log.reason_at(last) == LunaLogWorkerExit)
          }
          _ => raise Failure("failure connection emitted an extra observation")
        }
      }
      LunaOnlineTcpServerConnectionRetired => {
        assert_eq(observations, if disconnected { 4 } else { 5 })
        return if disconnected { 1 } else { 0 }
      }
      LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      LunaOnlineTcpServerListeningProgress
      | LunaOnlineTcpServerAcceptTimedOut =>
        raise Failure("failure connection returned to accept before retirement")
      LunaOnlineTcpServerDrainingProgress | LunaOnlineTcpServerClosedProgress =>
        raise Failure("failure connection drained before retirement")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("failure connection did not retire")
}

///|
async fn assert_failed_server_observability(failure_executable : Bytes) -> Unit {
  let tokenizer = session_tokenizer()
  let preparation = prepare_tcp_service(failure_executable, tokenizer)
  let server = @online_tcp.bind_luna_online_tcp_server(
    @socket.Addr::parse("127.0.0.1:0"),
    preparation,
    reusable_server_limits(tcp_request_frame(983UL).length()),
    log_capacity=32,
  )

  // Connection one freezes the full canonical failure chain. The finite
  // telemetry deltas prove Admission -> WorkerFailure -> WorkerRestart ->
  // FailureUsage, while the byte stream proves the trailing Failed event.
  let first = tcp_request_frame(983UL)
  let client1 = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client1, first, source_offset=0, length=first.length())
  assert_eq(progress_failed_connection(server, client1, 1, false), 0)
  assert_eq(server.state(), LunaOnlineTcpServerListening)
  match tcp_read_event(client1) {
    Accepted(_) => ()
    _ => raise Failure("failed request lost Accepted")
  }
  match tcp_read_event(client1) {
    Usage(value) => assert_eq(value.output_tokens(), 0)
    _ => raise Failure("failed request lost FailureUsage")
  }
  match tcp_read_event(client1) {
    Failed(value) => {
      assert_eq(value.code().as_string(), "worker_unavailable")
      assert_true(value.retryable())
    }
    _ => raise Failure("failed request lost canonical Failed")
  }
  client1.close()

  // The restarted worker and same listener accept the next Stream. Closing
  // its socket exactly after FailureUsage must release the failed claim and
  // return to Listening without consuming the healthy replacement lease.
  let second = tcp_request_frame(984UL)
  let client2 = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  tcp_write_fixed(client2, second, source_offset=0, length=second.length())
  assert_eq(progress_failed_connection(server, client2, 2, true), 1)
  assert_eq(server.state(), LunaOnlineTcpServerListening)

  drain_reusable_server(server)
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(NetworkAccepts), 2UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 2UL)
  assert_eq(metrics.counter_value(Admissions), 2UL)
  assert_eq(metrics.counter_value(WorkerFailures), 2UL)
  assert_eq(metrics.counter_value(WorkerRestarts), 2UL)
  assert_eq(metrics.counter_value(Failures), 2UL)
  assert_eq(metrics.counter_value(PromptTokens), 4UL)
  assert_eq(metrics.counter_value(GeneratedTokens), 0UL)
  let log = server.log_snapshot()
  assert_eq(log_event_count(log, LunaNetworkAccepted), 2)
  assert_eq(log_event_count(log, LunaNetworkDisconnected), 2)
  assert_eq(log_event_count(log, LunaAdmissionAccepted), 2)
  assert_eq(log_event_count(log, LunaWorkerFailed), 2)
  assert_eq(log_event_count(log, LunaWorkerRestarted), 2)
  assert_eq(log_event_count(log, LunaRequestFailed), 2)
}

///|
async fn run_luna_reusable_server_evidence(
  executable : Bytes,
  failure_executable : Bytes?,
) -> Unit {
  assert_healthy_server_sequence(executable)
  match failure_executable {
    Some(value) => assert_failed_server_observability(value)
    None => ()
  }
}
