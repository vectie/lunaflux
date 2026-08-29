///|
const PHASE23_SOAK_DURATION_MILLIS : UInt64 = 86400000UL

///|
const PHASE23_SOAK_WAVE_PERIOD_MILLIS : UInt64 = 2000UL

///|
const PHASE23_SOAK_VERSION : String = "luna-phase23-soak-v3"

///|
const PHASE23_SOAK_POLICY_DIGEST : String = "ab268f305c71658b53b9f8347eb77a79d1fd4c414fc01faa0492a48e3886751a"

///|
struct Phase23SoakLedger {
  mut cycles : Int
  mut admitted : UInt64
  mut completed : UInt64
  mut cancelled : UInt64
  mut rejected : UInt64
  mut accepts : UInt64
  mut disconnects : UInt64
  mut admitted_id_sum : UInt64
  mut terminal_id_sum : UInt64
  mut two_live_waves : UInt64
  mut batched_waves : UInt64
  mut backpressured_waves : UInt64
  mut scheduled_cancel_waves : UInt64
  mut attempted_cancel_waves : UInt64
  mut accepted_cancel_waves : UInt64
  mut cancelled_waves : UInt64
  initial_kv_free : Int
}

///|
fn Phase23SoakLedger::new(initial_kv_free : Int) -> Phase23SoakLedger {
  {
    cycles: 0,
    admitted: 0UL,
    completed: 0UL,
    cancelled: 0UL,
    rejected: 0UL,
    accepts: 1UL,
    disconnects: 0UL,
    admitted_id_sum: 0UL,
    terminal_id_sum: 0UL,
    two_live_waves: 0UL,
    batched_waves: 0UL,
    backpressured_waves: 0UL,
    scheduled_cancel_waves: 0UL,
    attempted_cancel_waves: 0UL,
    accepted_cancel_waves: 0UL,
    cancelled_waves: 0UL,
    initial_kv_free,
  }
}

///|
fn phase23_soak_record_balanced_wave(
  ledger : Phase23SoakLedger,
  saw_two_live : Bool,
  saw_two_row_plan : Bool,
  saw_backpressure : Bool,
  scheduled_cancel : Bool,
  cancel_attempted : Bool,
  cancel_accepted : Bool,
  cancelled : Bool,
) -> Unit {
  if saw_two_live {
    ledger.two_live_waves += 1UL
  }
  if saw_two_row_plan {
    ledger.batched_waves += 1UL
  }
  if saw_backpressure {
    ledger.backpressured_waves += 1UL
  }
  if scheduled_cancel {
    ledger.scheduled_cancel_waves += 1UL
  }
  if cancel_attempted {
    ledger.attempted_cancel_waves += 1UL
  }
  if cancel_accepted {
    ledger.accepted_cancel_waves += 1UL
  }
  if cancelled {
    ledger.cancelled_waves += 1UL
  }
}

///|
async fn phase23_soak_progress_wave(
  server : @online_tcp.LunaOnlineTcpServer,
  cancel_wave : Bool,
  before_admissions : UInt64,
  before_completions : UInt64,
  before_cancellations : UInt64,
  before_batch_rows : UInt64,
  ledger : Phase23SoakLedger,
  cycle : Int,
  trace? : Phase23SoakTrace,
) -> Bool {
  let mut cancel_attempted = false
  let mut cancel_accepted = false
  let mut saw_two_live = false
  let mut saw_two_row_plan = false
  let mut saw_backpressure = false
  let mut saw_missing_boundary_metrics = false
  let mut attempts = 0
  while attempts < 1000000 {
    let mut progress_complete = false
    defer (if !progress_complete {
      try! phase23_soak_trace_progress_predicate(
        trace, server, ledger, cycle, attempts, saw_two_live, saw_two_row_plan, saw_backpressure,
        cancel_attempted, cancel_accepted, "server-progress-error",
      )
    })
    let progress = server.progress_on_reactor()
    progress_complete = true
    if progress == LunaOnlineTcpServerBackpressured {
      saw_backpressure = true
    }
    match progress {
      LunaOnlineTcpServerObservationRecorded
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      LunaOnlineTcpServerConnectionRetired
      | LunaOnlineTcpServerListeningProgress
      | LunaOnlineTcpServerAcceptTimedOut
      | LunaOnlineTcpServerDrainingProgress
      | LunaOnlineTcpServerClosedProgress => {
        phase23_soak_trace_progress_predicate(
          trace,
          server,
          ledger,
          cycle,
          attempts,
          saw_two_live,
          saw_two_row_plan,
          saw_backpressure,
          cancel_attempted,
          cancel_accepted,
          if saw_missing_boundary_metrics {
            "server-progress-left-connected-after-missing-boundary"
          } else {
            "server-progress-left-connected"
          },
        )
        raise Failure("phase23 soak pipeline left its connected wave")
      }
    }
    let metrics = server.metrics_snapshot()
    let admissions = metrics.counter_value(Admissions)
    let live = metrics.gauge_value(ActiveRequests) +
      metrics.gauge_value(QueueDepth)
    if admissions == before_admissions + 2UL && live == 2 {
      saw_two_live = true
    }
    if metrics.histogram_bucket_value(BatchRows, 1) > before_batch_rows {
      saw_two_row_plan = true
    }
    if cancel_wave && saw_two_live && !cancel_attempted {
      cancel_attempted = true
      cancel_accepted = try {
        server.request_cancel()
        true
      } catch {
        LunaOnlineTcpFailed(LunaOnlineTcpCoordinator) => false
        error => {
          phase23_soak_trace_progress_predicate(
            trace, server, ledger, cycle, attempts, saw_two_live, saw_two_row_plan,
            saw_backpressure, cancel_attempted, cancel_accepted, "cancel-unexpected-error",
          )
          raise error
        }
      }
    }
    let terminal_cancelled = metrics.counter_value(Completions) ==
      before_completions + 1UL &&
      metrics.counter_value(Cancellations) == before_cancellations + 1UL
    let terminal_natural = metrics.counter_value(Completions) ==
      before_completions + 2UL &&
      metrics.counter_value(Cancellations) == before_cancellations
    if progress == LunaOnlineTcpServerConnectedProgress &&
      admissions == before_admissions + 2UL &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 &&
      !terminal_cancelled &&
      !terminal_natural {
      saw_missing_boundary_metrics = true
    }
    if progress == LunaOnlineTcpServerConnectedProgress &&
      admissions == before_admissions + 2UL &&
      metrics.gauge_value(ActiveRequests) == 0 &&
      metrics.gauge_value(QueueDepth) == 0 &&
      (terminal_cancelled || terminal_natural) {
      phase23_soak_record_balanced_wave(
        ledger, saw_two_live, saw_two_row_plan, saw_backpressure, cancel_wave, cancel_attempted,
        cancel_accepted, terminal_cancelled,
      )
      return terminal_cancelled
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  phase23_soak_trace_progress_predicate(
    trace, server, ledger, cycle, attempts, saw_two_live, saw_two_row_plan, saw_backpressure,
    cancel_attempted, cancel_accepted, "server-progress-exhausted",
  )
  raise Failure("phase23 soak wave did not reach its balanced boundary")
}

///|
fn phase23_soak_assert_quiescent(
  server : @online_tcp.LunaOnlineTcpServer,
  ledger : Phase23SoakLedger,
) -> Unit raise {
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(Admissions), ledger.admitted)
  assert_eq(metrics.counter_value(Completions), ledger.completed)
  assert_eq(metrics.counter_value(Cancellations), ledger.cancelled)
  assert_eq(metrics.counter_value(NetworkRejections), ledger.rejected)
  assert_eq(metrics.counter_value(NetworkAccepts), ledger.accepts)
  assert_eq(metrics.counter_value(NetworkDisconnects), ledger.disconnects)
  assert_eq(metrics.counter_value(PromptTokens), ledger.admitted * 2UL)
  assert_eq(metrics.counter_value(GeneratedTokens), ledger.completed)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  assert_eq(metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(metrics.gauge_value(KvPagesFree), ledger.initial_kv_free)
  assert_eq(ledger.admitted_id_sum, ledger.terminal_id_sum)
}

///|
async fn phase23_soak_wave(
  server : @online_tcp.LunaOnlineTcpServer,
  connection : @socket.Tcp,
  ledger : Phase23SoakLedger,
  cancel_wave : Bool,
  trace? : Phase23SoakTrace,
) -> Unit {
  let cycle = ledger.cycles
  let metrics = server.metrics_snapshot()
  let before_admissions = metrics.counter_value(Admissions)
  let before_completions = metrics.counter_value(Completions)
  let before_cancellations = metrics.counter_value(Cancellations)
  let before_batch_rows = metrics.histogram_bucket_value(BatchRows, 1)
  let first_id = (ledger.cycles * 2 + 1).to_uint64()
  let second_id = first_id + 1UL
  let combined = pipeline_two_frames(
    tcp_request_frame(first_id),
    tcp_request_frame(second_id),
  )
  phase23_soak_trace_breadcrumb(
    trace, server, ledger, cycle, "client-write-before",
  )
  phase23_soak_write_fragmented(connection, combined, ledger.cycles)
  ledger.admitted += 2UL
  ledger.admitted_id_sum += first_id + second_id
  phase23_soak_trace_breadcrumb(
    trace, server, ledger, cycle, "server-progress-before",
  )
  let cancelled = phase23_soak_progress_wave(
    server,
    cancel_wave,
    before_admissions,
    before_completions,
    before_cancellations,
    before_batch_rows,
    ledger,
    cycle,
    trace?,
  )
  if cancelled {
    ledger.cancelled += 1UL
    ledger.completed += 1UL
  } else {
    ledger.completed += 2UL
  }
  phase23_soak_trace_breadcrumb(
    trace, server, ledger, cycle, "client-read-before",
  )
  phase23_soak_read_wave(connection, cancelled, first_id, second_id, ledger)
  phase23_soak_trace_breadcrumb(
    trace, server, ledger, cycle, "quiescence-before",
  )
  ledger.cycles += 1
  phase23_soak_assert_quiescent(server, ledger)
  phase23_soak_trace_breadcrumb(trace, server, ledger, cycle, "wave-complete")
}

///|
async fn phase23_soak_reconnect(
  server : @online_tcp.LunaOnlineTcpServer,
  connection : @socket.Tcp,
  ledger : Phase23SoakLedger,
) -> @socket.Tcp {
  connection.close()
  assert_eq(progress_server_to_retired(server), 0)
  ledger.disconnects += 1UL
  let replacement = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  ledger.accepts += 1UL
  phase23_soak_assert_quiescent(server, ledger)
  replacement
}

///|
async fn phase23_soak_invalid_and_reconnect(
  server : @online_tcp.LunaOnlineTcpServer,
  connection : @socket.Tcp,
  ledger : Phase23SoakLedger,
  source : FixedArray[Byte],
  counts_as_rejection : Bool,
) -> @socket.Tcp {
  let invalid = phase23_soak_reconnect(server, connection, ledger)
  phase23_soak_write_fragmented(invalid, source, ledger.cycles)
  assert_eq(progress_server_to_retired(server), 0)
  invalid.close()
  if counts_as_rejection {
    ledger.rejected += 1UL
  }
  ledger.disconnects += 1UL
  let replacement = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  ledger.accepts += 1UL
  phase23_soak_assert_quiescent(server, ledger)
  replacement
}

///|
async fn phase23_soak_finish(
  server : @online_tcp.LunaOnlineTcpServer,
  connection : @socket.Tcp,
  ledger : Phase23SoakLedger,
  start_millis : UInt64,
  end_millis : UInt64,
  outcome : String,
) -> Unit {
  drain_reusable_server(server)
  let eof = FixedArray::make(1, b'\x00')
  assert_eq(connection.read(eof, offset=0, max_len=1), 0)
  connection.close()
  ledger.disconnects += 1UL
  assert_eq(server.state(), LunaOnlineTcpServerClosed)
  phase23_soak_assert_quiescent(server, ledger)
  assert_eq(ledger.accepts, ledger.disconnects)
  let cycles = ledger.cycles.to_uint64()
  assert_true(ledger.two_live_waves > 0UL && ledger.two_live_waves <= cycles)
  assert_true(ledger.batched_waves > 0UL && ledger.batched_waves <= cycles)
  assert_true(
    ledger.backpressured_waves > 0UL && ledger.backpressured_waves <= cycles,
  )
  assert_true(
    ledger.scheduled_cancel_waves > 0UL &&
    ledger.scheduled_cancel_waves <= cycles,
  )
  assert_true(
    ledger.attempted_cancel_waves > 0UL &&
    ledger.attempted_cancel_waves <= ledger.scheduled_cancel_waves,
  )
  assert_true(
    ledger.accepted_cancel_waves > 0UL &&
    ledger.accepted_cancel_waves <= ledger.attempted_cancel_waves,
  )
  assert_true(ledger.cancelled_waves > 0UL && ledger.cancelled_waves <= cycles)
  assert_true(ledger.cancelled_waves <= ledger.accepted_cancel_waves)
  assert_eq(ledger.cancelled_waves, ledger.cancelled)
  println(
    "LunaFlux phase23-soak version=\{PHASE23_SOAK_VERSION} digest=\{PHASE23_SOAK_POLICY_DIGEST} start_millis=\{start_millis} end_millis=\{end_millis} cycles=\{ledger.cycles} requests=\{ledger.admitted} completed=\{ledger.completed} cancelled=\{ledger.cancelled} rejected=\{ledger.rejected} accepts=\{ledger.accepts} disconnects=\{ledger.disconnects} two_live_waves=\{ledger.two_live_waves} batched_waves=\{ledger.batched_waves} backpressured_waves=\{ledger.backpressured_waves} scheduled_cancel_waves=\{ledger.scheduled_cancel_waves} attempted_cancel_waves=\{ledger.attempted_cancel_waves} accepted_cancel_waves=\{ledger.accepted_cancel_waves} cancelled_waves=\{ledger.cancelled_waves} outcome=\{outcome} resources=queue0,active0,kv_used0,kv_free\{ledger.initial_kv_free},pending0,child_closed",
  )
}

///|
async fn phase23_soak_prepare(
  executable : Bytes,
  fixture_root : String,
) -> (@online_tcp.LunaOnlineTcpServer, @socket.Tcp, Phase23SoakLedger) {
  let server = @online_tcp.bind_luna_online_tcp_pipeline_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_pipeline_service(executable, fixture_root~),
    reusable_server_limits(4096),
    log_capacity=128,
  )
  let initial = server.metrics_snapshot()
  let initial_kv_free = initial.gauge_value(KvPagesFree)
  assert_eq(initial.gauge_value(QueueDepth), 0)
  assert_eq(initial.gauge_value(ActiveRequests), 0)
  assert_eq(initial.gauge_value(KvPagesUsed), 0)
  let connection = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  (server, connection, Phase23SoakLedger::new(initial_kv_free))
}

///|
async fn run_phase23_soak_smoke(
  executable : Bytes,
  fixture_root : String,
) -> Unit {
  let (server, initial_connection, ledger) = phase23_soak_prepare(
    executable, fixture_root,
  )
  let mut connection = initial_connection
  phase23_soak_wave(server, connection, ledger, false)
  phase23_soak_wave(server, connection, ledger, true)
  connection = phase23_soak_invalid_and_reconnect(
    server,
    connection,
    ledger,
    tcp_request_frame_for_identity(1000000000UL, coordinator_foreign_identity()),
    true,
  )
  let malformed = tcp_request_frame(1000000001UL)
  malformed[0] = b'\x00'
  connection = phase23_soak_invalid_and_reconnect(
    server, connection, ledger, malformed, true,
  )
  phase23_soak_wave(server, connection, ledger, false)
  connection = phase23_soak_reconnect(server, connection, ledger)
  phase23_soak_finish(server, connection, ledger, 0UL, 0UL, "smoke-pass")
}

///|
async fn run_phase23_soak_24h(
  executable : Bytes,
  fixture_root : String,
) -> Unit {
  let clock = @monotonic_clock.MonotonicClock::system()
  let start_millis = clock.now_millis()
  let (server, initial_connection, ledger) = phase23_soak_prepare(
    executable, fixture_root,
  )
  let mut connection = initial_connection
  let mut observed_millis = start_millis
  while observed_millis - start_millis < PHASE23_SOAK_DURATION_MILLIS {
    let cycle = ledger.cycles
    phase23_soak_wave(server, connection, ledger, (cycle + 1) % 17 == 0)
    if ledger.cycles % 211 == 0 {
      connection = phase23_soak_invalid_and_reconnect(
        server,
        connection,
        ledger,
        tcp_request_frame_for_identity(
          1000000000UL + ledger.cycles.to_uint64(),
          coordinator_foreign_identity(),
        ),
        true,
      )
    }
    if ledger.cycles % 337 == 0 {
      let malformed = tcp_request_frame(
        2000000000UL + ledger.cycles.to_uint64(),
      )
      malformed[0] = b'\x00'
      connection = phase23_soak_invalid_and_reconnect(
        server, connection, ledger, malformed, true,
      )
    }
    if ledger.cycles % 257 == 0 {
      connection = phase23_soak_reconnect(server, connection, ledger)
    }
    observed_millis = clock.now_millis()
    if observed_millis < start_millis {
      raise Failure("phase23 soak monotonic clock moved backward")
    }
    let target_elapsed = ledger.cycles.to_uint64() *
      PHASE23_SOAK_WAVE_PERIOD_MILLIS
    let elapsed = observed_millis - start_millis
    if elapsed < target_elapsed {
      @async.sleep((target_elapsed - elapsed).to_int())
      observed_millis = clock.now_millis()
      if observed_millis < start_millis {
        raise Failure("phase23 soak monotonic clock moved backward")
      }
    }
  }
  let end_millis = clock.now_millis()
  if end_millis < start_millis ||
    end_millis - start_millis < PHASE23_SOAK_DURATION_MILLIS {
    raise Failure("phase23 soak ended before its hard 24-hour threshold")
  }
  phase23_soak_finish(
    server, connection, ledger, start_millis, end_millis, "24h-pass",
  )
}
