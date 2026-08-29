///|
const PHASE23_SOAK_DIAGNOSTIC_CYCLES : Int = 2300

///|
priv struct Phase23SoakTrace {
  mode : String
}

///|
fn phase23_soak_trace_server_phase(
  state : @online_tcp.LunaOnlineTcpServerState,
) -> String {
  match state {
    LunaOnlineTcpServerListening => "listening"
    LunaOnlineTcpServerAccepting => "accepting"
    LunaOnlineTcpServerConnected => "connected"
    LunaOnlineTcpServerRetiring => "retiring"
    LunaOnlineTcpServerDraining => "draining"
    LunaOnlineTcpServerClosed => "closed"
  }
}

///|
fn phase23_soak_trace_breadcrumb(
  trace : Phase23SoakTrace?,
  server : @online_tcp.LunaOnlineTcpServer,
  ledger : Phase23SoakLedger,
  cycle : Int,
  stage : String,
) -> Unit raise {
  guard trace is Some(owner) else { return }
  let metrics = server.metrics_snapshot()
  let next = cycle + 1
  println(
    "LunaFlux phase23-soak-diagnostic mode=\{owner.mode} cycle=\{cycle} ledger_cycles=\{ledger.cycles} stage=\{stage} cancel=\{next % 17 == 0} foreign=\{next % 211 == 0} malformed=\{next % 337 == 0} reconnect=\{next % 257 == 0} server=\{phase23_soak_trace_server_phase(server.state())} admissions=\{metrics.counter_value(Admissions)} completions=\{metrics.counter_value(Completions)} cancellations=\{metrics.counter_value(Cancellations)} deadlines=\{metrics.counter_value(Deadlines)} failures=\{metrics.counter_value(Failures)} worker_failures=\{metrics.counter_value(WorkerFailures)} rejections=\{metrics.counter_value(NetworkRejections)} accepts=\{metrics.counter_value(NetworkAccepts)} disconnects=\{metrics.counter_value(NetworkDisconnects)} two_live_waves=\{ledger.two_live_waves} batched_waves=\{ledger.batched_waves} backpressured_waves=\{ledger.backpressured_waves} scheduled_cancel_waves=\{ledger.scheduled_cancel_waves} attempted_cancel_waves=\{ledger.attempted_cancel_waves} accepted_cancel_waves=\{ledger.accepted_cancel_waves} cancelled_waves=\{ledger.cancelled_waves} queue=\{metrics.gauge_value(QueueDepth)} active=\{metrics.gauge_value(ActiveRequests)} kv_used=\{metrics.gauge_value(KvPagesUsed)} kv_free=\{metrics.gauge_value(KvPagesFree)}",
  )
}

///|
fn phase23_soak_trace_progress_predicate(
  trace : Phase23SoakTrace?,
  server : @online_tcp.LunaOnlineTcpServer,
  ledger : Phase23SoakLedger,
  cycle : Int,
  attempts : Int,
  saw_two_live : Bool,
  saw_two_row_plan : Bool,
  saw_backpressure : Bool,
  cancel_attempted : Bool,
  cancel_accepted : Bool,
  stage : String,
) -> Unit raise {
  guard trace is Some(owner) else { return }
  let metrics = server.metrics_snapshot()
  println(
    "LunaFlux phase23-soak-diagnostic mode=\{owner.mode} cycle=\{cycle} ledger_cycles=\{ledger.cycles} stage=\{stage} attempts=\{attempts} saw_two_live=\{saw_two_live} saw_two_row_plan=\{saw_two_row_plan} saw_backpressure=\{saw_backpressure} cancel_attempted=\{cancel_attempted} cancel_accepted=\{cancel_accepted} server=\{phase23_soak_trace_server_phase(server.state())} admissions=\{metrics.counter_value(Admissions)} completions=\{metrics.counter_value(Completions)} cancellations=\{metrics.counter_value(Cancellations)} deadlines=\{metrics.counter_value(Deadlines)} failures=\{metrics.counter_value(Failures)} worker_failures=\{metrics.counter_value(WorkerFailures)} two_live_waves=\{ledger.two_live_waves} batched_waves=\{ledger.batched_waves} backpressured_waves=\{ledger.backpressured_waves} scheduled_cancel_waves=\{ledger.scheduled_cancel_waves} attempted_cancel_waves=\{ledger.attempted_cancel_waves} accepted_cancel_waves=\{ledger.accepted_cancel_waves} cancelled_waves=\{ledger.cancelled_waves} queue=\{metrics.gauge_value(QueueDepth)} active=\{metrics.gauge_value(ActiveRequests)} kv_used=\{metrics.gauge_value(KvPagesUsed)} kv_free=\{metrics.gauge_value(KvPagesFree)}",
  )
}

///|
async fn run_phase23_soak_cycle_replay(
  executable : Bytes,
  fixture_root : String,
  timer_per_cycle : Bool,
) -> Unit {
  let mode = if timer_per_cycle { "timer" } else { "fast" }
  let trace : Phase23SoakTrace = { mode, }
  let (server, initial_connection, ledger) = phase23_soak_prepare(
    executable, fixture_root,
  )
  let mut connection = initial_connection
  while ledger.cycles < PHASE23_SOAK_DIAGNOSTIC_CYCLES {
    let cycle = ledger.cycles
    phase23_soak_trace_breadcrumb(
      Some(trace),
      server,
      ledger,
      cycle,
      "cycle-before",
    )
    phase23_soak_wave(server, connection, ledger, (cycle + 1) % 17 == 0, trace~)
    if ledger.cycles % 211 == 0 {
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "foreign-reconnect-before",
      )
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
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "foreign-reconnect-complete",
      )
    }
    if ledger.cycles % 337 == 0 {
      let malformed = tcp_request_frame(
        2000000000UL + ledger.cycles.to_uint64(),
      )
      malformed[0] = b'\x00'
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "malformed-reconnect-before",
      )
      connection = phase23_soak_invalid_and_reconnect(
        server, connection, ledger, malformed, true,
      )
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "malformed-reconnect-complete",
      )
    }
    if ledger.cycles % 257 == 0 {
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "plain-reconnect-before",
      )
      connection = phase23_soak_reconnect(server, connection, ledger)
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "plain-reconnect-complete",
      )
    }
    if timer_per_cycle {
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "timer-before",
      )
      @async.sleep(1)
      phase23_soak_trace_breadcrumb(
        Some(trace),
        server,
        ledger,
        cycle,
        "timer-complete",
      )
    }
  }
  phase23_soak_trace_breadcrumb(
    Some(trace),
    server,
    ledger,
    ledger.cycles,
    "finish-before",
  )
  phase23_soak_finish(
    server,
    connection,
    ledger,
    0UL,
    0UL,
    "diagnostic-\{mode}-pass",
  )
}
