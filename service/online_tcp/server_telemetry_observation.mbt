///|
fn luna_server_observation_requires_record(recorded : Bool) -> Bool {
  !recorded
}

///|
#valtype
priv struct LunaServerObservationSample {
  kind : Int
  latency_kind : Int
  latency_millis : Int
  input_tokens : Int
  output_tokens : Int
}

///|
fn luna_server_observation_kind_code(
  kind : @online_session.LunaOnlineFramedObservationKind,
) -> Int {
  match kind {
    LunaOnlineFramedAdmissionObservation => 1
    LunaOnlineFramedTokenObservation => 2
    LunaOnlineFramedUsageObservation => 3
    LunaOnlineFramedCompletionObservation => 4
    LunaOnlineFramedCancellationObservation => 5
    LunaOnlineFramedDeadlineObservation => 6
    LunaOnlineFramedRequestFailureObservation => 7
    LunaOnlineFramedWorkerFailureObservation => 8
    LunaOnlineFramedWorkerRestartObservation => 9
    LunaOnlineFramedWorkerRequestTerminalObservation => 10
  }
}

///|
fn luna_server_latency_kind_code(
  kind : @online_session.LunaOnlineFramedLatencyKind,
) -> Int {
  match kind {
    LunaOnlineFramedNoLatency => 0
    LunaOnlineFramedRequestLatency => 1
    LunaOnlineFramedFirstTokenLatency => 2
    LunaOnlineFramedInterTokenLatency => 3
  }
}

///|
fn LunaServerTelemetryBridge::record_latency_sample_code(
  self : LunaServerTelemetryBridge,
  kind : Int,
  latency_millis : Int,
) -> Bool {
  if latency_millis < 0 {
    return false
  }
  match kind {
    0 => ()
    1 =>
      try! self.metrics.record_histogram(RequestLatencyMillis, latency_millis)
    2 =>
      try! self.metrics.record_histogram(
        FirstTokenLatencyMillis,
        latency_millis,
      )
    3 =>
      try! self.metrics.record_histogram(
        InterTokenLatencyMillis,
        latency_millis,
      )
    _ => return false
  }
  true
}

///|
fn LunaServerTelemetryBridge::record_observation_sample(
  self : LunaServerTelemetryBridge,
  sample : LunaServerObservationSample,
) -> Int {
  if !self.record_latency_sample_code(
      sample.latency_kind,
      sample.latency_millis,
    ) {
    return -1
  }
  match sample.kind {
    1 => {
      try! self.metrics.record_counter(Admissions)
      self.record_log(LunaAdmissionAccepted, LunaLogNoReason)
      0
    }
    3 => {
      try! self.metrics.record_counter(PromptTokens, amount=sample.input_tokens)
      try! self.metrics.record_counter(
        GeneratedTokens,
        amount=sample.output_tokens,
      )
      0
    }
    2 => 0
    4 => {
      try! self.metrics.record_counter(Completions)
      self.record_log(LunaRequestCompleted, LunaLogNoReason)
      1
    }
    5 => {
      try! self.metrics.record_counter(Cancellations)
      self.record_log(LunaRequestCancelled, LunaLogCancellation)
      1
    }
    6 => {
      try! self.metrics.record_counter(Deadlines)
      self.record_log(LunaRequestDeadline, LunaLogDeadline)
      1
    }
    7 => {
      try! self.metrics.record_counter(Failures)
      self.record_log(LunaRequestFailed, LunaLogInternal)
      1
    }
    8 => {
      try! self.metrics.record_counter(WorkerFailures)
      try! self.metrics.record_counter(Failures)
      self.record_log(LunaWorkerFailed, LunaLogWorkerExit)
      self.record_log(LunaRequestFailed, LunaLogInternal)
      0
    }
    10 => 1
    9 => {
      try! self.metrics.record_counter(WorkerRestarts)
      self.record_log(LunaWorkerRestarted, LunaLogWorkerExit)
      0
    }
    _ => -1
  }
}

///|
/// Records one already-pinned observation exactly once. Returns -1 when a
/// fallible scalar read failed, 0 for a nonterminal pulse, and 1 for a
/// terminal pulse. The caller retains the observation ACK retry guard.
fn LunaServerTelemetryBridge::record_observation(
  self : LunaServerTelemetryBridge,
  observation : @online_session.LunaOnlineFramedObservation,
) -> Int {
  let kind = observation.kind() catch { _ => return -1 }
  let input = if kind == LunaOnlineFramedUsageObservation {
    observation.input_tokens() catch {
      _ => return -1
    }
  } else {
    0
  }
  let output = if kind == LunaOnlineFramedUsageObservation {
    observation.output_tokens() catch {
      _ => return -1
    }
  } else {
    0
  }
  let latency_kind = observation.latency_kind() catch { _ => return -1 }
  let latency = if latency_kind == LunaOnlineFramedNoLatency {
    0
  } else {
    observation.latency_millis() catch {
      _ => return -1
    }
  }
  self.record_observation_sample({
    kind: luna_server_observation_kind_code(kind),
    latency_kind: luna_server_latency_kind_code(latency_kind),
    latency_millis: latency,
    input_tokens: input,
    output_tokens: output,
  })
}

///|
fn LunaServerTelemetryBridge::metrics_snapshot(
  self : LunaServerTelemetryBridge,
) -> @metrics.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError {
  self.metrics.snapshot() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
}

///|
fn LunaServerTelemetryBridge::log_snapshot(
  self : LunaServerTelemetryBridge,
) -> @instance_log.LunaInstanceLogSnapshot raise LunaOnlineTcpError {
  self.log.snapshot() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
}
