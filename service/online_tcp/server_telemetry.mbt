///|
fn LunaOnlineTcpServer::record_log(
  self : LunaOnlineTcpServer,
  event : @instance_log.LunaInstanceLogEvent,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  self.telemetry.record_log(event, reason)
}

///|
fn LunaOnlineTcpServer::record_network_accept(
  self : LunaOnlineTcpServer,
) -> Unit {
  self.telemetry.record_network_accept()
}

///|
fn LunaOnlineTcpServer::record_network_disconnect(
  self : LunaOnlineTcpServer,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  self.telemetry.record_network_disconnect(reason)
}

///|
fn LunaOnlineTcpServer::record_backpressure(self : LunaOnlineTcpServer) -> Unit {
  self.telemetry.record_backpressure()
}

///|
fn luna_online_tcp_pipeline_admission_observation(
  kind : @online_session.LunaOnlineFramedObservationKind,
) -> Bool {
  kind == LunaOnlineFramedAdmissionObservation
}

///|
fn luna_online_tcp_pipeline_terminal_observation(
  kind : @online_session.LunaOnlineFramedObservationKind,
) -> Bool {
  match kind {
    LunaOnlineFramedCompletionObservation
    | LunaOnlineFramedCancellationObservation
    | LunaOnlineFramedDeadlineObservation
    | LunaOnlineFramedRequestFailureObservation
    | LunaOnlineFramedWorkerRequestTerminalObservation => true
    _ => false
  }
}

///|
fn luna_online_tcp_pipeline_pending_after_record(
  pending : Int,
  admission : Bool,
  terminal : Bool,
) -> Int {
  if pending < 0 || (admission && terminal) {
    return -1
  }
  if admission {
    if pending == 0x7fffffff {
      return -1
    }
    return pending + 1
  }
  if terminal {
    if pending == 0 {
      return -1
    }
    return pending - 1
  }
  pending
}

///|
fn LunaOnlineTcpServer::record_observation(self : LunaOnlineTcpServer) -> Bool {
  if self.observations.length() != 1 {
    return false
  }
  let observation = self.observations[0]
  if luna_server_observation_requires_record(self.observation_recorded) {
    let kind = observation.kind() catch { _ => return false }
    let admission_candidate = luna_online_tcp_pipeline_admission_observation(
      kind,
    )
    let terminal_candidate = luna_online_tcp_pipeline_terminal_observation(kind)
    let pending_after = if self.pipeline_mode {
      let value = luna_online_tcp_pipeline_pending_after_record(
        self.pipeline_receipts_pending,
        admission_candidate,
        terminal_candidate,
      )
      if value < 0 {
        return false
      }
      value
    } else {
      0
    }
    let terminal = self.telemetry.record_observation(observation)
    if terminal < 0 {
      return false
    }
    if self.pipeline_mode {
      self.pipeline_receipts_pending = pending_after
      if admission_candidate {
        // Only authenticated admission evidence starts a pending request.
        // Parsed or rejected frames never create terminal-accounting debt.
        self.pipeline_receipt_observed = true
      }
    }
    if terminal == 1 {
      self.terminal_observed = true
    }
    self.observation_recorded = true
  }
  // The local recorded bit stays set across an invariant ACK retry, so the
  // shared bridge never counts or logs the same observation twice.
  observation.ack() catch {
    _ => return false
  }
  self.observations.clear()
  self.observation_recorded = false
  true
}

///|
/// Records the future operator owner's measured root-open to listener-ready
/// interval exactly once. This telemetry-only call does not confer readiness.
pub fn LunaOnlineTcpServer::record_cold_start_latency_millis(
  self : LunaOnlineTcpServer,
  duration_millis : Int,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  match self.telemetry.record_cold_start_latency_millis(duration_millis) {
    -1 => raise luna_online_tcp_error(LunaOnlineTcpTelemetry)
    0 => raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
    _ => ()
  }
}

///|
fn LunaOnlineTcpServer::refresh_metrics(self : LunaOnlineTcpServer) -> Bool {
  let telemetry = self.service().luna_telemetry() catch { _ => return false }
  let plan = self.service().luna_plan_telemetry() catch { _ => return false }
  let graph_ready = self.service().has_luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  if !self.refresh_metrics_from(telemetry, plan) {
    return false
  }
  if !graph_ready {
    return true
  }
  let graph = self.service().luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  self.telemetry.refresh_graph_runtime(graph)
}

///|
fn LunaOnlineTcpServer::refresh_metrics_from(
  self : LunaOnlineTcpServer,
  telemetry : @online_session.LunaOnlineFramedTelemetry,
  plan : @online_session.LunaOnlineFramedPlanTelemetry,
) -> Bool {
  luna_native_server_refresh_sample(
    self.telemetry,
    luna_server_scheduler_sample(telemetry, plan),
  )
}

///|
pub fn LunaOnlineTcpServer::metrics_snapshot(
  self : LunaOnlineTcpServer,
) -> @metrics.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  if self.phase != LUNA_ONLINE_TCP_SERVER_CLOSED && !self.refresh_metrics() {
    raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
  self.telemetry.metrics_snapshot()
}

///|
pub fn LunaOnlineTcpServer::graph_runtime_telemetry(
  self : LunaOnlineTcpServer,
) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.service().luna_graph_runtime_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
}

///|
pub fn LunaOnlineTcpServer::log_snapshot(
  self : LunaOnlineTcpServer,
) -> @instance_log.LunaInstanceLogSnapshot raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.telemetry.log_snapshot()
}
