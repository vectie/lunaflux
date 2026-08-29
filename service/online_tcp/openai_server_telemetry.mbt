///|
fn LunaOnlineOpenAIServer::record_log(
  self : LunaOnlineOpenAIServer,
  event : @instance_log.LunaInstanceLogEvent,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  self.telemetry.record_log(event, reason)
}

///|
fn LunaOnlineOpenAIServer::record_network_accept(
  self : LunaOnlineOpenAIServer,
) -> Unit {
  self.telemetry.record_network_accept()
}

///|
fn LunaOnlineOpenAIServer::record_network_disconnect(
  self : LunaOnlineOpenAIServer,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  self.telemetry.record_network_disconnect(reason)
}

///|
fn LunaOnlineOpenAIServer::record_backpressure(
  self : LunaOnlineOpenAIServer,
) -> Unit {
  self.telemetry.record_backpressure()
}

///|
fn LunaOnlineOpenAIServer::record_observation(
  self : LunaOnlineOpenAIServer,
) -> Bool {
  if self.observations.length() != 1 {
    return false
  }
  let observation = self.observations[0]
  if luna_server_observation_requires_record(self.observation_recorded) {
    let disposition = self.telemetry.record_observation(observation)
    if disposition < 0 {
      return false
    }
    self.observation_recorded = true
  }
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
pub fn LunaOnlineOpenAIServer::record_cold_start_latency_millis(
  self : LunaOnlineOpenAIServer,
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
  self.cold_start_latency_millis = duration_millis
}

///|
fn LunaOnlineOpenAIServer::refresh_metrics(
  self : LunaOnlineOpenAIServer,
) -> Bool {
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
fn LunaOnlineOpenAIServer::refresh_metrics_from(
  self : LunaOnlineOpenAIServer,
  telemetry : @online_session.LunaOnlineFramedTelemetry,
  plan : @online_session.LunaOnlineFramedPlanTelemetry,
) -> Bool {
  luna_openai_server_refresh_sample(
    self.telemetry,
    luna_server_scheduler_sample(telemetry, plan),
  )
}

///|
pub fn LunaOnlineOpenAIServer::metrics_snapshot(
  self : LunaOnlineOpenAIServer,
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
pub fn LunaOnlineOpenAIServer::graph_runtime_telemetry(
  self : LunaOnlineOpenAIServer,
) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.service().luna_graph_runtime_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
}

///|
pub fn LunaOnlineOpenAIServer::log_snapshot(
  self : LunaOnlineOpenAIServer,
) -> @instance_log.LunaInstanceLogSnapshot raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.telemetry.log_snapshot()
}
