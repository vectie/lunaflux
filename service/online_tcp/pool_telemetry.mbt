///|
fn LunaOnlineFramedConnectionPool::refresh_metrics(
  self : LunaOnlineFramedConnectionPool,
) -> Bool {
  if self.services.length() != 1 {
    return self.closed
  }
  let telemetry = self.service().luna_telemetry() catch { _ => return false }
  let plan = self.service().luna_plan_telemetry() catch { _ => return false }
  if !self.telemetry.refresh(telemetry, plan) {
    return false
  }
  if !(self.service().has_luna_graph_runtime_telemetry() catch {
      _ => return false
    }) {
    return true
  }
  let graph = self.service().luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  self.telemetry.refresh_graph_runtime(graph)
}

///|
pub fn LunaOnlineFramedConnectionPool::record_cold_start_latency_millis(
  self : LunaOnlineFramedConnectionPool,
  duration_millis : Int,
) -> Unit raise LunaOnlineTcpError {
  match self.telemetry.record_cold_start_latency_millis(duration_millis) {
    -1 => raise luna_online_tcp_error(LunaOnlineTcpTelemetry)
    0 => raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
    _ => ()
  }
}

///|
pub fn LunaOnlineFramedConnectionPool::metrics_snapshot(
  self : LunaOnlineFramedConnectionPool,
) -> @metrics.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError {
  if !self.closed && !self.refresh_metrics() {
    raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
  self.telemetry.metrics_snapshot()
}

///|
fn LunaOnlineOpenAIConnectionPool::refresh_metrics(
  self : LunaOnlineOpenAIConnectionPool,
) -> Bool {
  if self.services.length() != 1 {
    return self.closed
  }
  let telemetry = self.service().luna_telemetry() catch { _ => return false }
  let plan = self.service().luna_plan_telemetry() catch { _ => return false }
  if !self.telemetry.refresh(telemetry, plan) {
    return false
  }
  if !(self.service().has_luna_graph_runtime_telemetry() catch {
      _ => return false
    }) {
    return true
  }
  let graph = self.service().luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  self.telemetry.refresh_graph_runtime(graph)
}

///|
pub fn LunaOnlineOpenAIConnectionPool::record_cold_start_latency_millis(
  self : LunaOnlineOpenAIConnectionPool,
  duration_millis : Int,
) -> Unit raise LunaOnlineTcpError {
  match self.telemetry.record_cold_start_latency_millis(duration_millis) {
    -1 => raise luna_online_tcp_error(LunaOnlineTcpTelemetry)
    0 => raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
    _ => ()
  }
  self.cold_start_latency_millis = duration_millis
}

///|
pub fn LunaOnlineOpenAIConnectionPool::metrics_snapshot(
  self : LunaOnlineOpenAIConnectionPool,
) -> @metrics.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError {
  if !self.closed && !self.refresh_metrics() {
    raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
  self.telemetry.metrics_snapshot()
}
