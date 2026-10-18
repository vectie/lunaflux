///|
fn LunaOnlineOpenAIConnectionPool::embedded_control_readiness(
  self : LunaOnlineOpenAIConnectionPool,
) -> Bool {
  self.production_ready &&
  self.readiness() == LunaOnlineOpenAIConnectionPoolReady
}

///|
fn LunaOnlineOpenAIConnectionPool::refresh_pool_metrics(
  self : LunaOnlineOpenAIConnectionPool,
) -> Bool {
  let scheduler = self.service().luna_telemetry() catch { _ => return false }
  let plan = self.service().luna_plan_telemetry() catch { _ => return false }
  if !luna_openai_server_refresh_sample(
      self.telemetry,
      luna_server_scheduler_sample(scheduler, plan),
    ) {
    return false
  }
  let has_graph = self.service().has_luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  if !has_graph {
    return true
  }
  let graph = self.service().luna_graph_runtime_telemetry() catch {
      _ => return false
    }
  self.telemetry.refresh_graph_runtime(graph)
}

///|
fn LunaOnlineOpenAIConnectionPool::prepare_slot_embedded_control_response(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
  route : Int,
) -> Bool {
  let slot = self.slots[index]
  let healthy = self.health() == LunaOnlineTcpServerHealthy
  let ready = self.embedded_control_readiness()
  let prepared = match route {
    1 =>
      slot.embedded_control.prepare_static(
        if healthy {
          LunaHttp1ControlHealthy
        } else {
          LunaHttp1ControlUnhealthy
        },
      )
    2 =>
      slot.embedded_control.prepare_static(
        if ready {
          LunaHttp1ControlReadyResponse
        } else {
          LunaHttp1ControlNotReadyResponse
        },
      )
    3 => {
      // Read the already-retained worker/scheduler counters only when metrics
      // are requested. This does not query the GPU or add token-step work.
      if !self.refresh_pool_metrics() {
        return false
      }
      let snapshot = self.telemetry.metrics_snapshot() catch {
        _ => return false
      }
      slot.embedded_control.prepare_metrics(
        snapshot,
        self.cold_start_latency_millis,
        ready,
      )
    }
    5 => slot.embedded_control.prepare_models()
    _ => slot.embedded_control.prepare_static(LunaHttp1ControlBadRequest)
  }
  if prepared {
    slot.phase = LUNA_OPENAI_POOL_CONTROL_WRITE
  }
  prepared
}

///|
fn LunaOnlineOpenAIConnectionPool::offer_slot_embedded_control(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.tail_length <= 0 ||
    slot.tail_start < 0 ||
    slot.tail_start > slot.input.length() - slot.tail_length {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let route = slot.embedded_control.offer(
    slot.input,
    slot.tail_start,
    slot.tail_length,
  )
  slot.tail_start = 0
  slot.tail_length = 0
  if route < 0 ||
    (route > 0 && !self.prepare_slot_embedded_control_response(index, route)) {
    self.begin_slot_retirement(index)
  }
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
async fn LunaOnlineOpenAIConnectionPool::write_slot_embedded_control(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  if slot.connections.length() != 1 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let progress = slot.embedded_control.write_once(
    slot.connections[0],
    luna_online_http_control_poll_millis(
      self.pool_limits.io_poll_timeout_millis,
    ),
  )
  if progress < 0 {
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  if progress == 0 {
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  self.begin_slot_retirement(index)
  LunaOnlineOpenAIConnectionPoolAdvanced
}
