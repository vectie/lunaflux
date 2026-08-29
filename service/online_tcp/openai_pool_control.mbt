///|
fn LunaOnlineOpenAIConnectionPool::embedded_control_readiness(
  self : LunaOnlineOpenAIConnectionPool,
) -> Bool {
  self.production_ready &&
  self.readiness() == LunaOnlineOpenAIConnectionPoolReady
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
      let snapshot = self.telemetry.metrics_snapshot() catch {
        _ => return false
      }
      slot.embedded_control.prepare_metrics(
        snapshot,
        self.cold_start_latency_millis,
        ready,
      )
    }
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
