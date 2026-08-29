///|
fn LunaOnlineOpenAIServer::embedded_control_readiness(
  self : LunaOnlineOpenAIServer,
) -> Bool {
  self.production_ready &&
  luna_online_control_readiness_available(
    self.failure_rule,
    self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
    self.drain_requested,
    self.listeners.length(),
    self.phase == LUNA_ONLINE_TCP_SERVER_DRAINING,
    self.service().readiness() == LunaOnlineFramedServiceReady,
  )
}

///|
fn LunaOnlineOpenAIServer::prepare_embedded_control_response(
  self : LunaOnlineOpenAIServer,
  route : Int,
) -> Bool {
  let healthy = luna_online_control_health_available(
    self.failure_rule,
    self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
  )
  let ready = self.embedded_control_readiness()
  let prepared = match route {
    1 =>
      self.embedded_control.prepare_static(
        if healthy {
          LunaHttp1ControlHealthy
        } else {
          LunaHttp1ControlUnhealthy
        },
      )
    2 =>
      self.embedded_control.prepare_static(
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
      self.embedded_control.prepare_metrics(
        snapshot,
        self.cold_start_latency_millis,
        ready,
      )
    }
    _ => self.embedded_control.prepare_static(LunaHttp1ControlBadRequest)
  }
  if prepared {
    self.connection_phase = LUNA_OPENAI_CONTROL_WRITE
  }
  prepared
}

///|
fn LunaOnlineOpenAIServer::offer_embedded_control_tail(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.tail_length <= 0 ||
    self.tail_start < 0 ||
    self.tail_start > self.input.length() - self.tail_length {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let route = self.embedded_control.offer(
    self.input,
    self.tail_start,
    self.tail_length,
  )
  self.tail_start = 0
  self.tail_length = 0
  if route < 0 || (route > 0 && !self.prepare_embedded_control_response(route)) {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
  }
  LunaOnlineTcpServerAdvanced
}

///|
async fn LunaOnlineOpenAIServer::write_embedded_control_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.connections.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let progress = self.embedded_control.write_once(
    self.connections[0],
    luna_online_http_control_poll_millis(self.limits.write_timeout_millis),
  )
  if progress < 0 {
    self.begin_retirement(LunaLogNetworkWrite, failure=LunaOnlineTcpWrite)
    return LunaOnlineTcpServerAdvanced
  }
  if progress == 0 {
    return LunaOnlineTcpServerAdvanced
  }
  self.begin_retirement(LunaLogNoReason)
  LunaOnlineTcpServerAdvanced
}
