///|
fn LunaOnlineOpenAIServer::progress_connected(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  match self.connection_phase {
    2 => self.progress_http()
    3 => self.progress_inbound()
    4 => self.offer_handoff()
    5 => {
      let progress = self.progress_stream()
      if progress == LunaOnlineTcpServerBackpressured && self.terminal_observed {
        self.begin_retirement(LunaLogNoReason)
        LunaOnlineTcpServerAdvanced
      } else {
        progress
      }
    }
    6 => self.progress_compat()
    7 => self.progress_response()
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      LunaOnlineTcpServerAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIServer::progress_retirement(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.observations.length() == 1 {
    return if self.record_observation() {
      LunaOnlineTcpServerObservationRecorded
    } else {
      LunaOnlineTcpServerBackpressured
    }
  }
  if self.streams.length() != 1 {
    self.reset_connection()
    if self.service().readiness() == LunaOnlineFramedServiceReady &&
      !self.drain_requested {
      self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
      return LunaOnlineTcpServerConnectionRetired
    }
    self.latch_service_drain()
    self.begin_service_drain_if_ready()
    return LunaOnlineTcpServerDrainingProgress
  }
  let readiness = self.service().readiness()
  if !self.stream().is_live() {
    self.reset_connection()
    if readiness == LunaOnlineFramedServiceReady && !self.drain_requested {
      self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
      return LunaOnlineTcpServerConnectionRetired
    }
    self.latch_service_drain()
    self.begin_service_drain_if_ready()
    return LunaOnlineTcpServerDrainingProgress
  }
  if self.disconnect_pending && self.disconnect_progress_pending {
    self.disconnect_progress_pending = false
    let progress = self.stream().progress() catch {
        _ => {
          self.disconnect_progress_pending = true
          return LunaOnlineTcpServerBackpressured
        }
      }
    return self.map_semantic_progress(progress)
  }
  if self.disconnect_pending {
    self.stream().disconnect() catch {
      _ => {
        self.disconnect_progress_pending = true
        return LunaOnlineTcpServerAdvanced
      }
    }
    self.disconnect_pending = false
    return LunaOnlineTcpServerAdvanced
  }
  if readiness == LunaOnlineFramedServiceReady {
    self.reset_connection()
    if self.drain_requested {
      self.begin_service_drain_if_ready()
      return LunaOnlineTcpServerDrainingProgress
    }
    self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
    return LunaOnlineTcpServerConnectionRetired
  }
  let progress = self.stream().progress() catch {
      _ => return LunaOnlineTcpServerBackpressured
    }
  self.map_semantic_progress(progress)
}

///|
fn LunaOnlineOpenAIServer::progress_service_drain(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  self.begin_service_drain_if_ready()
  if !self.service_drain_started {
    return LunaOnlineTcpServerDrainingProgress
  }
  let readiness = self.service().progress() catch {
      _ => return LunaOnlineTcpServerDrainingProgress
    }
  if readiness == LunaOnlineFramedServiceClosed {
    if self.streams.length() != 0 {
      self.reset_connection()
    }
    self.telemetry.close_gauges()
    self.http_workspace.close_authentication()
    self.auth_policy.close()
    self.phase = LUNA_ONLINE_TCP_SERVER_CLOSED
    if !self.closed_logged {
      self.record_log(LunaInstanceClosed, LunaLogNoReason)
      self.closed_logged = true
    }
    LunaOnlineTcpServerClosedProgress
  } else {
    LunaOnlineTcpServerDrainingProgress
  }
}

///|
pub async fn LunaOnlineOpenAIServer::progress_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  self.activity.enter_reactor()
  defer self.activity.leave_reactor()
  if self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED {
    return LunaOnlineTcpServerClosedProgress
  }
  ignore(self.refresh_metrics())
  if self.drain_requested && self.close_listener_pending {
    self.close_listener_on_reactor()
    if !self.drain_logged {
      self.record_log(LunaDrainStarted, LunaLogServiceDraining)
      self.drain_logged = true
    }
    return LunaOnlineTcpServerDrainingProgress
  }
  match self.phase {
    0 => {
      if self.drain_requested {
        return self.progress_service_drain()
      }
      match self.service().readiness() {
        LunaOnlineFramedServiceReady => self.accept_on_reactor()
        LunaOnlineFramedServiceDraining | LunaOnlineFramedServiceClosed => {
          self.latch_service_drain()
          LunaOnlineTcpServerDrainingProgress
        }
        _ => {
          ignore(
            self.service().progress() catch {
              _ => {
                self.latch_service_drain()
                return LunaOnlineTcpServerDrainingProgress
              }
            },
          )
          LunaOnlineTcpServerAdvanced
        }
      }
    }
    1 => {
      self.latch_service_drain()
      LunaOnlineTcpServerDrainingProgress
    }
    2 => {
      if self.drain_requested {
        self.begin_retirement(LunaLogServiceDraining)
        return LunaOnlineTcpServerDrainingProgress
      }
      match self.connection_phase {
        1 =>
          if self.tail_length > 0 {
            self.offer_http_tail()
          } else {
            self.read_http_on_reactor()
          }
        8 | 9 => self.write_output_on_reactor()
        10 => self.write_embedded_control_on_reactor()
        _ => self.progress_connected()
      }
    }
    3 => self.progress_retirement()
    4 => self.progress_service_drain()
    5 => LunaOnlineTcpServerClosedProgress
    _ => abort("Luna OpenAI server phase invariant")
  }
}
