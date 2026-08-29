///|
fn LunaOnlineTcpServer::capture_observation(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.observations.length() == 0 {
    let observation = self.stream().take_observation() catch {
        _ => {
          self.begin_connection_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
    self.observations.push(observation)
  }
  if self.record_observation() {
    LunaOnlineTcpServerObservationRecorded
  } else {
    LunaOnlineTcpServerBackpressured
  }
}

///|
fn LunaOnlineTcpServer::capture_rejection(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  let credit = self.stream().take_rejection() catch {
      _ => {
        self.begin_connection_retirement(
          LunaLogInternal,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
    }
  let rule = credit.rule() catch {
    _ => {
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  ignore(
    credit.sequence() catch {
      _ => {
        self.begin_connection_retirement(
          LunaLogInternal,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
    },
  )
  credit.ack() catch {
    _ => {
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  let reason : @instance_log.LunaInstanceLogReason = match rule {
    LunaOnlineFramedRejectedDeadline => LunaLogDeadline
    LunaOnlineFramedRejectedCapacity => LunaLogCapacity
    LunaOnlineFramedRejectedService => LunaLogInternal
    _ => LunaLogValidation
  }
  self.telemetry.record_network_rejection(reason)
  self.begin_connection_retirement(reason)
  LunaOnlineTcpServerAdvanced
}

///|
fn LunaOnlineTcpServer::map_stream_progress(
  self : LunaOnlineTcpServer,
  progress : @online_session.LunaOnlineFramedCoordinatorProgress,
) -> LunaOnlineTcpServerProgress {
  match progress {
    LunaOnlineFramedCoordinatorObservationReady => self.capture_observation()
    LunaOnlineFramedCoordinatorRejectionReady => self.capture_rejection()
    LunaOnlineFramedCoordinatorSemanticEventReady => {
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      LunaOnlineTcpServerAdvanced
    }
    LunaOnlineFramedCoordinatorDraining | LunaOnlineFramedCoordinatorClosed => {
      self.begin_connection_retirement(LunaLogServiceDraining)
      self.latch_service_drain()
      LunaOnlineTcpServerAdvanced
    }
    LunaOnlineFramedCoordinatorIdle
    | LunaOnlineFramedCoordinatorAwaitingInput =>
      LunaOnlineTcpServerBackpressured
    LunaOnlineFramedCoordinatorAdvanced
    | LunaOnlineFramedCoordinatorEventPreparing
    | LunaOnlineFramedCoordinatorEventReady
    | LunaOnlineFramedCoordinatorMaintenanceRequired =>
      LunaOnlineTcpServerAdvanced
  }
}

///|
fn luna_online_tcp_pipeline_boundary_ready(
  pipeline_mode : Bool,
  completed_evidence : Bool,
  tail_length : Int,
  tail_backpressured : Bool,
  coordinator_boundary_clear : Bool,
) -> Bool {
  pipeline_mode &&
  completed_evidence &&
  tail_length == 0 &&
  !tail_backpressured &&
  coordinator_boundary_clear
}

///|
async fn LunaOnlineTcpServer::progress_connected_on_reactor(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.tail_length > 0 &&
    !self.tail_backpressured &&
    !self.request_receipt_complete {
    return self.offer_tail()
  }
  let progress = self.stream().progress() catch {
      _ => {
        self.begin_connection_retirement(
          LunaLogInternal,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
    }
  let tail_was_backpressured = self.tail_backpressured
  match progress {
    LunaOnlineFramedCoordinatorIdle
    | LunaOnlineFramedCoordinatorAwaitingInput => {
      // A retained tail can be offered again only after the lower scanner has
      // explicitly returned to an input-receiving disposition. Advanced and
      // event-producing progress may have failed the current Work; preserving
      // backpressure lets the coordinator publish its typed rejection before
      // this server can re-enter that stale Work.
      self.tail_backpressured = false
      let completed_evidence = self.pipeline_receipt_observed &&
        self.pipeline_receipts_pending == 0
      let pipeline_clear_candidate = self.pipeline_mode &&
        self.tail_length == 0 &&
        !tail_was_backpressured
      let coordinator_input_clear = if pipeline_clear_candidate {
        self.stream().luna_framed_input_clear() catch {
          _ => {
            self.begin_connection_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpCoordinator,
            )
            return LunaOnlineTcpServerAdvanced
          }
        }
      } else {
        false
      }
      let coordinator_boundary_clear = if pipeline_clear_candidate &&
        coordinator_input_clear {
        self.stream().luna_framed_boundary_clear() catch {
          _ => {
            self.begin_connection_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpCoordinator,
            )
            return LunaOnlineTcpServerAdvanced
          }
        }
      } else {
        false
      }
      if luna_online_tcp_pipeline_boundary_ready(
          self.pipeline_mode,
          completed_evidence,
          self.tail_length,
          tail_was_backpressured,
          coordinator_boundary_clear,
        ) {
        // The completed marker is only consumed at a true connection boundary:
        // no transport tail and no coordinator receipt/request/pool authority
        // remains. Idle/AwaitingInput has already passed the coordinator's
        // observation, rejection, outbound-event, semantic-event, and
        // maintenance prechecks; the scalar reauthenticates those empty states.
        self.terminal_observed = false
        self.pipeline_receipt_observed = false
        self.pipeline_receipts_pending = 0
        self.request_receipt_complete = false
        return LunaOnlineTcpServerConnectedProgress
      }
      // No retained input may still coexist with queued, active, outbound, or
      // maintenance authority. Keep driving the lower coordinator without
      // blocking for bytes that the peer does not owe while work is live.
      if pipeline_clear_candidate &&
        coordinator_input_clear &&
        !coordinator_boundary_clear {
        return LunaOnlineTcpServerAdvanced
      }
      if self.terminal_observed && !self.pipeline_mode {
        self.begin_connection_retirement(LunaLogNoReason)
        return LunaOnlineTcpServerAdvanced
      }
      if self.tail_length > 0 {
        if self.request_receipt_complete {
          return LunaOnlineTcpServerAdvanced
        }
        return self.offer_tail()
      }
      if self.request_receipt_complete {
        return LunaOnlineTcpServerAdvanced
      }
      self.read_on_reactor()
    }
    LunaOnlineFramedCoordinatorEventReady => self.write_event_on_reactor()
    _ => self.map_stream_progress(progress)
  }
}

///|
fn LunaOnlineTcpServer::progress_retirement(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.observations.length() == 1 {
    return if self.record_observation() {
      LunaOnlineTcpServerObservationRecorded
    } else {
      LunaOnlineTcpServerBackpressured
    }
  }
  if self.streams.length() != 1 {
    self.latch_failure(LunaOnlineTcpLifecycle)
    self.drain_requested = true
    self.begin_service_drain_if_ready()
    return LunaOnlineTcpServerDrainingProgress
  }
  let readiness = self.service().readiness()
  // Recovery can retire the lower stream before the outer connection owner
  // observes Draining/Closed. Never retry disconnect or progress through that
  // stale alias: release the exact outer slot, then converge through the
  // listener-first service-drain path.
  if !self.stream().is_live() {
    self.reset_connection_scalars()
    if readiness == LunaOnlineFramedServiceReady && !self.drain_requested {
      self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
      return LunaOnlineTcpServerConnectionRetired
    }
    self.latch_service_drain()
    self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
    self.begin_service_drain_if_ready()
    return LunaOnlineTcpServerDrainingProgress
  }
  if self.disconnect_pending && self.disconnect_progress_pending {
    self.disconnect_progress_pending = false
    let progress = self.stream().progress() catch {
        _ => {
          self.latch_failure(LunaOnlineTcpCoordinator)
          return LunaOnlineTcpServerAdvanced
        }
      }
    return match progress {
      LunaOnlineFramedCoordinatorDraining | LunaOnlineFramedCoordinatorClosed => {
        self.latch_service_drain()
        LunaOnlineTcpServerDrainingProgress
      }
      _ => self.map_stream_progress(progress)
    }
  }
  if self.disconnect_pending {
    self.stream().disconnect() catch {
      _ => {
        self.latch_failure(LunaOnlineTcpCoordinator)
        self.disconnect_progress_pending = true
        return LunaOnlineTcpServerAdvanced
      }
    }
    self.disconnect_pending = false
    self.disconnect_progress_pending = false
    return LunaOnlineTcpServerAdvanced
  }
  if readiness == LunaOnlineFramedServiceReady {
    self.reset_connection_scalars()
    if self.drain_requested {
      self.begin_service_drain_if_ready()
      return LunaOnlineTcpServerDrainingProgress
    }
    self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
    return LunaOnlineTcpServerConnectionRetired
  }
  let progress = self.stream().progress() catch {
      _ => {
        self.latch_failure(LunaOnlineTcpCoordinator)
        return LunaOnlineTcpServerBackpressured
      }
    }
  self.map_stream_progress(progress)
}

///|
fn LunaOnlineTcpServer::progress_service_drain(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  self.begin_service_drain_if_ready()
  if !self.service_drain_started {
    return LunaOnlineTcpServerDrainingProgress
  }
  let readiness = self.service().progress() catch {
      _ => {
        self.latch_failure(LunaOnlineTcpCoordinator)
        return LunaOnlineTcpServerDrainingProgress
      }
    }
  if readiness == LunaOnlineFramedServiceClosed {
    if self.streams.length() != 0 {
      self.reset_connection_scalars()
    }
    self.telemetry.close_gauges()
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
pub async fn LunaOnlineTcpServer::progress_on_reactor(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  self.activity.enter_reactor()
  defer self.activity.leave_reactor()
  if self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED {
    return LunaOnlineTcpServerClosedProgress
  }
  ignore(self.refresh_metrics())
  // Listener authority is always retired in the first reactor transition of
  // drain, before an active Stream or the persistent Service is touched.
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
          let readiness = self.service().progress() catch {
              _ => {
                self.latch_failure(LunaOnlineTcpCoordinator)
                return LunaOnlineTcpServerAdvanced
              }
            }
          if readiness == LunaOnlineFramedServiceDraining ||
            readiness == LunaOnlineFramedServiceClosed {
            self.latch_service_drain()
            LunaOnlineTcpServerDrainingProgress
          } else {
            LunaOnlineTcpServerAdvanced
          }
        }
      }
    }
    1 => {
      self.latch_failure(LunaOnlineTcpLifecycle)
      self.drain_requested = true
      LunaOnlineTcpServerDrainingProgress
    }
    2 =>
      if self.drain_requested {
        self.begin_connection_retirement(LunaLogServiceDraining)
        LunaOnlineTcpServerDrainingProgress
      } else {
        self.progress_connected_on_reactor()
      }
    3 => self.progress_retirement()
    4 => self.progress_service_drain()
    5 => LunaOnlineTcpServerClosedProgress
    _ => abort("Luna online TCP server phase invariant")
  }
}
