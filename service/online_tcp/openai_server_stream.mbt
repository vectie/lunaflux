///|
fn LunaOnlineOpenAIServer::capture_observation(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.observations.length() == 0 {
    let observation = self.stream().take_observation() catch {
        _ => {
          self.begin_retirement(
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
fn LunaOnlineOpenAIServer::capture_rejection(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.rejections.length() == 0 {
    let credit = self.stream().take_rejection() catch {
        _ => {
          self.begin_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
    self.rejections.push(credit)
  }
  let rule = self.rejections[0].rule() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let reason : @instance_log.LunaInstanceLogReason = match rule {
    LunaOnlineFramedRejectedDeadline => LunaLogDeadline
    LunaOnlineFramedRejectedCapacity => LunaLogCapacity
    LunaOnlineFramedRejectedService => LunaLogInternal
    _ => LunaLogValidation
  }
  let kind : @http1.LunaHttp1ErrorResponseKind = match rule {
    LunaOnlineFramedRejectedCapacity => LunaHttp1CapacityError
    LunaOnlineFramedRejectedService | LunaOnlineFramedRejectedDeadline =>
      LunaHttp1ServiceError
    _ => LunaHttp1ValidationError
  }
  if self.sse_started {
    self.telemetry.record_network_rejection(reason)
    self.begin_retirement(reason, failure=LunaOnlineTcpCoordinator)
    return LunaOnlineTcpServerAdvanced
  }
  self.telemetry.record_admission_rejection(reason)
  self.begin_http_error(kind)
}

///|
fn LunaOnlineOpenAIServer::capture_semantic_event(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.semantic_events.length() != 0 || self.outbound_works.length() != 0 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let semantic = self.stream().take_semantic_event() catch {
      _ => {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpServerAdvanced
      }
    }
  let view = semantic.view() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let kind = view.kind() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let mode : @openai_compat.LunaOpenAICompatMode = if self.route_mode == 1 {
    LunaOpenAIResponses
  } else {
    LunaOpenAIChatCompletions
  }
  let work = self.outbound_workspace.begin(
    view,
    mode,
    response_sequence=self.request_sequence,
  ) catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.event_terminal = match kind {
    LunaCompletedEventKind | LunaFailedEventKind => true
    _ => false
  }
  self.semantic_events.push(semantic)
  self.outbound_works.push(work)
  self.connection_phase = LUNA_OPENAI_COMPAT_PROGRESS
  LunaOnlineTcpServerAdvanced
}

///|
fn LunaOnlineOpenAIServer::map_semantic_progress(
  self : LunaOnlineOpenAIServer,
  progress : @online_session.LunaOnlineFramedCoordinatorProgress,
) -> LunaOnlineTcpServerProgress {
  match progress {
    LunaOnlineFramedCoordinatorObservationReady => self.capture_observation()
    LunaOnlineFramedCoordinatorRejectionReady => self.capture_rejection()
    LunaOnlineFramedCoordinatorSemanticEventReady =>
      self.capture_semantic_event()
    LunaOnlineFramedCoordinatorDraining | LunaOnlineFramedCoordinatorClosed => {
      self.latch_service_drain()
      LunaOnlineTcpServerDrainingProgress
    }
    LunaOnlineFramedCoordinatorIdle
    | LunaOnlineFramedCoordinatorAwaitingInput =>
      LunaOnlineTcpServerBackpressured
    _ => LunaOnlineTcpServerAdvanced
  }
}

///|
fn LunaOnlineOpenAIServer::progress_stream(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  let progress = self.stream().progress() catch {
      _ => {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpServerAdvanced
      }
    }
  self.map_semantic_progress(progress)
}

///|
fn LunaOnlineOpenAIServer::progress_compat(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if !self.enforce_output_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.outbound_works.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let state = self.outbound_works[0].state() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      return LunaOnlineTcpServerAdvanced
    }
  }
  match state {
    LunaOpenAICompatReady => {
      let view = self.outbound_works[0].take_view() catch {
        _ => {
          self.begin_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.outbound_works.clear()
      let length = view.length() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          self.begin_retirement(
            LunaLogInternal,
            failure=LunaOnlineTcpCoordinator,
          )
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.outbound_views.push(view)
      if !self.sse_started {
        self.begin_success_head()
      } else {
        self.output_source = 2
        self.output_cursor = 0
        self.output_length = length
        self.connection_phase = LUNA_OPENAI_COMPAT_WRITE
        LunaOnlineTcpServerAdvanced
      }
    }
    LunaOpenAICompatFailed => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
      LunaOnlineTcpServerAdvanced
    }
    LunaOpenAICompatPending => {
      ignore(
        self.outbound_works[0].progress() catch {
          _ => {
            self.begin_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpCoordinator,
            )
            return LunaOnlineTcpServerAdvanced
          }
        },
      )
      LunaOnlineTcpServerAdvanced
    }
  }
}
