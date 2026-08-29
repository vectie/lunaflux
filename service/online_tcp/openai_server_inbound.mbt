///|
fn LunaOnlineOpenAIServer::begin_success_head(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  let work = self.response_workspace.begin_event_stream() catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.response_works.push(work)
  self.response_kind = 1
  self.connection_phase = LUNA_OPENAI_RESPONSE_PROGRESS
  LunaOnlineTcpServerAdvanced
}

///|
fn LunaOnlineOpenAIServer::progress_inbound(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if !self.enforce_receipt_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.inbound_works.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let state = self.inbound_works[0].state() catch {
    _ => return self.begin_http_error(LunaHttp1ValidationError)
  }
  match state {
    LunaOpenAIInboundReady => {
      if self.streams.length() != 0 {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
      let view = self.inbound_works[0].take_view() catch {
        _ => return self.begin_http_error(LunaHttp1ValidationError)
      }
      self.inbound_works.clear()
      let opened = self.service().open_semantic_stream()
      if opened.kind() != LunaOnlineFramedOpened {
        view.release() catch {
          _ => ()
        }
        return self.begin_http_error(LunaHttp1ServiceError)
      }
      let stream = self.service().take_open_stream(opened) catch {
          _ => {
            view.release() catch {
              _ => ()
            }
            return self.begin_http_error(LunaHttp1ServiceError)
          }
        }
      self.streams.push(stream)
      self.inbound_views.push(view)
      self.connection_phase = LUNA_OPENAI_HANDOFF_FEED
      LunaOnlineTcpServerAdvanced
    }
    LunaOpenAIInboundFailed => self.begin_http_error(LunaHttp1ValidationError)
    LunaOpenAIInboundPending => {
      ignore(
        self.inbound_works[0].progress() catch {
          _ => return self.begin_http_error(LunaHttp1ValidationError)
        },
      )
      LunaOnlineTcpServerAdvanced
    }
  }
}

///|
fn luna_openai_handoff_deadline_owner(
  receipt_count : Int,
  stream_count : Int,
) -> Int {
  if receipt_count == 1 {
    1
  } else if receipt_count == 0 && stream_count == 1 {
    2
  } else {
    0
  }
}

///|
fn LunaOnlineOpenAIServer::enforce_handoff_deadline(
  self : LunaOnlineOpenAIServer,
) -> Bool {
  match
    luna_openai_handoff_deadline_owner(
      self.receipts.length(),
      self.streams.length(),
    ) {
    1 => self.enforce_receipt_deadline()
    2 => {
      let remaining = self.stream().transport_wait_remaining_millis() catch {
          _ => {
            self.begin_retirement(
              LunaLogDeadline,
              failure=LunaOnlineTcpCoordinator,
            )
            return false
          }
        }
      if remaining <= 0 {
        self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
        return false
      }
      true
    }
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      false
    }
  }
}

///|
fn luna_openai_ingress_consumption_valid(
  kind : @online_session.LunaOnlineFramedIngressKind,
  consumed : Int,
  offered : Int,
) -> Bool {
  if consumed < 0 || consumed > offered {
    return false
  }
  match kind {
    LunaOnlineFramedIngressAccepted => consumed > 0
    LunaOnlineFramedIngressBackpressured | LunaOnlineFramedIngressDraining =>
      consumed == 0
  }
}

///|
fn LunaOnlineOpenAIServer::offer_handoff(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if !self.enforce_handoff_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.inbound_views.length() != 1 || self.receipts.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let handoff = self.inbound_views[0].handoff() catch {
    _ => {
      self.begin_retirement(LunaLogValidation, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  let ingress = self
    .stream()
    .offer_luna_text_handoff_with_receipt(self.receipts[0], handoff) catch {
      _ => {
        self.begin_retirement(
          LunaLogValidation,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
    }
  let consumed = ingress.consumed_bytes()
  let kind = ingress.kind()
  if !luna_openai_ingress_consumption_valid(kind, consumed, 1) {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpCoordinator)
    return LunaOnlineTcpServerAdvanced
  }
  match kind {
    LunaOnlineFramedIngressAccepted => {
      self.receipts.clear()
      self.inbound_views[0].release_after_transfer() catch {
        _ => {
          self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
          return LunaOnlineTcpServerAdvanced
        }
      }
      self.inbound_views.clear()
      self.backpressure_episode_active = false
      self.connection_phase = LUNA_OPENAI_STREAM_PROGRESS
      LunaOnlineTcpServerAdvanced
    }
    LunaOnlineFramedIngressBackpressured => {
      if !self.backpressure_episode_active {
        self.backpressure_episode_active = true
        self.record_backpressure()
      }
      LunaOnlineTcpServerBackpressured
    }
    LunaOnlineFramedIngressDraining => {
      self.latch_service_drain()
      LunaOnlineTcpServerDrainingProgress
    }
  }
}
