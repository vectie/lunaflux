///|
fn LunaOnlineOpenAIServer::receipt_wait(
  self : LunaOnlineOpenAIServer,
) -> Int raise LunaOnlineTcpError {
  if self.receipts.length() != 1 {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  let remaining = self.receipts[0].remaining_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpTimeout)
  }
  luna_online_tcp_safe_wait(self.limits.input_idle_timeout_millis, remaining)
}

///|
fn luna_openai_http_requires_more_input(
  state : @http1.LunaHttp1WorkState,
) -> Bool {
  match state {
    LunaHttp1ReceivingHead | LunaHttp1ReceivingBody => true
    _ => false
  }
}

///|
fn luna_openai_http_resume_phase(state : @http1.LunaHttp1WorkState) -> Int {
  if luna_openai_http_requires_more_input(state) {
    LUNA_OPENAI_HTTP_RECEIVE
  } else {
    LUNA_OPENAI_HTTP_PROGRESS
  }
}

///|
fn LunaOnlineOpenAIServer::enforce_receipt_deadline(
  self : LunaOnlineOpenAIServer,
) -> Bool {
  if self.receipts.length() == 0 {
    return true
  }
  if self.receipts.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return false
  }
  let remaining = self.receipts[0].remaining_millis() catch {
    _ => {
      self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
      return false
    }
  }
  if remaining <= 0 {
    self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
    return false
  }
  true
}

///|
fn LunaOnlineOpenAIServer::begin_http_error(
  self : LunaOnlineOpenAIServer,
  kind : @http1.LunaHttp1ErrorResponseKind,
) -> LunaOnlineTcpServerProgress {
  self.clear_http()
  let reason : @instance_log.LunaInstanceLogReason = match kind {
    LunaHttp1AuthenticationError => LunaLogAuthentication
    LunaHttp1ValidationError => LunaLogValidation
    LunaHttp1CapacityError => LunaLogCapacity
    LunaHttp1ServiceError => LunaLogInternal
  }
  self.telemetry.record_network_rejection_only(reason)
  let work = self.response_workspace.begin_error(kind) catch {
    _ => {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.response_works.push(work)
  self.response_kind = match kind {
    LunaHttp1AuthenticationError => 2
    LunaHttp1ValidationError => 3
    LunaHttp1CapacityError => 4
    LunaHttp1ServiceError => 5
  }
  self.connection_phase = LUNA_OPENAI_RESPONSE_PROGRESS
  LunaOnlineTcpServerAdvanced
}

///|
fn LunaOnlineOpenAIServer::map_http_failure(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  let error = self.http_works[0].failure() catch {
    _ => return self.begin_http_error(LunaHttp1ValidationError)
  }
  match error {
    LunaHttp1Failed(rule=LunaHttp1Authorization, ..) =>
      self.begin_http_error(LunaHttp1AuthenticationError)
    _ => self.begin_http_error(LunaHttp1ValidationError)
  }
}

///|
fn LunaOnlineOpenAIServer::offer_http_tail(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.route_mode == 0 {
    if self.tail_length <= 0 ||
      self.tail_start < 0 ||
      self.tail_start > self.input.length() - self.tail_length {
      self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
      return LunaOnlineTcpServerAdvanced
    }
    if self.input[self.tail_start] == b'G' {
      self.clear_receipt()
      self.clear_http()
      if !self.embedded_control.begin() {
        self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
        return LunaOnlineTcpServerAdvanced
      }
      self.route_mode = 3
      return self.offer_embedded_control_tail()
    }
    self.route_mode = -1
  }
  if self.route_mode == 3 {
    return self.offer_embedded_control_tail()
  }
  if !self.enforce_receipt_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.tail_length <= 0 ||
    self.tail_start < 0 ||
    self.tail_start > self.input.length() - self.tail_length ||
    self.http_works.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let consumed = self.http_works[0].offer(
    self.input,
    source_offset=self.tail_start,
    length=self.tail_length,
  ) catch {
    _ => return self.begin_http_error(LunaHttp1ValidationError)
  }
  if consumed < 0 || consumed > self.tail_length {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  self.tail_start += consumed
  self.tail_length -= consumed
  if self.tail_length == 0 {
    self.tail_start = 0
  }
  self.connection_phase = LUNA_OPENAI_HTTP_PROGRESS
  LunaOnlineTcpServerAdvanced
}

///|
async fn LunaOnlineOpenAIServer::read_http_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.connections.length() != 1 ||
    self.tail_length != 0 ||
    (self.route_mode == 3 && self.embedded_control.requests.length() != 1) ||
    (self.route_mode != 3 && self.http_works.length() != 1) {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let wait = if self.route_mode == 3 {
    luna_online_http_control_poll_millis(self.limits.input_idle_timeout_millis)
  } else {
    self.receipt_wait() catch {
      _ => {
        self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
        return LunaOnlineTcpServerAdvanced
      }
    }
  }
  if wait <= 0 {
    self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
    return LunaOnlineTcpServerAdvanced
  }
  let received = @async.with_timeout_opt(wait, () => {
    let count = self.connections[0].read(
      self.input,
      offset=0,
      max_len=self.input.length(),
    )
    if @async.is_being_cancelled() {
      @async.pause()
      raise luna_online_tcp_error(LunaOnlineTcpCancelled)
    }
    count
  }) catch {
    error if @async.is_being_cancelled() => {
      self.begin_retirement(LunaLogCancellation, failure=LunaOnlineTcpCancelled)
      raise error
    }
    _ => {
      self.begin_retirement(LunaLogNetworkRead, failure=LunaOnlineTcpRead)
      return LunaOnlineTcpServerAdvanced
    }
  }
  match received {
    None => {
      self.begin_retirement(LunaLogDeadline, failure=LunaOnlineTcpTimeout)
      LunaOnlineTcpServerAdvanced
    }
    Some(count) => {
      if count < 0 || count > self.input.length() {
        self.begin_retirement(LunaLogNetworkRead, failure=LunaOnlineTcpRead)
        return LunaOnlineTcpServerAdvanced
      }
      if count == 0 {
        if self.route_mode == 3 {
          self.embedded_control.requests[0].finish_input() catch {
            _ => ()
          }
          if !self.prepare_embedded_control_response(4) {
            self.begin_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpLifecycle,
            )
          }
        } else {
          self.http_works[0].finish_input() catch {
            _ => return self.begin_http_error(LunaHttp1ValidationError)
          }
          self.connection_phase = LUNA_OPENAI_HTTP_PROGRESS
        }
        return LunaOnlineTcpServerAdvanced
      }
      self.tail_start = 0
      self.tail_length = count
      LunaOnlineTcpServerAdvanced
    }
  }
}

///|
fn LunaOnlineOpenAIServer::progress_http(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if !self.enforce_receipt_deadline() {
    return LunaOnlineTcpServerAdvanced
  }
  if self.http_works.length() != 1 {
    self.begin_retirement(LunaLogInternal, failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpServerAdvanced
  }
  let state = self.http_works[0].state() catch {
    _ => return self.begin_http_error(LunaHttp1ValidationError)
  }
  match state {
    LunaHttp1Ready => {
      if self.tail_length > 0 {
        ignore(
          self.http_works[0].offer(
            self.input,
            source_offset=self.tail_start,
            length=self.tail_length,
          ) catch {
            _ => return self.begin_http_error(LunaHttp1ValidationError)
          },
        )
        return self.begin_http_error(LunaHttp1ValidationError)
      }
      let view = self.http_works[0].take_view() catch {
        _ => return self.begin_http_error(LunaHttp1ValidationError)
      }
      self.http_works.clear()
      let route = view.route() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.begin_http_error(LunaHttp1ValidationError)
        }
      }
      self.route_mode = match route {
        LunaHttp1Responses => 1
        LunaHttp1ChatCompletions => 2
      }
      if self.request_sequence == 0xffffffffffffffffUL {
        view.release() catch {
          _ => ()
        }
        self.latch_service_drain()
        return LunaOnlineTcpServerDrainingProgress
      }
      self.request_sequence += 1UL
      let request_id = @inference.RequestId::new(self.request_sequence) catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.begin_http_error(LunaHttp1ValidationError)
        }
      }
      let inbound = self.inbound_workspace.begin(view, request_id) catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return self.begin_http_error(LunaHttp1ValidationError)
        }
      }
      self.inbound_works.push(inbound)
      self.connection_phase = LUNA_OPENAI_INBOUND_PROGRESS
      LunaOnlineTcpServerAdvanced
    }
    LunaHttp1Failed => self.map_http_failure()
    state if luna_openai_http_requires_more_input(state) => {
      self.connection_phase = luna_openai_http_resume_phase(state)
      LunaOnlineTcpServerAdvanced
    }
    _ => {
      ignore(
        self.http_works[0].progress() catch {
          _ => return self.begin_http_error(LunaHttp1ValidationError)
        },
      )
      LunaOnlineTcpServerAdvanced
    }
  }
}
