///|
/// The control listener shares a serialized runtime owner with inference. Its
/// socket turns therefore use a separate, strict poll cap rather than inheriting
/// an inference request's admitted transport deadline.
const LUNA_ONLINE_HTTP_CONTROL_MAX_POLL_MILLIS : Int = 1

///|
fn luna_online_http_control_poll_millis(configured : Int) -> Int {
  Int::min(configured, LUNA_ONLINE_HTTP_CONTROL_MAX_POLL_MILLIS)
}

///|
fn LunaOnlineHttpControlServer::prepare_response(
  self : LunaOnlineHttpControlServer,
  kind : @http1.LunaHttp1ControlResponseKind,
) -> Bool {
  if self.metrics_flights.length() != 0 || !self.metrics_output.is_idle() {
    return false
  }
  let index = match kind {
    LunaHttp1ControlHealthy => 0
    LunaHttp1ControlUnhealthy => 1
    LunaHttp1ControlReadyResponse => 2
    LunaHttp1ControlNotReadyResponse => 3
    LunaHttp1ControlBadRequest => 4
  }
  let output = self.responses[index]
  let length = output.length()
  if length <= 0 {
    return false
  }
  self.output = output
  self.output_cursor = 0
  self.output_length = length
  self.phase = LUNA_CONTROL_SERVER_WRITING
  true
}

///|
fn LunaOnlineHttpControlServer::prepare_metrics_response(
  self : LunaOnlineHttpControlServer,
) -> Bool {
  if self.metrics_flights.length() != 0 || !self.metrics_output.is_idle() {
    return false
  }
  let snapshot = self.source.metrics_snapshot() catch { _ => return false }
  let write = self.metrics_output.begin_write() catch { _ => return false }
  let length = luna_online_render_control_metrics_response(
    write,
    snapshot,
    self.cold_start_latency_millis,
    self.source_readiness(),
  ) catch {
    _ => {
      write.abort() catch {
        _ => ()
      }
      return false
    }
  }
  let flight = write.publish(length) catch {
    _ => {
      write.abort() catch {
        _ => ()
      }
      return false
    }
  }
  self.metrics_flights.push(flight)
  self.output = self.metrics_output.immutable
  self.output_cursor = 0
  self.output_length = length
  self.phase = LUNA_CONTROL_SERVER_WRITING
  true
}

///|
async fn LunaOnlineHttpControlServer::accept_on_reactor(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlProgress {
  if self.listeners.length() != 1 ||
    self.connections.length() != 0 ||
    self.requests.length() != 0 {
    self.fail()
    return LunaOnlineHttpControlFailedProgress
  }
  self.phase = LUNA_CONTROL_SERVER_ACCEPTING
  let accepted = @async.with_timeout_opt(
    luna_online_http_control_poll_millis(self.limits.accept_timeout_millis),
    () => {
      let (connection, _) = self.listeners[0].accept()
      let mut installed = false
      defer (if !installed { connection.close() })
      if @async.is_being_cancelled() {
        @async.pause()
        raise luna_online_tcp_error(LunaOnlineTcpCancelled)
      }
      if self.drain_requested || self.phase != LUNA_CONTROL_SERVER_ACCEPTING {
        return false
      }
      let request = self.workspace.begin() catch { _ => return false }
      self.connections.push(connection)
      self.requests.push(request)
      self.phase = LUNA_CONTROL_SERVER_RECEIVING
      installed = true
      true
    },
  ) catch {
    error if @async.is_being_cancelled() => {
      self.fail()
      raise error
    }
    _ => {
      self.fail()
      return LunaOnlineHttpControlFailedProgress
    }
  }
  match accepted {
    Some(true) => LunaOnlineHttpControlConnectedProgress
    Some(false) =>
      if self.drain_requested {
        self.phase = LUNA_CONTROL_SERVER_DRAINING
        LunaOnlineHttpControlDrainingProgress
      } else {
        self.fail()
        LunaOnlineHttpControlFailedProgress
      }
    None => {
      self.phase = LUNA_CONTROL_SERVER_LISTENING
      LunaOnlineHttpControlAcceptTimedOut
    }
  }
}

///|
async fn LunaOnlineHttpControlServer::read_on_reactor(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlProgress {
  if self.connections.length() != 1 || self.requests.length() != 1 {
    self.fail()
    return LunaOnlineHttpControlFailedProgress
  }
  let received = @async.with_timeout_opt(
    luna_online_http_control_poll_millis(self.limits.input_idle_timeout_millis),
    () => {
      self.connections[0].read(
        self.input,
        offset=0,
        max_len=self.input.length(),
      )
    },
  ) catch {
    error if @async.is_being_cancelled() => {
      self.fail()
      raise error
    }
    _ => {
      self.retire_connection()
      return LunaOnlineHttpControlConnectionRetired
    }
  }
  let count = match received {
    Some(value) => value
    None => {
      self.retire_connection()
      return LunaOnlineHttpControlConnectionRetired
    }
  }
  if count < 0 || count > self.input.length() {
    self.fail()
    return LunaOnlineHttpControlFailedProgress
  }
  if count == 0 {
    self.requests[0].finish_input() catch {
      _ => ()
    }
  } else {
    let consumed = self.requests[0].offer(
      self.input,
      source_offset=0,
      length=count,
    ) catch {
      _ => 0
    }
    if consumed != count {
      self.requests[0].abort() catch {
        _ => ()
      }
      self.requests.clear()
      if !self.prepare_response(LunaHttp1ControlBadRequest) {
        self.fail()
        return LunaOnlineHttpControlFailedProgress
      }
      return LunaOnlineHttpControlAdvanced
    }
  }
  let request_state = self.requests[0].state() catch {
    _ => LunaHttp1ControlFailed
  }
  match request_state {
    LunaHttp1ControlReceiving => LunaOnlineHttpControlAdvanced
    LunaHttp1ControlFailed => {
      self.requests[0].abort() catch {
        _ => ()
      }
      self.requests.clear()
      if !self.prepare_response(LunaHttp1ControlBadRequest) {
        self.fail()
        LunaOnlineHttpControlFailedProgress
      } else {
        LunaOnlineHttpControlAdvanced
      }
    }
    LunaHttp1ControlReady => {
      let view = self.requests[0].take_view() catch {
        _ => {
          self.fail()
          return LunaOnlineHttpControlFailedProgress
        }
      }
      self.requests.clear()
      let route = view.route() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          self.fail()
          return LunaOnlineHttpControlFailedProgress
        }
      }
      view.release() catch {
        _ => {
          self.fail()
          return LunaOnlineHttpControlFailedProgress
        }
      }
      if route == LunaHttp1Metrics {
        if !self.prepare_metrics_response() {
          self.fail()
          return LunaOnlineHttpControlFailedProgress
        }
        return LunaOnlineHttpControlAdvanced
      }
      let kind : @http1.LunaHttp1ControlResponseKind = match route {
        LunaHttp1Health =>
          if self.source_health() {
            LunaHttp1ControlHealthy
          } else {
            LunaHttp1ControlUnhealthy
          }
        LunaHttp1Readiness =>
          if self.source_readiness() {
            LunaHttp1ControlReadyResponse
          } else {
            LunaHttp1ControlNotReadyResponse
          }
        LunaHttp1Metrics => abort("control metrics route handled above")
      }
      if !self.prepare_response(kind) {
        self.fail()
        LunaOnlineHttpControlFailedProgress
      } else {
        LunaOnlineHttpControlAdvanced
      }
    }
  }
}

///|
async fn LunaOnlineHttpControlServer::write_on_reactor(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlProgress {
  if self.connections.length() != 1 ||
    self.output_cursor < 0 ||
    self.output_cursor >= self.output_length ||
    self.output_length > self.output.length() {
    self.fail()
    return LunaOnlineHttpControlFailedProgress
  }
  let remaining = self.output_length - self.output_cursor
  let written = @async.with_timeout_opt(
    luna_online_http_control_poll_millis(self.limits.write_timeout_millis),
    () => {
      self.connections[0].write_once(
        self.output,
        offset=self.output_cursor,
        len=remaining,
      )
    },
  ) catch {
    error if @async.is_being_cancelled() => {
      self.fail()
      raise error
    }
    _ => {
      self.retire_connection()
      return LunaOnlineHttpControlConnectionRetired
    }
  }
  let count = match written {
    Some(value) => value
    None => {
      self.retire_connection()
      return LunaOnlineHttpControlConnectionRetired
    }
  }
  if count <= 0 || count > remaining {
    self.fail()
    return LunaOnlineHttpControlFailedProgress
  }
  self.output_cursor += count
  if self.output_cursor == self.output_length {
    self.retire_connection()
    LunaOnlineHttpControlConnectionRetired
  } else {
    LunaOnlineHttpControlAdvanced
  }
}

///|
pub async fn LunaOnlineHttpControlServer::progress_on_reactor(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlProgress {
  if self.failed {
    if self.drain_requested {
      ignore(self.progress_drain())
    }
    return LunaOnlineHttpControlFailedProgress
  }
  if self.phase == LUNA_CONTROL_SERVER_CLOSED {
    return LunaOnlineHttpControlClosedProgress
  }
  if self.drain_requested {
    return self.progress_drain()
  }
  match self.phase {
    0 => self.accept_on_reactor()
    1 => {
      self.fail()
      LunaOnlineHttpControlFailedProgress
    }
    2 => self.read_on_reactor()
    3 => self.write_on_reactor()
    4 => self.progress_drain()
    5 => LunaOnlineHttpControlClosedProgress
    _ => LunaOnlineHttpControlFailedProgress
  }
}
