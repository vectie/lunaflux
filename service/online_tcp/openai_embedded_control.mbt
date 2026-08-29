///|
/// Startup-owned, per-connection control state for OpenAI listeners. Routing
/// remains outside this cell: only an exact GET-prefixed request may enter it.
priv struct LunaOpenAIEmbeddedControl {
  workspace : @http1.LunaHttp1ControlWorkspace
  requests : Array[@http1.LunaHttp1ControlRequest]
  responses : FixedArray[Bytes]
  metrics_output : LunaOnlineTcpOutputScratch
  metrics_flights : Array[LunaOnlineTcpOutputFlight]
  mut output : Bytes
  mut output_cursor : Int
  mut output_length : Int
}

///|
fn new_luna_openai_embedded_control() -> LunaOpenAIEmbeddedControl raise LunaOnlineTcpError {
  let limits = @http1.LunaHttp1ControlLimits::new(max_head_bytes=1024) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let workspace = @http1.LunaHttp1ControlWorkspace::new(limits) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let metrics_output = LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    workspace,
    requests: Array::new(capacity=1),
    responses: luna_online_control_responses(),
    metrics_output,
    metrics_flights: Array::new(capacity=1),
    output: b"",
    output_cursor: 0,
    output_length: 0,
  }
}

///|
fn LunaOpenAIEmbeddedControl::begin(self : LunaOpenAIEmbeddedControl) -> Bool {
  if self.requests.length() != 0 ||
    self.metrics_flights.length() != 0 ||
    !self.metrics_output.is_idle() {
    return false
  }
  let request = self.workspace.begin() catch { _ => return false }
  self.requests.push(request)
  true
}

///|
fn LunaOpenAIEmbeddedControl::clear(self : LunaOpenAIEmbeddedControl) -> Unit {
  if self.requests.length() == 1 {
    self.requests[0].abort() catch {
      _ => ()
    }
    self.requests.clear()
  }
  if self.metrics_flights.length() == 1 {
    self.metrics_flights[0].release() catch {
      _ => ()
    }
    self.metrics_flights.clear()
  }
  self.output = b""
  self.output_cursor = 0
  self.output_length = 0
}

///|
fn LunaOpenAIEmbeddedControl::prepare_static(
  self : LunaOpenAIEmbeddedControl,
  kind : @http1.LunaHttp1ControlResponseKind,
) -> Bool {
  let index = match kind {
    LunaHttp1ControlHealthy => 0
    LunaHttp1ControlUnhealthy => 1
    LunaHttp1ControlReadyResponse => 2
    LunaHttp1ControlNotReadyResponse => 3
    LunaHttp1ControlBadRequest => 4
  }
  self.output = self.responses[index]
  self.output_cursor = 0
  self.output_length = self.output.length()
  self.output_length > 0
}

///|
fn LunaOpenAIEmbeddedControl::prepare_metrics(
  self : LunaOpenAIEmbeddedControl,
  snapshot : @metrics.LunaInstanceMetricsSnapshot,
  cold_start_latency_millis : Int,
  readiness : Bool,
) -> Bool {
  let write = self.metrics_output.begin_write() catch { _ => return false }
  let length = luna_online_render_control_metrics_response(
    write, snapshot, cold_start_latency_millis, readiness,
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
  true
}

///|
fn LunaOpenAIEmbeddedControl::offer(
  self : LunaOpenAIEmbeddedControl,
  input : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Int {
  if self.requests.length() != 1 || length <= 0 {
    return -1
  }
  let consumed = self.requests[0].offer(input, source_offset=offset, length~) catch {
    _ => return -1
  }
  if consumed != length {
    return 4
  }
  let state = self.requests[0].state() catch { _ => LunaHttp1ControlFailed }
  match state {
    LunaHttp1ControlReceiving => 0
    LunaHttp1ControlFailed => 4
    LunaHttp1ControlReady => {
      let view = self.requests[0].take_view() catch { _ => return -1 }
      self.requests.clear()
      let route = view.route() catch {
        _ => {
          view.release() catch {
            _ => ()
          }
          return -1
        }
      }
      view.release() catch {
        _ => return -1
      }
      match route {
        LunaHttp1Health => 1
        LunaHttp1Readiness => 2
        LunaHttp1Metrics => 3
      }
    }
  }
}

///|
async fn LunaOpenAIEmbeddedControl::write_once(
  self : LunaOpenAIEmbeddedControl,
  connection : @socket.Tcp,
  timeout_millis : Int,
) -> Int {
  if self.output_cursor < 0 ||
    self.output_cursor >= self.output_length ||
    self.output_length > self.output.length() {
    return -1
  }
  let remaining = self.output_length - self.output_cursor
  let written = @async.with_timeout_opt(timeout_millis, () => {
    connection.write_once(self.output, offset=self.output_cursor, len=remaining)
  }) catch {
    _ => return -1
  }
  let count = match written {
    Some(value) => value
    None => return -1
  }
  if count <= 0 || count > remaining {
    return -1
  }
  self.output_cursor += count
  if self.output_cursor == self.output_length {
    1
  } else {
    0
  }
}
