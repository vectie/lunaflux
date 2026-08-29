///|
pub fn LunaHttp1ControlResponse::new(
  kind : LunaHttp1ControlResponseKind,
) -> LunaHttp1ControlResponse {
  {
    kind: match kind {
      LunaHttp1ControlHealthy => 1
      LunaHttp1ControlUnhealthy => 2
      LunaHttp1ControlReadyResponse => 3
      LunaHttp1ControlNotReadyResponse => 4
      LunaHttp1ControlBadRequest => 5
    },
  }
}

///|
pub fn LunaHttp1ControlResponse::status_code(
  self : LunaHttp1ControlResponse,
) -> Int {
  match self.kind {
    1 | 3 => 200
    2 | 4 => 503
    _ => 400
  }
}

///|
fn luna_http1_control_response_text(kind : Int) -> Bytes {
  match kind {
    1 =>
      b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: 21\r\n\r\n{\"status\":\"healthy\"}\n"
    2 =>
      b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: 23\r\n\r\n{\"status\":\"unhealthy\"}\n"
    3 =>
      b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: 19\r\n\r\n{\"status\":\"ready\"}\n"
    4 =>
      b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: 23\r\n\r\n{\"status\":\"not_ready\"}\n"
    _ =>
      b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: 25\r\n\r\n{\"status\":\"bad_request\"}\n"
  }
}

///|
pub fn LunaHttp1ControlResponse::length(self : LunaHttp1ControlResponse) -> Int {
  luna_http1_control_response_text(self.kind).length()
}

///|
pub fn LunaHttp1ControlResponse::copy_chunk(
  self : LunaHttp1ControlResponse,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaHttp1Error {
  let source = luna_http1_control_response_text(self.kind)
  if destination_offset < 0 ||
    source_offset < 0 ||
    length < 0 ||
    destination_offset > destination.length() - length ||
    source_offset > source.length() - length {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  for index in 0..<length {
    destination[destination_offset + index] = source[source_offset + index]
  }
  length
}

///|
/// Builds one exact operational startup measurement. Absence is permitted only
/// before publication and is always not-ready. Durations are bounded to the
/// same 24-hour cold-start window enforced by the runtime owner.
pub fn LunaHttp1ControlMetricsResponse::new(
  cold_start_latency_millis : Int?,
  readiness~ : Bool,
) -> LunaHttp1ControlMetricsResponse raise LunaHttp1Error {
  match cold_start_latency_millis {
    Some(duration) if duration < 0 || duration > 86400000 =>
      raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
    None if readiness =>
      raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Invalid)
    _ => ()
  }
  let body = match cold_start_latency_millis {
    Some(duration) =>
      "{\"schema\":\"lunaflux.runtime.startup.v1\",\"cold_start_latency_millis\":" +
      duration.to_string() +
      ",\"readiness\":" +
      readiness.to_string() +
      "}\n"
    None =>
      "{\"schema\":\"lunaflux.runtime.startup.v1\",\"cold_start_latency_millis\":null,\"readiness\":false}\n"
  }
  let status = if readiness { 200 } else { 503 }
  let status_line = if status == 200 {
    "HTTP/1.1 200 OK\r\n"
  } else {
    "HTTP/1.1 503 Service Unavailable\r\n"
  }
  let encoded = @utf8.encode(
    status_line +
    "Content-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: " +
    @utf8.encode(body).length().to_string() +
    "\r\n\r\n" +
    body,
  )
  { status, encoded }
}

///|
pub fn LunaHttp1ControlMetricsResponse::status_code(
  self : LunaHttp1ControlMetricsResponse,
) -> Int {
  self.status
}

///|
pub fn LunaHttp1ControlMetricsResponse::length(
  self : LunaHttp1ControlMetricsResponse,
) -> Int {
  self.encoded.length()
}

///|
pub fn LunaHttp1ControlMetricsResponse::copy_chunk(
  self : LunaHttp1ControlMetricsResponse,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaHttp1Error {
  if destination_offset < 0 ||
    source_offset < 0 ||
    length < 0 ||
    destination_offset > destination.length() - length ||
    source_offset > self.encoded.length() - length {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  for index in 0..<length {
    destination[destination_offset + index] = self.encoded[source_offset + index]
  }
  length
}
