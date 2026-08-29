///|
pub fn LunaHttp1ControlLimits::new(
  max_head_bytes~ : Int,
) -> LunaHttp1ControlLimits raise LunaHttp1Error {
  if max_head_bytes < LUNA_HTTP1_CONTROL_MIN_HEAD_BYTES ||
    max_head_bytes > LUNA_HTTP1_CONTROL_MAX_HEAD_BYTES {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
  }
  { max_head_bytes, }
}

///|
pub fn LunaHttp1ControlWorkspace::new(
  limits : LunaHttp1ControlLimits,
) -> LunaHttp1ControlWorkspace raise LunaHttp1Error {
  if limits.max_head_bytes < LUNA_HTTP1_CONTROL_MIN_HEAD_BYTES ||
    limits.max_head_bytes > LUNA_HTTP1_CONTROL_MAX_HEAD_BYTES {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1Invalid)
  }
  {
    storage: FixedArray::make(limits.max_head_bytes, b'\x00'),
    epoch: 0UL,
    phase: LUNA_HTTP1_CONTROL_IDLE,
    length: 0,
    terminator_state: 0,
    route_code: 0,
  }
}

///|
pub fn LunaHttp1ControlWorkspace::begin(
  self : LunaHttp1ControlWorkspace,
) -> LunaHttp1ControlRequest raise LunaHttp1Error {
  if self.phase != LUNA_HTTP1_CONTROL_IDLE {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Busy)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Exhausted)
  }
  self.epoch += 1UL
  self.phase = LUNA_HTTP1_CONTROL_RECEIVING
  self.length = 0
  self.terminator_state = 0
  self.route_code = 0
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaHttp1ControlRequest::require_workspace(
  self : LunaHttp1ControlRequest,
) -> LunaHttp1ControlWorkspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_HTTP1_CONTROL_IDLE ||
    self.workspace.phase == LUNA_HTTP1_CONTROL_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}

///|
fn luna_http1_control_terminator_next(state : Int, byte : Byte) -> Int {
  match state {
    0 => if byte == b'\r' { 1 } else { 0 }
    1 => if byte == b'\n' { 2 } else if byte == b'\r' { 1 } else { 0 }
    2 => if byte == b'\r' { 3 } else { 0 }
    3 => if byte == b'\n' { 4 } else if byte == b'\r' { 1 } else { 0 }
    _ => 4
  }
}

///|
fn luna_http1_control_ascii_equal(
  storage : FixedArray[Byte],
  start : Int,
  end : Int,
  expected : Bytes,
) -> Bool {
  if start < 0 || end < start || end - start != expected.length() {
    return false
  }
  for index in 0..<expected.length() {
    if storage[start + index] != expected[index] {
      return false
    }
  }
  true
}

///|
fn luna_http1_control_ascii_case_equal(
  storage : FixedArray[Byte],
  start : Int,
  end : Int,
  expected : Bytes,
) -> Bool {
  if start < 0 || end < start || end - start != expected.length() {
    return false
  }
  for index in 0..<expected.length() {
    if luna_http1_lower_ascii(storage[start + index]) != expected[index] {
      return false
    }
  }
  true
}

///|
fn luna_http1_control_trim_left(
  storage : FixedArray[Byte],
  start : Int,
  end : Int,
) -> Int {
  for cursor = start; cursor < end; {
    if storage[cursor] == b' ' || storage[cursor] == b'\t' {
      continue cursor + 1
    }
    break cursor
  } nobreak {
    end
  }
}

///|
fn luna_http1_control_parse_request_line(
  storage : FixedArray[Byte],
  end : Int,
) -> Int {
  let mut first_space = -1
  let mut second_space = -1
  for index in 0..<end {
    let byte = storage[index]
    if byte < b' ' || byte >= b'\x7f' {
      return 0
    }
    if byte == b' ' {
      if first_space < 0 {
        first_space = index
      } else if second_space < 0 {
        second_space = index
      } else {
        return 0
      }
    }
  }
  if first_space <= 0 || second_space <= first_space + 1 {
    return 0
  }
  if !luna_http1_control_ascii_equal(storage, 0, first_space, b"GET") ||
    !luna_http1_control_ascii_equal(storage, second_space + 1, end, b"HTTP/1.1") {
    return 0
  }
  if luna_http1_control_ascii_equal(
      storage,
      first_space + 1,
      second_space,
      b"/healthz",
    ) {
    1
  } else if luna_http1_control_ascii_equal(
      storage,
      first_space + 1,
      second_space,
      b"/readyz",
    ) {
    2
  } else if luna_http1_control_ascii_equal(
      storage,
      first_space + 1,
      second_space,
      b"/metrics",
    ) {
    3
  } else {
    0
  }
}

///|
fn luna_http1_control_validate_head(
  workspace : LunaHttp1ControlWorkspace,
) -> Int {
  let end = workspace.length - 2
  let mut line_start = 0
  let mut line_end = -1
  let mut route = 0
  let mut content_length_seen = false
  for cursor = 0; cursor < end; {
    if cursor + 1 >= end ||
      workspace.storage[cursor] != b'\r' ||
      workspace.storage[cursor + 1] != b'\n' {
      continue cursor + 1
    }
    line_end = cursor
    if route == 0 {
      route = luna_http1_control_parse_request_line(workspace.storage, line_end)
      if route == 0 {
        return 0
      }
    } else {
      if line_end == line_start ||
        workspace.storage[line_start] == b' ' ||
        workspace.storage[line_start] == b'\t' {
        return 0
      }
      let mut colon = -1
      for index in line_start..<line_end {
        let byte = workspace.storage[index]
        if byte == b':' && colon < 0 {
          colon = index
        } else if byte < b' ' || byte >= b'\x7f' {
          return 0
        }
      }
      if colon <= line_start {
        return 0
      }
      let value_start = luna_http1_control_trim_left(
        workspace.storage,
        colon + 1,
        line_end,
      )
      if luna_http1_control_ascii_case_equal(
          workspace.storage,
          line_start,
          colon,
          b"transfer-encoding",
        ) {
        return 0
      }
      if luna_http1_control_ascii_case_equal(
          workspace.storage,
          line_start,
          colon,
          b"content-length",
        ) {
        if content_length_seen ||
          !luna_http1_control_ascii_equal(
            workspace.storage,
            value_start,
            line_end,
            b"0",
          ) {
          return 0
        }
        content_length_seen = true
      }
    }
    line_start = cursor + 2
    continue cursor + 2
  }
  if line_end < 0 || line_start != workspace.length - 2 {
    0
  } else {
    route
  }
}

///|
pub fn LunaHttp1ControlRequest::offer(
  self : LunaHttp1ControlRequest,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_HTTP1_CONTROL_RECEIVING {
    raise luna_http1_error(LunaHttp1Body, LunaHttp1Trailing)
  }
  if source_offset < 0 || length < 0 || source_offset > source.length() - length {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  let mut consumed = 0
  while consumed < length && workspace.terminator_state != 4 {
    if workspace.length == workspace.storage.length() {
      workspace.phase = LUNA_HTTP1_CONTROL_FAILED
      return consumed
    }
    let byte = source[source_offset + consumed]
    workspace.storage[workspace.length] = byte
    workspace.length += 1
    consumed += 1
    workspace.terminator_state = luna_http1_control_terminator_next(
      workspace.terminator_state,
      byte,
    )
  }
  if workspace.terminator_state == 4 {
    let route = luna_http1_control_validate_head(workspace)
    if route == 0 || consumed != length {
      workspace.phase = LUNA_HTTP1_CONTROL_FAILED
    } else {
      workspace.route_code = route
      workspace.phase = LUNA_HTTP1_CONTROL_READY
    }
  }
  consumed
}

///|
pub fn LunaHttp1ControlRequest::finish_input(
  self : LunaHttp1ControlRequest,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase == LUNA_HTTP1_CONTROL_RECEIVING {
    workspace.phase = LUNA_HTTP1_CONTROL_FAILED
  }
}

///|
pub fn LunaHttp1ControlRequest::state(
  self : LunaHttp1ControlRequest,
) -> LunaHttp1ControlRequestState raise LunaHttp1Error {
  match self.require_workspace().phase {
    1 => LunaHttp1ControlReceiving
    2 => LunaHttp1ControlReady
    _ => LunaHttp1ControlFailed
  }
}

///|
pub fn LunaHttp1ControlRequest::take_view(
  self : LunaHttp1ControlRequest,
) -> LunaHttp1ControlRequestView raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_HTTP1_CONTROL_READY {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1NotReady)
  }
  workspace.phase = LUNA_HTTP1_CONTROL_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaHttp1ControlRequest::abort(
  self : LunaHttp1ControlRequest,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.phase = LUNA_HTTP1_CONTROL_IDLE
}

///|
fn LunaHttp1ControlRequestView::require_workspace(
  self : LunaHttp1ControlRequestView,
) -> LunaHttp1ControlWorkspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_HTTP1_CONTROL_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}

///|
pub fn LunaHttp1ControlRequestView::route(
  self : LunaHttp1ControlRequestView,
) -> LunaHttp1ControlRoute raise LunaHttp1Error {
  match self.require_workspace().route_code {
    1 => LunaHttp1Health
    2 => LunaHttp1Readiness
    3 => LunaHttp1Metrics
    _ => abort("Luna HTTP/1 control route invariant")
  }
}

///|
pub fn LunaHttp1ControlRequestView::release(
  self : LunaHttp1ControlRequestView,
) -> Unit raise LunaHttp1Error {
  self.require_workspace().phase = LUNA_HTTP1_CONTROL_IDLE
}
