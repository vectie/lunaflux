///|
fn LunaHttp1Workspace::accept_head_byte(
  self : LunaHttp1Workspace,
  byte : Byte,
) -> Unit {
  self.storage[self.head_length] = byte
  self.head_length += 1
  self.terminator_state = match (self.terminator_state, byte) {
    (0, b'\r') => 1
    (1, b'\n') => 2
    (1, b'\r') => 1
    (2, b'\r') => 3
    (3, b'\n') => 4
    (_, b'\r') => 1
    _ => 0
  }
  if self.terminator_state == 4 {
    self.phase = LUNA_HTTP1_SCANNING_LINE
    self.scan_cursor = 0
    self.line_start = 0
    self.scan_expect_lf = false
  } else if self.head_length == self.limits.max_head_bytes {
    self.fail(LunaHttp1Head, LunaHttp1TooLong)
  }
}

///|
fn LunaHttp1Workspace::accept_body_byte(
  self : LunaHttp1Workspace,
  byte : Byte,
) -> Unit {
  self.storage[self.limits.max_head_bytes + self.body_written] = byte
  self.body_written += 1
  if self.body_written == self.body_length {
    self.phase = LUNA_HTTP1_FINALIZING
  }
}

///|
/// Copies no more than one configured quantum and never retains `source`.
pub fn LunaHttp1Work::offer(
  self : LunaHttp1Work,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Int raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if source_offset < 0 ||
    length < 0 ||
    source_offset > source.length() ||
    length > source.length() - source_offset {
    raise luna_http1_error(LunaHttp1Source, LunaHttp1OutOfRange)
  }
  if length == 0 {
    return 0
  }
  if workspace.phase == LUNA_HTTP1_FAILED {
    raise workspace.failure_value()
  }
  if workspace.phase == LUNA_HTTP1_READY {
    workspace.fail(LunaHttp1Body, LunaHttp1Trailing)
    raise workspace.failure_value()
  }
  let mut consumed = 0
  let budget = workspace.budget.work_units()
  while consumed < length &&
        consumed < budget &&
        (
          workspace.phase == LUNA_HTTP1_RECEIVING_HEAD ||
          workspace.phase == LUNA_HTTP1_RECEIVING_BODY
        ) {
    let byte = source[source_offset + consumed]
    if workspace.phase == LUNA_HTTP1_RECEIVING_HEAD {
      workspace.accept_head_byte(byte)
    } else {
      workspace.accept_body_byte(byte)
    }
    consumed += 1
  }
  workspace.charge(consumed)
  consumed
}

///|
/// Records transport EOF. Validation and authentication may still progress.
pub fn LunaHttp1Work::finish_input(
  self : LunaHttp1Work,
) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_HTTP1_FAILED {
    raise workspace.failure_value()
  }
  if workspace.input_ended {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1NotReady)
  }
  workspace.input_ended = true
  if workspace.phase == LUNA_HTTP1_RECEIVING_HEAD {
    workspace.fail(LunaHttp1Head, LunaHttp1PrematureEnd)
  } else if workspace.phase == LUNA_HTTP1_RECEIVING_BODY &&
    workspace.body_written != workspace.body_length {
    workspace.fail(LunaHttp1Body, LunaHttp1PrematureEnd)
  }
  workspace.charge(1)
}

///|
pub fn LunaHttp1Work::state(
  self : LunaHttp1Work,
) -> LunaHttp1WorkState raise LunaHttp1Error {
  let workspace = self.require_workspace()
  match workspace.phase {
    LUNA_HTTP1_RECEIVING_HEAD => LunaHttp1ReceivingHead
    LUNA_HTTP1_AUTH_BEGIN
    | LUNA_HTTP1_AUTH_WRITE
    | LUNA_HTTP1_AUTH_FINISH
    | LUNA_HTTP1_AUTH_PROGRESS
    | LUNA_HTTP1_AUTH_DECIDE => LunaHttp1Authenticating
    LUNA_HTTP1_RECEIVING_BODY => LunaHttp1ReceivingBody
    LUNA_HTTP1_READY => LunaHttp1Ready
    LUNA_HTTP1_FAILED => LunaHttp1Failed
    _ => LunaHttp1ValidatingHead
  }
}

///|
pub fn LunaHttp1Work::last_work_units(
  self : LunaHttp1Work,
) -> Int raise LunaHttp1Error {
  self.require_workspace().last_work_units
}

///|
pub fn LunaHttp1Work::total_work_units(
  self : LunaHttp1Work,
) -> UInt64 raise LunaHttp1Error {
  self.require_workspace().total_work_units
}

///|
pub fn LunaHttp1Work::failure(
  self : LunaHttp1Work,
) -> LunaHttp1Error raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_HTTP1_FAILED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1NotReady)
  }
  workspace.failure_value()
}

///|
pub fn LunaHttp1Work::take_view(
  self : LunaHttp1Work,
) -> LunaHttp1View raise LunaHttp1Error {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_HTTP1_READY {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1NotReady)
  }
  workspace.phase = LUNA_HTTP1_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaHttp1Work::abort(self : LunaHttp1Work) -> Unit raise LunaHttp1Error {
  let workspace = self.require_workspace()
  workspace.clear_auth()
  workspace.failure_rule = 0
  workspace.failure_issue = 0
  workspace.last_work_units = 0
  workspace.phase = LUNA_HTTP1_IDLE
}
