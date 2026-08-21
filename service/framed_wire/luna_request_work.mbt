///|
fn luna_framed_request_checked_multiply_cells(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise FramedWireError {
  if right != 0UL && left > 0xffffffffffffffffUL / right {
    raise InvalidLimit(FrameBytes)
  }
  left * right
}

///|
fn luna_framed_request_workspace_requirements(
  limits : FramedWireLimits,
) -> (Int, Int, UInt64, UInt64) raise FramedWireError {
  let byte_capacity = limits.max_frame_bytes()
  if byte_capacity < REQUEST_HEADER_BYTES ||
    byte_capacity > ABSOLUTE_MAX_FRAME_BYTES {
    raise InvalidLimit(FrameBytes)
  }
  let stop_capacity = limits.inference_limits().max_stop_strings()
  if stop_capacity <= 0 {
    raise InvalidLimit(FrameBytes)
  }
  let byte_cells = byte_capacity.to_uint64()
  let int_cells = luna_framed_request_checked_multiply_cells(
    stop_capacity.to_uint64(),
    2UL,
  )
  (byte_capacity, stop_capacity, byte_cells, int_cells)
}

///|
/// Exact fixed integer cells used by the stop-string offset and length tables.
pub fn LunaFramedRequestWorkspace::required_int_cells(
  limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  let (_, _, _, int_cells) = luna_framed_request_workspace_requirements(limits)
  int_cells
}

///|
/// Exact fixed byte cells used by the canonical request-frame backing.
pub fn LunaFramedRequestWorkspace::required_byte_cells(
  limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  let (_, _, byte_cells, _) = luna_framed_request_workspace_requirements(limits)
  byte_cells
}

///|
fn new_luna_framed_request_workspace(
  limits : FramedWireLimits,
  budget : LunaFramedRequestStepBudget,
  storage : FixedArray[Byte],
) -> LunaFramedRequestWorkspace {
  let (byte_capacity, stop_capacity, _, _) = try! luna_framed_request_workspace_requirements(
    limits,
  )
  if storage.length() != byte_capacity {
    abort("Luna framed request storage-capacity invariant")
  }
  {
    limits,
    budget,
    storage,
    stop_string_offsets: FixedArray::make(stop_capacity, 0),
    stop_string_lengths: FixedArray::make(stop_capacity, 0),
    epoch: 0UL,
    phase: LUNA_FRAMED_REQUEST_IDLE,
    failure: None,
    length: 0,
    declared_length: 0,
    flags: 0,
    input_count: 0,
    stop_token_count: 0,
    stop_string_count: 0,
    declared_stop_bytes: 0,
    scope_length: 0,
    trace_length: 0,
    input_offset: REQUEST_HEADER_BYTES,
    stop_token_offset: 0,
    scope_offset: 0,
    trace_offset: 0,
    phase_index: 0,
    nested_index: 0,
    byte_index: 0,
    cursor: 0,
    actual_stop_bytes: 0,
    checksum: 2166136261,
    utf8_remaining: 0,
    utf8_codepoint: 0,
    utf8_minimum: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
pub fn LunaFramedRequestWorkspace::new(
  limits : FramedWireLimits,
  budget : LunaFramedRequestStepBudget,
) -> LunaFramedRequestWorkspace {
  let (byte_capacity, _, _, _) = try! luna_framed_request_workspace_requirements(
    limits,
  )
  new_luna_framed_request_workspace(
    limits,
    budget,
    FixedArray::make(byte_capacity, b'\x00'),
  )
}

///|
fn LunaFramedRequestWorkspace::reset_operation(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  self.failure = None
  self.length = 0
  self.declared_length = 0
  self.flags = 0
  self.input_count = 0
  self.stop_token_count = 0
  self.stop_string_count = 0
  self.declared_stop_bytes = 0
  self.scope_length = 0
  self.trace_length = 0
  self.input_offset = REQUEST_HEADER_BYTES
  self.stop_token_offset = 0
  self.scope_offset = 0
  self.trace_offset = 0
  self.phase_index = 0
  self.nested_index = 0
  self.byte_index = 0
  self.cursor = 0
  self.actual_stop_bytes = 0
  self.checksum = 2166136261
  self.utf8_remaining = 0
  self.utf8_codepoint = 0
  self.utf8_minimum = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
pub fn LunaFramedRequestWorkspace::begin(
  self : LunaFramedRequestWorkspace,
) -> LunaFramedRequestWork raise FramedWireError {
  if self.phase != LUNA_FRAMED_REQUEST_IDLE {
    raise LunaFramedRequestBusy
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaFramedRequestStale
  }
  self.epoch += 1UL
  self.reset_operation()
  self.phase = LUNA_FRAMED_REQUEST_RECEIVING_PREFIX
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaFramedRequestWork::require_workspace(
  self : LunaFramedRequestWork,
) -> LunaFramedRequestWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_FRAMED_REQUEST_IDLE ||
    self.workspace.phase == LUNA_FRAMED_REQUEST_VIEW_LEASED {
    raise LunaFramedRequestStale
  }
  self.workspace
}

///|
fn LunaFramedRequestWorkspace::fail(
  self : LunaFramedRequestWorkspace,
  error : FramedWireError,
) -> Unit {
  self.failure = Some(error)
  self.phase = LUNA_FRAMED_REQUEST_FAILED
}

///|
fn LunaFramedRequestWorkspace::charge(
  self : LunaFramedRequestWorkspace,
  work_units : Int,
) -> Unit {
  self.last_work_units = work_units
  self.total_work_units += work_units.to_uint64()
}

///|
/// Copies at most one configured quantum and returns the exact consumed byte
/// count. Unconsumed bytes, including a following pipelined frame, remain with
/// the caller.
pub fn LunaFramedRequestWork::offer(
  self : LunaFramedRequestWork,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if source_offset < 0 || length < 0 || source_offset > source.length() - length {
    let error = request_invalid(Length)
    workspace.fail(error)
    raise error
  }
  if length == 0 {
    return 0
  }
  let capacity = match workspace.phase {
    LUNA_FRAMED_REQUEST_RECEIVING_PREFIX => 16 - workspace.length
    LUNA_FRAMED_REQUEST_RECEIVING_BODY =>
      workspace.declared_length - workspace.length
    LUNA_FRAMED_REQUEST_FAILED => raise LunaFramedRequestFailed
    LUNA_FRAMED_REQUEST_READY => raise LunaFramedRequestNotReady
    _ => return 0
  }
  let amount = Int::min(
    length,
    Int::min(capacity, workspace.budget.work_units()),
  )
  for index in 0..<amount {
    workspace.storage[workspace.length + index] = source[source_offset + index]
  }
  workspace.length += amount
  workspace.charge(amount)
  if workspace.phase == LUNA_FRAMED_REQUEST_RECEIVING_PREFIX &&
    workspace.length == 16 {
    workspace.phase = LUNA_FRAMED_REQUEST_VALIDATING_PREFIX
    workspace.phase_index = 0
  } else if workspace.phase == LUNA_FRAMED_REQUEST_RECEIVING_BODY &&
    workspace.length == workspace.declared_length {
    workspace.phase = LUNA_FRAMED_REQUEST_CHECKSUM
    workspace.phase_index = 0
    workspace.checksum = 2166136261
  }
  amount
}

///|
pub fn LunaFramedRequestWork::state(
  self : LunaFramedRequestWork,
) -> LunaFramedRequestWorkState raise FramedWireError {
  let workspace = self.require_workspace()
  match workspace.phase {
    LUNA_FRAMED_REQUEST_RECEIVING_PREFIX => LunaFramedRequestReceivingPrefix
    LUNA_FRAMED_REQUEST_RECEIVING_BODY => LunaFramedRequestReceivingBody
    LUNA_FRAMED_REQUEST_READY => LunaFramedRequestReady
    LUNA_FRAMED_REQUEST_FAILED => LunaFramedRequestFailed
    _ => LunaFramedRequestValidating
  }
}

///|
pub fn LunaFramedRequestWork::last_work_units(
  self : LunaFramedRequestWork,
) -> Int raise FramedWireError {
  self.require_workspace().last_work_units
}

///|
pub fn LunaFramedRequestWork::total_work_units(
  self : LunaFramedRequestWork,
) -> UInt64 raise FramedWireError {
  self.require_workspace().total_work_units
}

///|
pub fn LunaFramedRequestWork::failure(
  self : LunaFramedRequestWork,
) -> FramedWireError raise FramedWireError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_FRAMED_REQUEST_FAILED {
    raise LunaFramedRequestNotReady
  }
  workspace.failure.unwrap()
}

///|
pub fn LunaFramedRequestWork::take_view(
  self : LunaFramedRequestWork,
) -> LunaFramedRequestView raise FramedWireError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_FRAMED_REQUEST_READY {
    raise LunaFramedRequestNotReady
  }
  workspace.phase = LUNA_FRAMED_REQUEST_VIEW_LEASED
  { workspace, epoch: self.epoch }
}

///|
pub fn LunaFramedRequestWork::abort(
  self : LunaFramedRequestWork,
) -> Unit raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.failure = None
  workspace.phase = LUNA_FRAMED_REQUEST_IDLE
  workspace.last_work_units = 0
}

///|
fn LunaFramedRequestWorkspace::progress_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  match self.phase {
    LUNA_FRAMED_REQUEST_VALIDATING_PREFIX => self.progress_prefix_one()
    LUNA_FRAMED_REQUEST_CHECKSUM => self.progress_checksum_one()
    LUNA_FRAMED_REQUEST_HEADER => self.progress_header_one()
    LUNA_FRAMED_REQUEST_DIGEST => self.progress_digest_one()
    LUNA_FRAMED_REQUEST_LAYOUT => self.progress_layout_one()
    LUNA_FRAMED_REQUEST_INPUT => self.progress_input_one()
    LUNA_FRAMED_REQUEST_STOP_TOKENS => self.progress_stop_token_one()
    LUNA_FRAMED_REQUEST_STOP_TOKEN_DUPLICATES =>
      self.progress_stop_token_duplicate_one()
    LUNA_FRAMED_REQUEST_STOP_STRINGS => self.progress_stop_string_one()
    LUNA_FRAMED_REQUEST_STOP_STRING_DUPLICATES =>
      self.progress_stop_string_duplicate_one()
    LUNA_FRAMED_REQUEST_SCOPE => self.progress_scope_one()
    LUNA_FRAMED_REQUEST_TRACE => self.progress_trace_one()
    _ => ()
  }
}

///|
pub fn LunaFramedRequestWork::progress(
  self : LunaFramedRequestWork,
) -> LunaFramedRequestProgress raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  let mut used = 0
  while used < workspace.budget.work_units() &&
        workspace.phase != LUNA_FRAMED_REQUEST_RECEIVING_PREFIX &&
        workspace.phase != LUNA_FRAMED_REQUEST_RECEIVING_BODY &&
        workspace.phase != LUNA_FRAMED_REQUEST_READY &&
        workspace.phase != LUNA_FRAMED_REQUEST_FAILED {
    workspace.progress_one()
    used += 1
  }
  workspace.charge(used)
  if used == 0 {
    LunaFramedRequestIdle
  } else {
    LunaFramedRequestAdvanced
  }
}
