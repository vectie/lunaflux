///|
fn luna_framed_event_requirements(
  limits : FramedWireLimits,
) -> (Int, UInt64, UInt64) raise FramedWireError {
  let payload = Int::max(
    limits.inference_limits().max_decoded_delta_bytes(),
    64,
  )
  if payload < 0 || payload > ABSOLUTE_MAX_FRAME_BYTES - EVENT_HEADER_BYTES {
    raise InvalidLimit(FrameBytes)
  }
  let capacity = EVENT_HEADER_BYTES + payload
  if capacity > limits.max_frame_bytes() {
    raise InvalidLimit(FrameBytes)
  }
  (capacity, capacity.to_uint64(), 1UL)
}

///|
pub fn LunaFramedEventWorkspace::required_byte_cells(
  limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  let (_, bytes, _) = luna_framed_event_requirements(limits)
  bytes
}

///|
pub fn LunaFramedEventWorkspace::required_reference_cells(
  limits : FramedWireLimits,
) -> UInt64 raise FramedWireError {
  let (_, _, references) = luna_framed_event_requirements(limits)
  references
}

///|
pub fn LunaFramedEventWorkspace::new(
  limits : FramedWireLimits,
  budget : LunaFramedEventStepBudget,
) -> LunaFramedEventWorkspace raise FramedWireError {
  let (capacity, _, _) = luna_framed_event_requirements(limits)
  {
    limits,
    budget,
    storage: FixedArray::make(capacity, b'\x00'),
    semantic_views: Array::new(capacity=1),
    epoch: 0UL,
    phase: LUNA_FRAMED_EVENT_IDLE,
    event_kind: 0,
    length: 0,
    payload_length: 0,
    cursor: 0,
    checksum: 2166136261,
    failure_rule: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaFramedEventWorkspace::reset_operation(
  self : LunaFramedEventWorkspace,
) -> Unit {
  self.semantic_views.clear()
  self.event_kind = 0
  self.length = 0
  self.payload_length = 0
  self.cursor = 0
  self.checksum = 2166136261
  self.failure_rule = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
fn luna_framed_event_kind_wire(kind : @luna_event.LunaEventKind) -> Int {
  match kind {
    LunaAcceptedEventKind => 1
    LunaTokenEventKind => 2
    LunaUsageEventKind => 3
    LunaCompletedEventKind => 4
    LunaFailedEventKind => 5
  }
}

///|
pub fn LunaFramedEventWorkspace::begin(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> LunaFramedEventWork raise FramedWireError {
  if self.phase != LUNA_FRAMED_EVENT_IDLE {
    raise LunaFramedEventBusy
  }
  let kind = event.kind() catch { _ => raise StaleFrame }
  if self.epoch == 0xffffffffffffffffUL {
    raise LunaFramedEventStale
  }
  self.reset_operation()
  self.epoch += 1UL
  self.event_kind = luna_framed_event_kind_wire(kind)
  self.semantic_views.push(event)
  self.phase = LUNA_FRAMED_EVENT_CLEARING
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaFramedEventWork::require_workspace(
  self : LunaFramedEventWork,
) -> LunaFramedEventWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_FRAMED_EVENT_IDLE ||
    self.workspace.phase == LUNA_FRAMED_EVENT_VIEW_LEASED {
    raise LunaFramedEventStale
  }
  self.workspace
}

///|
fn LunaFramedEventWorkspace::event_view(
  self : LunaFramedEventWorkspace,
) -> @luna_event.LunaEventView {
  if self.semantic_views.length() != 1 {
    abort("Luna framed event semantic-view invariant")
  }
  self.semantic_views[0]
}

///|
fn LunaFramedEventWorkspace::authenticate_event(
  self : LunaFramedEventWorkspace,
) -> @luna_event.LunaEventView raise FramedWireError {
  let event = self.event_view()
  let kind = event.kind() catch { _ => raise StaleFrame }
  if luna_framed_event_kind_wire(kind) != self.event_kind {
    raise StaleFrame
  }
  event
}

///|
fn LunaFramedEventWorkspace::fail(
  self : LunaFramedEventWorkspace,
  error : FramedWireError,
) -> Unit {
  self.failure_rule = match error {
    StaleFrame => 1
    InvalidFrame(rule=Identity, ..) => 2
    InvalidFrame(rule=Scalar, ..) => 3
    InvalidFrame(rule=Length, ..) => 4
    InvalidFrame(rule=Payload, ..) => 5
    FrameTooLarge(_) => 6
    _ => 7
  }
  self.phase = LUNA_FRAMED_EVENT_FAILED
}

///|
fn LunaFramedEventWorkspace::failure_value(
  self : LunaFramedEventWorkspace,
) -> FramedWireError {
  match self.failure_rule {
    1 => StaleFrame
    2 => event_invalid(Identity)
    3 => event_invalid(Scalar)
    4 => event_invalid(Length)
    5 => event_invalid(Payload)
    6 => FrameTooLarge(frame=EventFrame)
    _ => LunaFramedEventFailed
  }
}

///|
pub fn LunaFramedEventWork::state(
  self : LunaFramedEventWork,
) -> LunaFramedEventWorkState raise FramedWireError {
  let workspace = self.require_workspace()
  match workspace.phase {
    LUNA_FRAMED_EVENT_READY => LunaFramedEventReady
    LUNA_FRAMED_EVENT_FAILED => LunaFramedEventFailed
    _ => LunaFramedEventPending
  }
}

///|
pub fn LunaFramedEventWork::last_work_units(
  self : LunaFramedEventWork,
) -> Int raise FramedWireError {
  self.require_workspace().last_work_units
}

///|
pub fn LunaFramedEventWork::total_work_units(
  self : LunaFramedEventWork,
) -> UInt64 raise FramedWireError {
  self.require_workspace().total_work_units
}

///|
pub fn LunaFramedEventWork::failure(
  self : LunaFramedEventWork,
) -> FramedWireError raise FramedWireError {
  let workspace = self.require_workspace()
  if workspace.phase != LUNA_FRAMED_EVENT_FAILED {
    raise LunaFramedEventNotReady
  }
  workspace.failure_value()
}

///|
pub fn LunaFramedEventWork::abort(
  self : LunaFramedEventWork,
) -> Unit raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.semantic_views.clear()
  workspace.phase = LUNA_FRAMED_EVENT_IDLE
  workspace.length = 0
  workspace.payload_length = 0
  workspace.cursor = 0
  workspace.last_work_units = 0
}
