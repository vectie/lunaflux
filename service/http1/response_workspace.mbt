///|
pub fn LunaHttp1ResponseStepBudget::new(
  work_units : Int,
) -> LunaHttp1ResponseStepBudget raise LunaHttp1Error {
  if work_units <= 0 || work_units > LUNA_HTTP1_MAX_STEP_WORK_UNITS {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
  }
  { work_units, }
}

///|
pub fn LunaHttp1ResponseStepBudget::work_units(
  self : LunaHttp1ResponseStepBudget,
) -> Int {
  self.work_units
}

///|
/// Exact fixed byte cells for the largest supported response.
pub fn LunaHttp1ResponseWorkspace::required_byte_cells() -> UInt64 {
  LUNA_HTTP1_RESPONSE_CAPACITY.to_uint64()
}

///|
pub fn LunaHttp1ResponseWorkspace::new(
  budget : LunaHttp1ResponseStepBudget,
) -> LunaHttp1ResponseWorkspace {
  {
    budget,
    storage: FixedArray::make(LUNA_HTTP1_RESPONSE_CAPACITY, b'\x00'),
    epoch: 0UL,
    phase: LUNA_HTTP1_RESPONSE_IDLE,
    response_kind: 0,
    status_code: 0,
    response_length: 0,
    cursor: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaHttp1ResponseWorkspace::begin_kind(
  self : LunaHttp1ResponseWorkspace,
  kind : Int,
  status_code : Int,
  response_length : Int,
) -> LunaHttp1ResponseWork raise LunaHttp1Error {
  if self.phase != LUNA_HTTP1_RESPONSE_IDLE {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Busy)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Exhausted)
  }
  if response_length <= 0 || response_length > self.storage.length() {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1Invalid)
  }
  self.epoch += 1UL
  self.response_kind = kind
  self.status_code = status_code
  self.response_length = response_length
  self.cursor = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
  self.phase = LUNA_HTTP1_RESPONSE_BUILDING
  { workspace: self, epoch: self.epoch }
}

///|
/// Begins the fixed 200 text/event-stream response head.
pub fn LunaHttp1ResponseWorkspace::begin_event_stream(
  self : LunaHttp1ResponseWorkspace,
) -> LunaHttp1ResponseWork raise LunaHttp1Error {
  self.begin_kind(0, 200, 96)
}

///|
/// Begins one payload-free fixed JSON error response.
pub fn LunaHttp1ResponseWorkspace::begin_error(
  self : LunaHttp1ResponseWorkspace,
  kind : LunaHttp1ErrorResponseKind,
) -> LunaHttp1ResponseWork raise LunaHttp1Error {
  match kind {
    LunaHttp1AuthenticationError => self.begin_kind(1, 401, 167)
    LunaHttp1ValidationError => self.begin_kind(2, 400, 136)
    LunaHttp1CapacityError => self.begin_kind(3, 429, 140)
    LunaHttp1ServiceError => self.begin_kind(4, 503, 141)
  }
}

///|
fn LunaHttp1ResponseWork::require_workspace(
  self : LunaHttp1ResponseWork,
) -> LunaHttp1ResponseWorkspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_HTTP1_RESPONSE_IDLE ||
    self.workspace.phase == LUNA_HTTP1_RESPONSE_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}
