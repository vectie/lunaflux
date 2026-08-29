///|
fn luna_http1_error(
  rule : LunaHttp1Rule,
  issue : LunaHttp1Issue,
) -> LunaHttp1Error {
  LunaHttp1Failed(rule~, issue~)
}

///|
pub fn LunaHttp1Limits::new(
  max_head_bytes~ : Int,
  max_body_bytes~ : Int,
) -> LunaHttp1Limits raise LunaHttp1Error {
  if max_head_bytes < LUNA_HTTP1_MIN_HEAD_BYTES ||
    max_head_bytes > LUNA_HTTP1_MAX_HEAD_BYTES {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
  }
  if max_body_bytes < 0 || max_body_bytes > LUNA_HTTP1_MAX_BODY_BYTES {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
  }
  { max_head_bytes, max_body_bytes }
}

///|
pub fn LunaHttp1StepBudget::new(
  work_units : Int,
) -> LunaHttp1StepBudget raise LunaHttp1Error {
  if work_units <= 0 || work_units > LUNA_HTTP1_MAX_STEP_WORK_UNITS {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1OutOfRange)
  }
  { work_units, }
}

///|
pub fn LunaHttp1StepBudget::work_units(self : LunaHttp1StepBudget) -> Int {
  self.work_units
}

///|
fn luna_http1_requirements(
  limits : LunaHttp1Limits,
) -> (Int, UInt64, UInt64) raise LunaHttp1Error {
  if limits.max_head_bytes < LUNA_HTTP1_MIN_HEAD_BYTES ||
    limits.max_head_bytes > LUNA_HTTP1_MAX_HEAD_BYTES ||
    limits.max_body_bytes < 0 ||
    limits.max_body_bytes > LUNA_HTTP1_MAX_BODY_BYTES {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1Invalid)
  }
  let head = limits.max_head_bytes.to_uint64()
  let body = limits.max_body_bytes.to_uint64()
  if head > 0xffffffffffffffffUL - body {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1TooLong)
  }
  let cells = head + body
  if cells > 0x7fffffffUL {
    raise luna_http1_error(LunaHttp1Limits, LunaHttp1TooLong)
  }
  (cells.to_int(), cells, 2UL)
}

///|
/// Exact byte cells owned by the HTTP head and body backing.
pub fn LunaHttp1Workspace::required_byte_cells(
  limits : LunaHttp1Limits,
) -> UInt64 raise LunaHttp1Error {
  let (_, bytes, _) = luna_http1_requirements(limits)
  bytes
}

///|
/// Exact preallocated capability slots used by cooperative authentication.
pub fn LunaHttp1Workspace::required_reference_cells(
  limits : LunaHttp1Limits,
) -> UInt64 raise LunaHttp1Error {
  let (_, _, references) = luna_http1_requirements(limits)
  references
}

///|
pub fn LunaHttp1Workspace::new(
  limits : LunaHttp1Limits,
  budget : LunaHttp1StepBudget,
  policy : @api_auth.LunaApiAuthPolicy,
) -> LunaHttp1Workspace raise LunaHttp1Error {
  let (cells, _, _) = luna_http1_requirements(limits)
  let auth_budget = @api_auth.LunaApiAuthStepBudget::new(1) catch {
    _ => raise luna_http1_error(LunaHttp1Authorization, LunaHttp1Invalid)
  }
  let auth_verifier = @api_auth.LunaApiAuthVerifierWorkspace::new(
    policy, auth_budget,
  ) catch {
    _ => raise luna_http1_error(LunaHttp1Authorization, LunaHttp1Invalid)
  }
  {
    limits,
    budget,
    storage: FixedArray::make(cells, b'\x00'),
    auth_verifier,
    auth_writes: Array::new(capacity=1),
    auth_works: Array::new(capacity=1),
    epoch: 0UL,
    phase: LUNA_HTTP1_IDLE,
    failure_rule: 0,
    failure_issue: 0,
    head_length: 0,
    body_length: 0,
    body_written: 0,
    terminator_state: 0,
    input_ended: false,
    scan_cursor: 0,
    scan_expect_lf: false,
    line_start: 0,
    line_end: 0,
    request_line_complete: false,
    parse_component: 0,
    parse_index: 0,
    route_responses: true,
    route_chat: true,
    header_kind: LUNA_HTTP1_HEADER_UNKNOWN,
    header_name_index: 0,
    header_authorization: true,
    header_content_length: true,
    header_content_type: true,
    header_transfer_encoding: true,
    value_index: 0,
    value_leading_space: false,
    value_number: 0,
    value_has_digit: false,
    value_content_type: true,
    credential_offset: 0,
    credential_length: 0,
    credential_padding: false,
    authorization_seen: false,
    content_length_seen: false,
    content_type_seen: false,
    duplicate_content_length: false,
    route_code: 0,
    auth_cursor: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
/// Closes the authentication owner retained by this parser. The underlying
/// policy wipe is idempotent and shared by every verifier alias.
pub fn LunaHttp1Workspace::close_authentication(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.phase == LUNA_HTTP1_CLOSED {
    return
  }
  self.clear_auth()
  self.auth_verifier.close()
  for index in 0..<self.storage.length() {
    self.storage[index] = b'\x00'
  }
  self.head_length = 0
  self.body_length = 0
  self.body_written = 0
  self.credential_offset = 0
  self.credential_length = 0
  self.phase = LUNA_HTTP1_CLOSED
}

///|
fn LunaHttp1Workspace::clear_auth(self : LunaHttp1Workspace) -> Unit {
  if self.auth_writes.length() == 1 {
    try! self.auth_writes[0].abort()
  }
  self.auth_writes.clear()
  if self.auth_works.length() == 1 {
    try! self.auth_works[0].abort()
  }
  self.auth_works.clear()
}

///|
/// Erases only the presented credential bytes. Request-head and body storage
/// is lifecycle-protected and overwritten by later input; sweeping it on every
/// reuse would add an O(request-bytes) pass without providing end-to-end secret
/// erasure for downstream semantic copies.
fn LunaHttp1Workspace::wipe_credential(self : LunaHttp1Workspace) -> Unit {
  if self.credential_offset >= 0 &&
    self.credential_length >= 0 &&
    self.credential_offset <= self.limits.max_head_bytes &&
    self.credential_length <=
    self.limits.max_head_bytes - self.credential_offset {
    for index in 0..<self.credential_length {
      self.storage[self.credential_offset + index] = b'\x00'
    }
  }
}

///|
fn LunaHttp1Workspace::reset_operation(self : LunaHttp1Workspace) -> Unit {
  if self.auth_writes.length() != 0 || self.auth_works.length() != 0 {
    abort("Luna HTTP/1 authentication-slot invariant")
  }
  self.wipe_credential()
  self.failure_rule = 0
  self.failure_issue = 0
  self.head_length = 0
  self.body_length = 0
  self.body_written = 0
  self.terminator_state = 0
  self.input_ended = false
  self.scan_cursor = 0
  self.scan_expect_lf = false
  self.line_start = 0
  self.line_end = 0
  self.request_line_complete = false
  self.parse_component = 0
  self.parse_index = 0
  self.route_responses = true
  self.route_chat = true
  self.header_kind = LUNA_HTTP1_HEADER_UNKNOWN
  self.header_name_index = 0
  self.header_authorization = true
  self.header_content_length = true
  self.header_content_type = true
  self.header_transfer_encoding = true
  self.value_index = 0
  self.value_leading_space = false
  self.value_number = 0
  self.value_has_digit = false
  self.value_content_type = true
  self.credential_offset = 0
  self.credential_length = 0
  self.credential_padding = false
  self.authorization_seen = false
  self.content_length_seen = false
  self.content_type_seen = false
  self.duplicate_content_length = false
  self.route_code = 0
  self.auth_cursor = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
pub fn LunaHttp1Workspace::begin(
  self : LunaHttp1Workspace,
) -> LunaHttp1Work raise LunaHttp1Error {
  if self.phase != LUNA_HTTP1_IDLE {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Busy)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Exhausted)
  }
  self.reset_operation()
  self.epoch += 1UL
  self.phase = LUNA_HTTP1_RECEIVING_HEAD
  { workspace: self, epoch: self.epoch }
}

///|
fn LunaHttp1Work::require_workspace(
  self : LunaHttp1Work,
) -> LunaHttp1Workspace raise LunaHttp1Error {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase == LUNA_HTTP1_IDLE ||
    self.workspace.phase == LUNA_HTTP1_CLOSED ||
    self.workspace.phase == LUNA_HTTP1_VIEW_LEASED {
    raise luna_http1_error(LunaHttp1Lifecycle, LunaHttp1Stale)
  }
  self.workspace
}

///|
fn luna_http1_rule_code(rule : LunaHttp1Rule) -> Int {
  match rule {
    LunaHttp1Limits => 1
    LunaHttp1Source => 2
    LunaHttp1RequestLine => 3
    LunaHttp1Method => 4
    LunaHttp1RouteRule => 5
    LunaHttp1Version => 6
    LunaHttp1Header => 7
    LunaHttp1Authorization => 8
    LunaHttp1ContentLength => 9
    LunaHttp1ContentType => 10
    LunaHttp1TransferEncoding => 11
    LunaHttp1Head => 12
    LunaHttp1Body => 13
    LunaHttp1Lifecycle => 14
  }
}

///|
fn luna_http1_issue_code(issue : LunaHttp1Issue) -> Int {
  match issue {
    LunaHttp1Invalid => 1
    LunaHttp1Missing => 2
    LunaHttp1Duplicate => 3
    LunaHttp1Conflicting => 4
    LunaHttp1Unsupported => 5
    LunaHttp1TooLong => 6
    LunaHttp1OutOfRange => 7
    LunaHttp1Rejected => 8
    LunaHttp1PrematureEnd => 9
    LunaHttp1Trailing => 10
    LunaHttp1Busy => 11
    LunaHttp1NotReady => 12
    LunaHttp1Stale => 13
    LunaHttp1Exhausted => 14
  }
}

///|
fn LunaHttp1Workspace::fail(
  self : LunaHttp1Workspace,
  rule : LunaHttp1Rule,
  issue : LunaHttp1Issue,
) -> Unit {
  self.failure_rule = luna_http1_rule_code(rule)
  self.failure_issue = luna_http1_issue_code(issue)
  self.phase = LUNA_HTTP1_FAILED
}

///|
fn LunaHttp1Workspace::failure_value(
  self : LunaHttp1Workspace,
) -> LunaHttp1Error {
  let rule = match self.failure_rule {
    1 => LunaHttp1Limits
    2 => LunaHttp1Source
    3 => LunaHttp1RequestLine
    4 => LunaHttp1Method
    5 => LunaHttp1RouteRule
    6 => LunaHttp1Version
    7 => LunaHttp1Header
    8 => LunaHttp1Authorization
    9 => LunaHttp1ContentLength
    10 => LunaHttp1ContentType
    11 => LunaHttp1TransferEncoding
    12 => LunaHttp1Head
    13 => LunaHttp1Body
    _ => LunaHttp1Lifecycle
  }
  let issue = match self.failure_issue {
    1 => LunaHttp1Invalid
    2 => LunaHttp1Missing
    3 => LunaHttp1Duplicate
    4 => LunaHttp1Conflicting
    5 => LunaHttp1Unsupported
    6 => LunaHttp1TooLong
    7 => LunaHttp1OutOfRange
    8 => LunaHttp1Rejected
    9 => LunaHttp1PrematureEnd
    10 => LunaHttp1Trailing
    11 => LunaHttp1Busy
    12 => LunaHttp1NotReady
    13 => LunaHttp1Stale
    _ => LunaHttp1Exhausted
  }
  luna_http1_error(rule, issue)
}

///|
fn LunaHttp1Workspace::charge(self : LunaHttp1Workspace, units : Int) -> Unit {
  self.last_work_units = units
  self.total_work_units += units.to_uint64()
}
