///|
/// Fail-closed rules for the bounded loopback qualification campaign.
pub enum RuntimeOpenAiQualificationValidationRule {
  OpenAiQualificationContract
  OpenAiQualificationOwner
  OpenAiQualificationConnect
  OpenAiQualificationIngress
  OpenAiQualificationResponse
  OpenAiQualificationAuthentication
  OpenAiQualificationEventOrder
  OpenAiQualificationMetrics
  OpenAiQualificationPostDrainAdmission
  OpenAiQualificationCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeOpenAiQualificationValidationRule::code(
  self : RuntimeOpenAiQualificationValidationRule,
) -> String {
  match self {
    OpenAiQualificationContract => "contract"
    OpenAiQualificationOwner => "owner"
    OpenAiQualificationConnect => "connect"
    OpenAiQualificationIngress => "ingress"
    OpenAiQualificationResponse => "response"
    OpenAiQualificationAuthentication => "authentication"
    OpenAiQualificationEventOrder => "event_order"
    OpenAiQualificationMetrics => "metrics"
    OpenAiQualificationPostDrainAdmission => "post_drain_admission"
    OpenAiQualificationCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeOpenAiQualificationValidationError {
  RuntimeOpenAiQualificationValidationFailed(
    RuntimeOpenAiQualificationValidationRule
  )
} derive(Debug, Eq)

///|
/// Exact bounded client authority. The credential is copied from the caller;
/// it is never discovered from ambient process or filesystem state.
pub struct RuntimeOpenAiQualificationValidationContract {
  priv credential : FixedArray[Byte]
  priv turn_limit : Int
  priv io_timeout_millis : Int
  priv maximum_response_bytes : Int
}

///|
/// Evidence scalars become available only after the listener, service, worker
/// child, scheduler, and KV authority have terminally closed.
pub struct RuntimeOpenAiQualificationValidationResult {
  priv event_count : Int
  priv network_accepts : UInt64
  priv network_disconnects : UInt64
  priv network_rejections : UInt64
  priv admissions : UInt64
  priv completions : UInt64
  priv kv_pages_free : Int
}

///|
fn openai_qualification_validation_error(
  rule : RuntimeOpenAiQualificationValidationRule,
) -> RuntimeOpenAiQualificationValidationError {
  RuntimeOpenAiQualificationValidationFailed(rule)
}

///|
fn qualification_credential_byte_allowed(value : Byte) -> Bool {
  value >= b'!' &&
  value <= b'~' &&
  value != b' ' &&
  value != b'\r' &&
  value != b'\n'
}

///|
pub fn RuntimeOpenAiQualificationValidationContract::new(
  credential : FixedArray[Byte],
  turn_limit : Int,
  io_timeout_millis : Int,
  maximum_response_bytes : Int,
) -> RuntimeOpenAiQualificationValidationContract raise RuntimeOpenAiQualificationValidationError {
  if credential.length() == 0 ||
    credential.length() > 4096 ||
    turn_limit <= 0 ||
    turn_limit > 1048576 ||
    io_timeout_millis <= 0 ||
    io_timeout_millis > 60000 ||
    maximum_response_bytes < 1024 ||
    maximum_response_bytes > 1048576 {
    raise openai_qualification_validation_error(OpenAiQualificationContract)
  }
  let owned = FixedArray::make(credential.length(), b'\x00')
  for index in 0..<credential.length() {
    if !qualification_credential_byte_allowed(credential[index]) {
      raise openai_qualification_validation_error(OpenAiQualificationContract)
    }
    owned[index] = credential[index]
  }
  { credential: owned, turn_limit, io_timeout_millis, maximum_response_bytes }
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::event_count(
  self : RuntimeOpenAiQualificationValidationResult,
) -> Int {
  self.event_count
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::network_accepts(
  self : RuntimeOpenAiQualificationValidationResult,
) -> UInt64 {
  self.network_accepts
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::network_disconnects(
  self : RuntimeOpenAiQualificationValidationResult,
) -> UInt64 {
  self.network_disconnects
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::network_rejections(
  self : RuntimeOpenAiQualificationValidationResult,
) -> UInt64 {
  self.network_rejections
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::admissions(
  self : RuntimeOpenAiQualificationValidationResult,
) -> UInt64 {
  self.admissions
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::completions(
  self : RuntimeOpenAiQualificationValidationResult,
) -> UInt64 {
  self.completions
}

///|
pub fn RuntimeOpenAiQualificationValidationResult::kv_pages_free(
  self : RuntimeOpenAiQualificationValidationResult,
) -> Int {
  self.kv_pages_free
}
