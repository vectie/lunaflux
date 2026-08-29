///|
const LUNA_TEXT_IDLE : Int = 0

///|
const LUNA_TEXT_WRITING : Int = 1

///|
const LUNA_TEXT_HEADER : Int = 2

///|
const LUNA_TEXT_SCOPE : Int = 3

///|
const LUNA_TEXT_CHECKSUM : Int = 4

///|
const LUNA_TEXT_CHECKSUM_WRITE : Int = 5

///|
const LUNA_TEXT_FINAL : Int = 6

///|
const LUNA_TEXT_READY : Int = 7

///|
const LUNA_TEXT_VIEW : Int = 8

///|
const LUNA_TEXT_FAILED : Int = 9

///|
const LUNA_TEXT_MAX_WORK : Int = 65536

///|
#valtype
pub struct LunaFramedTextRequestStepBudget {
  priv work_units : Int
}

///|
pub fn LunaFramedTextRequestStepBudget::new(
  work_units : Int,
) -> Self raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_TEXT_MAX_WORK {
    raise InvalidLimit(LunaFramedTextRequestWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedTextRequestStepBudget::work_units(self : Self) -> Int {
  self.work_units
}

///|
#valtype
pub(all) enum LunaFramedTextRequestWorkState {
  LunaFramedTextRequestPending
  LunaFramedTextRequestReady
  LunaFramedTextRequestFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaFramedTextRequestProgress {
  LunaFramedTextRequestIdle
  LunaFramedTextRequestAdvanced
} derive(Debug, Eq)

///|
pub struct LunaFramedTextRequestWorkspace {
  priv limits : FramedWireLimits
  priv budget : LunaFramedTextRequestStepBudget
  priv storage : FixedArray[Byte]
  priv content_digest : FixedArray[Byte]
  priv plan_digest : FixedArray[Byte]
  priv cache_scope : FixedArray[Byte]
  priv cache_scope_length : Int
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut input_length : Int
  priv mut input_cursor : Int
  priv mut stop_token_count : Int
  priv mut stop_token_cursor : Int
  priv mut stop_string_count : Int
  priv mut stop_string_cursor : Int
  priv mut stop_string_bytes : Int
  priv mut stop_string_remaining : Int
  priv mut payload_cursor : Int
  priv mut header_cursor : Int
  priv mut scope_cursor : Int
  priv mut checksum_cursor : Int
  priv mut checksum : UInt
  priv mut checksum_write_cursor : Int
  priv mut protocol_version : Int
  priv mut request_id : UInt64
  priv mut max_new_tokens : Int
  priv mut context_ceiling : Int
  priv mut sampling_mode : Int
  priv mut top_k : Int
  priv mut temperature_bits : UInt64
  priv mut top_p_bits : UInt64
  priv mut sampling_seed : UInt64
  priv mut stream : Bool
  priv mut deadline_millis : UInt64
  priv mut last_work_units : Int
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaFramedTextRequestWrite {
  priv workspace : LunaFramedTextRequestWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaFramedTextRequestWork {
  priv workspace : LunaFramedTextRequestWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaFramedTextRequestView {
  priv workspace : LunaFramedTextRequestWorkspace
  priv epoch : UInt64
}
