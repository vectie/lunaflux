///|
const LUNA_TEXT_IDLE : Int32 = 0

///|
const LUNA_TEXT_WRITING : Int32 = 1

///|
const LUNA_TEXT_HEADER : Int32 = 2

///|
const LUNA_TEXT_SCOPE : Int32 = 3

///|
const LUNA_TEXT_FINAL : Int32 = 6

///|
const LUNA_TEXT_READY : Int32 = 7

///|
const LUNA_TEXT_VIEW : Int32 = 8

///|
const LUNA_TEXT_FAILED : Int32 = 9

///|
const LUNA_TEXT_MAX_WORK : Int32 = 65536

///|
#valtype
pub struct LunaFramedTextRequestStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaFramedTextRequestStepBudget::new(
  work_units : Int32,
) -> Self raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_TEXT_MAX_WORK {
    raise InvalidLimit(LunaFramedTextRequestWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedTextRequestStepBudget::work_units(self : Self) -> Int32 {
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
  priv cache_scope_length : Int32
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut input_length : Int32
  priv mut input_cursor : Int32
  priv mut stop_token_count : Int32
  priv mut stop_token_cursor : Int32
  priv mut stop_string_count : Int32
  priv mut stop_string_cursor : Int32
  priv mut stop_string_bytes : Int32
  priv mut stop_string_remaining : Int32
  priv mut payload_cursor : Int32
  priv mut header_cursor : Int32
  priv mut scope_cursor : Int32
  priv mut protocol_version : Int32
  priv mut request_id : UInt64
  priv mut max_new_tokens : Int32
  priv mut context_ceiling : Int32
  priv mut sampling_mode : Int32
  priv mut top_k : Int32
  priv mut temperature_bits : UInt64
  priv mut top_p_bits : UInt64
  priv mut sampling_seed : UInt64
  priv mut stream : Bool
  priv mut deadline_millis : UInt64
  priv mut last_work_units : Int32
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
