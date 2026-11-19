///|
const LUNA_FRAMED_REQUEST_IDLE : Int32 = 0

///|
const LUNA_FRAMED_REQUEST_RECEIVING_PREFIX : Int32 = 1

///|
const LUNA_FRAMED_REQUEST_VALIDATING_PREFIX : Int32 = 2

///|
const LUNA_FRAMED_REQUEST_RECEIVING_BODY : Int32 = 3

///|
const LUNA_FRAMED_REQUEST_HEADER : Int32 = 5

///|
const LUNA_FRAMED_REQUEST_DIGEST : Int32 = 6

///|
const LUNA_FRAMED_REQUEST_LAYOUT : Int32 = 7

///|
const LUNA_FRAMED_REQUEST_INPUT : Int32 = 8

///|
const LUNA_FRAMED_REQUEST_STOP_TOKENS : Int32 = 9

///|
const LUNA_FRAMED_REQUEST_STOP_TOKEN_DUPLICATES : Int32 = 10

///|
const LUNA_FRAMED_REQUEST_STOP_STRINGS : Int32 = 11

///|
const LUNA_FRAMED_REQUEST_STOP_STRING_DUPLICATES : Int32 = 12

///|
const LUNA_FRAMED_REQUEST_SCOPE : Int32 = 13

///|
const LUNA_FRAMED_REQUEST_TRACE : Int32 = 14

///|
const LUNA_FRAMED_REQUEST_READY : Int32 = 15

///|
const LUNA_FRAMED_REQUEST_VIEW_LEASED : Int32 = 16

///|
const LUNA_FRAMED_REQUEST_FAILED : Int32 = 17

///|
const LUNA_FRAMED_REQUEST_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
#valtype
pub struct LunaFramedRequestStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaFramedRequestStepBudget::new(
  work_units : Int32,
) -> LunaFramedRequestStepBudget raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_FRAMED_REQUEST_MAX_STEP_WORK_UNITS {
    raise InvalidLimit(LunaFramedRequestWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedRequestStepBudget::work_units(
  self : LunaFramedRequestStepBudget,
) -> Int32 {
  self.work_units
}

///|
#valtype
pub(all) enum LunaFramedRequestWorkState {
  LunaFramedRequestReceivingPrefix
  LunaFramedRequestReceivingBody
  LunaFramedRequestValidating
  LunaFramedRequestReady
  LunaFramedRequestFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaFramedRequestProgress {
  LunaFramedRequestIdle
  LunaFramedRequestAdvanced
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaFramedRequestInputKind {
  LunaFramedRequestText
  LunaFramedRequestTokenIds
} derive(Debug, Eq)

///|
pub struct LunaFramedRequestWorkspace {
  priv limits : FramedWireLimits
  priv budget : LunaFramedRequestStepBudget
  priv storage : FixedArray[Byte]
  priv stop_string_offsets : FixedArray[Int32]
  priv stop_string_lengths : FixedArray[Int32]
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut failure : FramedWireError?
  priv mut length : Int32
  priv mut declared_length : Int32
  priv mut flags : UInt32
  priv mut input_count : Int32
  priv mut stop_token_count : Int32
  priv mut stop_string_count : Int32
  priv mut declared_stop_bytes : Int32
  priv mut scope_length : Int32
  priv mut trace_length : Int32
  priv mut input_offset : Int32
  priv mut stop_token_offset : Int32
  priv mut scope_offset : Int32
  priv mut trace_offset : Int32
  priv mut phase_index : Int32
  priv mut nested_index : Int32
  priv mut byte_index : Int32
  priv mut cursor : Int32
  priv mut actual_stop_bytes : Int32
  priv mut utf8_remaining : Int32
  priv mut utf8_codepoint : Int32
  priv mut utf8_minimum : Int32
  priv mut last_work_units : Int32
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaFramedRequestWork {
  priv workspace : LunaFramedRequestWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaFramedRequestView {
  priv workspace : LunaFramedRequestWorkspace
  priv epoch : UInt64
}
