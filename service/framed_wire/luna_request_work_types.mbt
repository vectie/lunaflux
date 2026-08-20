///|
const LUNA_FRAMED_REQUEST_IDLE : Int = 0

///|
const LUNA_FRAMED_REQUEST_RECEIVING_PREFIX : Int = 1

///|
const LUNA_FRAMED_REQUEST_VALIDATING_PREFIX : Int = 2

///|
const LUNA_FRAMED_REQUEST_RECEIVING_BODY : Int = 3

///|
const LUNA_FRAMED_REQUEST_CHECKSUM : Int = 4

///|
const LUNA_FRAMED_REQUEST_HEADER : Int = 5

///|
const LUNA_FRAMED_REQUEST_DIGEST : Int = 6

///|
const LUNA_FRAMED_REQUEST_LAYOUT : Int = 7

///|
const LUNA_FRAMED_REQUEST_INPUT : Int = 8

///|
const LUNA_FRAMED_REQUEST_STOP_TOKENS : Int = 9

///|
const LUNA_FRAMED_REQUEST_STOP_TOKEN_DUPLICATES : Int = 10

///|
const LUNA_FRAMED_REQUEST_STOP_STRINGS : Int = 11

///|
const LUNA_FRAMED_REQUEST_STOP_STRING_DUPLICATES : Int = 12

///|
const LUNA_FRAMED_REQUEST_SCOPE : Int = 13

///|
const LUNA_FRAMED_REQUEST_TRACE : Int = 14

///|
const LUNA_FRAMED_REQUEST_READY : Int = 15

///|
const LUNA_FRAMED_REQUEST_VIEW_LEASED : Int = 16

///|
const LUNA_FRAMED_REQUEST_FAILED : Int = 17

///|
const LUNA_FRAMED_REQUEST_MAX_STEP_WORK_UNITS : Int = 65536

///|
#valtype
pub struct LunaFramedRequestStepBudget {
  priv work_units : Int
}

///|
pub fn LunaFramedRequestStepBudget::new(
  work_units : Int,
) -> LunaFramedRequestStepBudget raise FramedWireError {
  if work_units <= 0 || work_units > LUNA_FRAMED_REQUEST_MAX_STEP_WORK_UNITS {
    raise InvalidLimit(LunaFramedRequestWorkUnits)
  }
  { work_units, }
}

///|
pub fn LunaFramedRequestStepBudget::work_units(
  self : LunaFramedRequestStepBudget,
) -> Int {
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
  priv stop_string_offsets : FixedArray[Int]
  priv stop_string_lengths : FixedArray[Int]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut failure : FramedWireError?
  priv mut length : Int
  priv mut declared_length : Int
  priv mut flags : UInt
  priv mut input_count : Int
  priv mut stop_token_count : Int
  priv mut stop_string_count : Int
  priv mut declared_stop_bytes : Int
  priv mut scope_length : Int
  priv mut trace_length : Int
  priv mut input_offset : Int
  priv mut stop_token_offset : Int
  priv mut scope_offset : Int
  priv mut trace_offset : Int
  priv mut phase_index : Int
  priv mut nested_index : Int
  priv mut byte_index : Int
  priv mut cursor : Int
  priv mut actual_stop_bytes : Int
  priv mut checksum : UInt
  priv mut utf8_remaining : Int
  priv mut utf8_codepoint : Int
  priv mut utf8_minimum : Int
  priv mut last_work_units : Int
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
