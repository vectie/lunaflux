///|
const LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES : Int32 = 256

///|
const LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES : Int32 = 64

///|
const LUNA_OPENAI_COMPAT_MAX_ERROR_CODE_BYTES : Int32 = 64

///|
const LUNA_OPENAI_COMPAT_FIXED_OUTPUT_BYTES : UInt64 = 2048UL

///|
const LUNA_OPENAI_COMPAT_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_OPENAI_COMPAT_IDLE : Int32 = 0

///|
const LUNA_OPENAI_COMPAT_SETUP : Int32 = 1

///|
const LUNA_OPENAI_COMPAT_CONFIGURE_PART : Int32 = 2

///|
const LUNA_OPENAI_COMPAT_LITERAL : Int32 = 3

///|
const LUNA_OPENAI_COMPAT_RAW : Int32 = 4

///|
const LUNA_OPENAI_COMPAT_LOAD_ESCAPE : Int32 = 5

///|
const LUNA_OPENAI_COMPAT_EMIT_ESCAPE : Int32 = 6

///|
const LUNA_OPENAI_COMPAT_DECIMAL_PREPARE : Int32 = 7

///|
const LUNA_OPENAI_COMPAT_DECIMAL_EMIT : Int32 = 8

///|
const LUNA_OPENAI_COMPAT_FINALIZING : Int32 = 9

///|
const LUNA_OPENAI_COMPAT_READY : Int32 = 10

///|
const LUNA_OPENAI_COMPAT_VIEW_LEASED : Int32 = 11

///|
const LUNA_OPENAI_COMPAT_FAILED : Int32 = 12

///|
const LUNA_OPENAI_COMPAT_BUFFER_PAYLOAD : Int32 = 13

///|
const LUNA_OPENAI_COMPAT_FILTER_PAYLOAD : Int32 = 14

///|
const LUNA_OPENAI_PART_END : Int32 = 0

///|
const LUNA_OPENAI_PART_LITERAL : Int32 = 1

///|
const LUNA_OPENAI_PART_ID_PREFIX : Int32 = 2

///|
const LUNA_OPENAI_PART_SEQUENCE : Int32 = 3

///|
const LUNA_OPENAI_PART_MODEL : Int32 = 4

///|
const LUNA_OPENAI_PART_PAYLOAD : Int32 = 5

///|
const LUNA_OPENAI_PART_INPUT : Int32 = 6

///|
const LUNA_OPENAI_PART_CACHED : Int32 = 7

///|
const LUNA_OPENAI_PART_OUTPUT : Int32 = 8

///|
const LUNA_OPENAI_PART_TOTAL : Int32 = 9

///|
const LUNA_OPENAI_PART_FINISH : Int32 = 10

///|
const LUNA_OPENAI_PART_RETRYABLE : Int32 = 11

///|
const LUNA_OPENAI_EVENT_ACCEPTED : Int32 = 1

///|
const LUNA_OPENAI_EVENT_TOKEN : Int32 = 2

///|
const LUNA_OPENAI_EVENT_USAGE : Int32 = 3

///|
const LUNA_OPENAI_EVENT_COMPLETED : Int32 = 4

///|
const LUNA_OPENAI_EVENT_FAILED : Int32 = 5

///|
pub(all) enum LunaOpenAICompatRule {
  LunaOpenAICompatLimits
  LunaOpenAICompatWorkUnits
  LunaOpenAICompatModelAlias
  LunaOpenAICompatResponseId
  LunaOpenAICompatLifecycle
  LunaOpenAICompatEpoch
  LunaOpenAICompatSemanticEvent
  LunaOpenAICompatScalar
  LunaOpenAICompatPayload
  LunaOpenAICompatCapacity
  LunaOpenAICompatDestination
  LunaOpenAICompatJson
  LunaOpenAICompatTemplate
  LunaOpenAICompatHttp
  LunaOpenAICompatFrame
} derive(Debug, Eq)

///|
pub(all) suberror LunaOpenAICompatError {
  LunaOpenAICompatInvalid(LunaOpenAICompatRule)
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOpenAICompatMode {
  LunaOpenAIResponses
  LunaOpenAIChatCompletions
} derive(Debug, Eq)

///|
#valtype
pub struct LunaOpenAICompatStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaOpenAICompatStepBudget::new(
  work_units : Int32,
) -> LunaOpenAICompatStepBudget raise LunaOpenAICompatError {
  let budget = { work_units, }
  budget.validate()
  budget
}

///|
fn LunaOpenAICompatStepBudget::validate(
  self : LunaOpenAICompatStepBudget,
) -> Unit raise LunaOpenAICompatError {
  if self.work_units <= 0 ||
    self.work_units > LUNA_OPENAI_COMPAT_MAX_STEP_WORK_UNITS {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatWorkUnits)
  }
}

///|
pub fn LunaOpenAICompatStepBudget::work_units(
  self : LunaOpenAICompatStepBudget,
) -> Int32 {
  self.work_units
}

///|
#valtype
pub(all) enum LunaOpenAICompatWorkState {
  LunaOpenAICompatPending
  LunaOpenAICompatReady
  LunaOpenAICompatFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOpenAICompatProgress {
  LunaOpenAICompatIdle
  LunaOpenAICompatAdvanced
} derive(Debug, Eq)

///|
pub struct LunaOpenAICompatWorkspace {
  priv inference : @inference.InferenceLimits
  priv budget : LunaOpenAICompatStepBudget
  priv storage : FixedArray[Byte]
  priv model_alias : FixedArray[Byte]
  priv model_alias_length : Int32
  priv response_id_prefix : FixedArray[Byte]
  priv response_id_prefix_length : Int32
  priv semantic_views : Array[@luna_event.LunaEventView]
  priv output_framing : LunaOpenAIOutputFraming?
  priv mut framing_matched : Int32
  priv mut answer_started : Bool
  priv mut payload_offset : Int32
  priv buffered_text : FixedArray[Byte]
  priv mut buffered_sequence : UInt64
  priv mut buffered_mode : Int32
  priv mut buffered_length : Int32
  priv mut buffering : Bool
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut mode : Int32
  priv mut event_kind : Int32
  priv mut response_sequence : UInt64
  priv mut setup_cursor : Int32
  priv mut part_index : Int32
  priv mut part_kind : Int32
  priv mut part_cursor : Int32
  priv mut source_cursor : Int32
  priv mut current_byte : Byte
  priv mut escape_kind : Int32
  priv mut escape_cursor : Int32
  priv mut escape_length : Int32
  priv mut decimal_value : UInt64
  priv mut decimal_divisor : UInt64
  priv mut token_id : Int32
  priv mut position : Int32
  priv mut input_tokens : Int32
  priv mut cached_input_tokens : Int32
  priv mut output_tokens : Int32
  priv mut total_tokens : Int32
  priv mut finish_reason : Int32
  priv mut retryable : Bool
  priv mut payload_length : Int32
  priv mut length : Int32
  priv mut failure_rule : Int32
  priv mut last_work_units : Int32
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaOpenAICompatWork {
  priv workspace : LunaOpenAICompatWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaOpenAICompatView {
  priv workspace : LunaOpenAICompatWorkspace
  priv epoch : UInt64
}
