///|
const LUNA_OPENAI_COMPAT_MAX_MODEL_ALIAS_BYTES : Int = 256

///|
const LUNA_OPENAI_COMPAT_MAX_RESPONSE_PREFIX_BYTES : Int = 64

///|
const LUNA_OPENAI_COMPAT_MAX_ERROR_CODE_BYTES : Int = 64

///|
const LUNA_OPENAI_COMPAT_FIXED_OUTPUT_BYTES : UInt64 = 2048UL

///|
const LUNA_OPENAI_COMPAT_MAX_STEP_WORK_UNITS : Int = 65536

///|
const LUNA_OPENAI_COMPAT_IDLE : Int = 0

///|
const LUNA_OPENAI_COMPAT_SETUP : Int = 1

///|
const LUNA_OPENAI_COMPAT_CONFIGURE_PART : Int = 2

///|
const LUNA_OPENAI_COMPAT_LITERAL : Int = 3

///|
const LUNA_OPENAI_COMPAT_RAW : Int = 4

///|
const LUNA_OPENAI_COMPAT_LOAD_ESCAPE : Int = 5

///|
const LUNA_OPENAI_COMPAT_EMIT_ESCAPE : Int = 6

///|
const LUNA_OPENAI_COMPAT_DECIMAL_PREPARE : Int = 7

///|
const LUNA_OPENAI_COMPAT_DECIMAL_EMIT : Int = 8

///|
const LUNA_OPENAI_COMPAT_FINALIZING : Int = 9

///|
const LUNA_OPENAI_COMPAT_READY : Int = 10

///|
const LUNA_OPENAI_COMPAT_VIEW_LEASED : Int = 11

///|
const LUNA_OPENAI_COMPAT_FAILED : Int = 12

///|
const LUNA_OPENAI_PART_END : Int = 0

///|
const LUNA_OPENAI_PART_LITERAL : Int = 1

///|
const LUNA_OPENAI_PART_ID_PREFIX : Int = 2

///|
const LUNA_OPENAI_PART_SEQUENCE : Int = 3

///|
const LUNA_OPENAI_PART_MODEL : Int = 4

///|
const LUNA_OPENAI_PART_PAYLOAD : Int = 5

///|
const LUNA_OPENAI_PART_INPUT : Int = 6

///|
const LUNA_OPENAI_PART_CACHED : Int = 7

///|
const LUNA_OPENAI_PART_OUTPUT : Int = 8

///|
const LUNA_OPENAI_PART_TOTAL : Int = 9

///|
const LUNA_OPENAI_PART_FINISH : Int = 10

///|
const LUNA_OPENAI_PART_RETRYABLE : Int = 11

///|
const LUNA_OPENAI_EVENT_ACCEPTED : Int = 1

///|
const LUNA_OPENAI_EVENT_TOKEN : Int = 2

///|
const LUNA_OPENAI_EVENT_USAGE : Int = 3

///|
const LUNA_OPENAI_EVENT_COMPLETED : Int = 4

///|
const LUNA_OPENAI_EVENT_FAILED : Int = 5

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
  priv work_units : Int
}

///|
pub fn LunaOpenAICompatStepBudget::new(
  work_units : Int,
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
) -> Int {
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
  priv model_alias_length : Int
  priv response_id_prefix : FixedArray[Byte]
  priv response_id_prefix_length : Int
  priv semantic_views : Array[@luna_event.LunaEventView]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut mode : Int
  priv mut event_kind : Int
  priv mut response_sequence : UInt64
  priv mut setup_cursor : Int
  priv mut part_index : Int
  priv mut part_kind : Int
  priv mut part_cursor : Int
  priv mut source_cursor : Int
  priv mut current_byte : Byte
  priv mut escape_kind : Int
  priv mut escape_cursor : Int
  priv mut escape_length : Int
  priv mut decimal_value : UInt64
  priv mut decimal_divisor : UInt64
  priv mut token_id : Int
  priv mut position : Int
  priv mut input_tokens : Int
  priv mut cached_input_tokens : Int
  priv mut output_tokens : Int
  priv mut total_tokens : Int
  priv mut finish_reason : Int
  priv mut retryable : Bool
  priv mut payload_length : Int
  priv mut length : Int
  priv mut failure_rule : Int
  priv mut last_work_units : Int
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
