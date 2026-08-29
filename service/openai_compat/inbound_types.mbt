///|
const LUNA_INBOUND_IDLE : Int = 0

///|
const LUNA_INBOUND_PARSE_START : Int = 1

///|
const LUNA_INBOUND_TOP_KEY : Int = 2

///|
const LUNA_INBOUND_TOP_COLON : Int = 3

///|
const LUNA_INBOUND_TOP_VALUE : Int = 4

///|
const LUNA_INBOUND_TOP_AFTER : Int = 5

///|
const LUNA_INBOUND_STRING : Int = 6

///|
const LUNA_INBOUND_STRING_EMIT : Int = 7

///|
const LUNA_INBOUND_TRUE : Int = 8

///|
const LUNA_INBOUND_MESSAGES_START : Int = 9

///|
const LUNA_INBOUND_MESSAGES_ITEM : Int = 10

///|
const LUNA_INBOUND_MESSAGE_KEY : Int = 11

///|
const LUNA_INBOUND_MESSAGE_COLON : Int = 12

///|
const LUNA_INBOUND_MESSAGE_VALUE : Int = 13

///|
const LUNA_INBOUND_MESSAGE_AFTER : Int = 14

///|
const LUNA_INBOUND_MESSAGES_AFTER : Int = 15

///|
const LUNA_INBOUND_PARSE_FINAL : Int = 16

///|
const LUNA_INBOUND_RENDER_SEGMENT : Int = 17

///|
const LUNA_INBOUND_RENDER_STRING : Int = 18

///|
const LUNA_INBOUND_RENDER_EMIT : Int = 19

///|
const LUNA_INBOUND_HANDOFF_BEGIN : Int = 20

///|
const LUNA_INBOUND_HANDOFF_PUSH : Int = 21

///|
const LUNA_INBOUND_HANDOFF_PROGRESS : Int = 22

///|
const LUNA_INBOUND_FINAL : Int = 23

///|
const LUNA_INBOUND_READY : Int = 24

///|
const LUNA_INBOUND_VIEW : Int = 25

///|
const LUNA_INBOUND_FAILED : Int = 26

///|
const LUNA_INBOUND_NUMBER : Int = 27

///|
const LUNA_INBOUND_NUMBER_SCALE : Int = 28

///|
const LUNA_INBOUND_NUMBER_FINAL : Int = 29

///|
const LUNA_INBOUND_STOP_ARRAY_START : Int = 30

///|
const LUNA_INBOUND_STOP_ARRAY_ITEM : Int = 31

///|
const LUNA_INBOUND_STOP_ARRAY_AFTER : Int = 32

///|
const LUNA_INBOUND_TOKEN_ARRAY_START : Int = 33

///|
const LUNA_INBOUND_TOKEN_ARRAY_ITEM : Int = 34

///|
const LUNA_INBOUND_TOKEN_ARRAY_AFTER : Int = 35

///|
const LUNA_STRING_TOP_KEY : Int = 1

///|
const LUNA_STRING_MODEL : Int = 2

///|
const LUNA_STRING_INPUT : Int = 3

///|
const LUNA_STRING_MESSAGE_KEY : Int = 4

///|
const LUNA_STRING_ROLE : Int = 5

///|
const LUNA_STRING_CONTENT : Int = 6

///|
const LUNA_STRING_STOP : Int = 7

///|
pub struct LunaOpenAIInboundLimits {
  priv max_messages : Int
}

///|
pub fn LunaOpenAIInboundLimits::new(
  max_messages~ : Int,
) -> Self raise LunaOpenAICompatError {
  if max_messages <= 0 || max_messages > 64 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  { max_messages, }
}

///|
#valtype
pub(all) enum LunaOpenAIInboundWorkState {
  LunaOpenAIInboundPending
  LunaOpenAIInboundReady
  LunaOpenAIInboundFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaOpenAIInboundProgress {
  LunaOpenAIInboundIdle
  LunaOpenAIInboundAdvanced
} derive(Debug, Eq)

///|
pub struct LunaOpenAIInboundWorkspace {
  priv inference : @inference.InferenceLimits
  priv limits : LunaOpenAIInboundLimits
  priv budget : LunaOpenAICompatStepBudget
  priv template : LunaOpenAIChatTemplate
  priv model_alias : FixedArray[Byte]
  priv model_alias_length : Int
  priv message_cells : FixedArray[Int]
  priv stop_tokens : FixedArray[Int]
  priv stop_string_cells : FixedArray[Int]
  priv stop_string_bytes : FixedArray[Byte]
  priv handoff_workspace : @inference.LunaTextRequestHandoffStorage
  priv effective : @inference.EffectiveLimits
  priv sampling : @inference.SamplingParameters
  priv sampling_seed : @inference.SamplingSeed
  priv deadline : @inference.DeadlineBudget
  priv http_views : Array[@http1.LunaHttp1View]
  priv request_ids : Array[@inference.RequestId]
  priv handoff_writes : Array[@inference.LunaTextRequestHandoffWrite]
  priv handoff_works : Array[@inference.LunaTextRequestHandoffWork]
  priv handoff_leases : Array[@inference.LunaTextRequestHandoffLease]
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut route : Int
  priv mut body_length : Int
  priv mut cursor : Int
  priv mut top_seen : Int
  priv mut message_seen : Int
  priv mut current_key : Int
  priv mut message_count : Int
  priv mut current_role : Int
  priv mut current_content_start : Int
  priv mut current_content_length : Int
  priv mut after_comma : Bool
  priv mut string_purpose : Int
  priv mut string_return_phase : Int
  priv mut string_start : Int
  priv mut string_decoded_index : Int
  priv mut string_candidates : Int
  priv mut escape_state : Int
  priv mut unicode_value : Int
  priv mut unicode_digits : Int
  priv mut high_surrogate : Int
  priv mut pending_codepoint : Int
  priv mut pending_width : Int
  priv mut pending_index : Int
  priv mut utf8_remaining : Int
  priv mut utf8_codepoint : Int
  priv mut utf8_minimum : Int
  priv mut true_index : Int
  priv mut input_start : Int
  priv mut input_length : Int
  priv mut sampling_mode : Int
  priv mut temperature : Double
  priv mut has_top_k : Bool
  priv mut top_k : Int
  priv mut has_top_p : Bool
  priv mut top_p : Double
  priv mut sampling_seed_value : UInt64
  priv mut stop_token_count : Int
  priv mut stop_string_count : Int
  priv mut stop_string_byte_count : Int
  priv mut current_stop_start : Int
  priv mut current_stop_length : Int
  priv mut number_return_phase : Int
  priv mut number_target : Int
  priv mut number_stage : Int
  priv mut number_negative : Bool
  priv mut number_significand : UInt64
  priv mut number_fraction_digits : Int
  priv mut number_exponent_negative : Bool
  priv mut number_exponent : Int
  priv mut number_leading_zero : Bool
  priv mut number_value : Double
  priv mut number_scale : Int
  priv mut render_message : Int
  priv mut render_segment : Int
  priv mut render_segment_index : Int
  priv mut render_source_start : Int
  priv mut render_source_length : Int
  priv mut render_source_index : Int
  priv mut prompt_length : Int
  priv mut handoff_push_stage : Int
  priv mut handoff_stop_index : Int
  priv mut handoff_stop_byte : Int
  priv mut failure_rule : Int
  priv mut last_work_units : Int
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaOpenAIInboundWork {
  priv workspace : LunaOpenAIInboundWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaOpenAIInboundView {
  priv workspace : LunaOpenAIInboundWorkspace
  priv epoch : UInt64
}
