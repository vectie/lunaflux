///|
const LUNA_INBOUND_IDLE : Int32 = 0

///|
const LUNA_INBOUND_PARSE_START : Int32 = 1

///|
const LUNA_INBOUND_TOP_KEY : Int32 = 2

///|
const LUNA_INBOUND_TOP_COLON : Int32 = 3

///|
const LUNA_INBOUND_TOP_VALUE : Int32 = 4

///|
const LUNA_INBOUND_TOP_AFTER : Int32 = 5

///|
const LUNA_INBOUND_STRING : Int32 = 6

///|
const LUNA_INBOUND_STRING_EMIT : Int32 = 7

///|
const LUNA_INBOUND_TRUE : Int32 = 8

///|
const LUNA_INBOUND_MESSAGES_START : Int32 = 9

///|
const LUNA_INBOUND_MESSAGES_ITEM : Int32 = 10

///|
const LUNA_INBOUND_MESSAGE_KEY : Int32 = 11

///|
const LUNA_INBOUND_MESSAGE_COLON : Int32 = 12

///|
const LUNA_INBOUND_MESSAGE_VALUE : Int32 = 13

///|
const LUNA_INBOUND_MESSAGE_AFTER : Int32 = 14

///|
const LUNA_INBOUND_MESSAGES_AFTER : Int32 = 15

///|
const LUNA_INBOUND_PARSE_FINAL : Int32 = 16

///|
const LUNA_INBOUND_RENDER_SEGMENT : Int32 = 17

///|
const LUNA_INBOUND_RENDER_STRING : Int32 = 18

///|
const LUNA_INBOUND_RENDER_EMIT : Int32 = 19

///|
const LUNA_INBOUND_HANDOFF_BEGIN : Int32 = 20

///|
const LUNA_INBOUND_HANDOFF_PUSH : Int32 = 21

///|
const LUNA_INBOUND_HANDOFF_PROGRESS : Int32 = 22

///|
const LUNA_INBOUND_FINAL : Int32 = 23

///|
const LUNA_INBOUND_READY : Int32 = 24

///|
const LUNA_INBOUND_VIEW : Int32 = 25

///|
const LUNA_INBOUND_FAILED : Int32 = 26

///|
const LUNA_INBOUND_NUMBER : Int32 = 27

///|
const LUNA_INBOUND_NUMBER_SCALE : Int32 = 28

///|
const LUNA_INBOUND_NUMBER_FINAL : Int32 = 29

///|
const LUNA_INBOUND_STOP_ARRAY_START : Int32 = 30

///|
const LUNA_INBOUND_STOP_ARRAY_ITEM : Int32 = 31

///|
const LUNA_INBOUND_STOP_ARRAY_AFTER : Int32 = 32

///|
const LUNA_INBOUND_TOKEN_ARRAY_START : Int32 = 33

///|
const LUNA_INBOUND_TOKEN_ARRAY_ITEM : Int32 = 34

///|
const LUNA_INBOUND_TOKEN_ARRAY_AFTER : Int32 = 35

///|
const LUNA_STRING_TOP_KEY : Int32 = 1

///|
const LUNA_STRING_MODEL : Int32 = 2

///|
const LUNA_STRING_INPUT : Int32 = 3

///|
const LUNA_STRING_MESSAGE_KEY : Int32 = 4

///|
const LUNA_STRING_ROLE : Int32 = 5

///|
const LUNA_STRING_CONTENT : Int32 = 6

///|
const LUNA_STRING_STOP : Int32 = 7

///|
pub struct LunaOpenAIInboundLimits {
  priv template_response_input : Bool
  priv max_messages : Int32
  priv required_stop_tokens : ReadOnlyArray[Int32]
}

///|
pub fn LunaOpenAIInboundLimits::new(
  max_messages~ : Int32,
  required_stop_tokens~ : ArrayView[Int32] = [],
  template_response_input~ : Bool = false,
) -> Self raise LunaOpenAICompatError {
  if max_messages <= 0 || max_messages > 64 {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  guard required_stop_tokens.length() <= 64 &&
    required_stop_tokens.all(t => t >= 0) else {
    raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
  }
  for i in 0..<required_stop_tokens.length() {
    for j in 0..<i {
      if required_stop_tokens[i] == required_stop_tokens[j] {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatLimits)
      }
    }
  }
  {
    max_messages,
    template_response_input,
    required_stop_tokens: ReadOnlyArray::from_array(
      required_stop_tokens.to_array(),
    ),
  }
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
  priv model_alias_length : Int32
  priv message_cells : FixedArray[Int32]
  priv stop_tokens : FixedArray[Int32]
  priv stop_string_cells : FixedArray[Int32]
  priv stop_string_bytes : FixedArray[Byte]
  priv handoff_workspace : @inference.LunaTextRequestHandoffStorage
  priv effective : @inference.EffectiveLimits
  priv mut requested_max_tokens : Int32
  priv mut requested_max_completion_tokens : Int32
  priv sampling : @inference.SamplingParameters
  priv sampling_seed : @inference.SamplingSeed
  priv deadline : @inference.DeadlineBudget
  priv http_views : Array[@http1.LunaHttp1View]
  priv mut normalized_body : Bytes?
  priv mut source_body_length : Int32
  priv request_ids : Array[@inference.RequestId]
  priv handoff_writes : Array[@inference.LunaTextRequestHandoffWrite]
  priv handoff_works : Array[@inference.LunaTextRequestHandoffWork]
  priv handoff_leases : Array[@inference.LunaTextRequestHandoffLease]
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut route : Int32
  priv mut body_length : Int32
  priv mut cursor : Int32
  priv mut top_seen : Int32
  priv mut message_seen : Int32
  priv mut current_key : Int32
  priv mut message_count : Int32
  priv mut current_role : Int32
  priv mut current_content_start : Int32
  priv mut current_content_length : Int32
  priv mut after_comma : Bool
  priv mut string_purpose : Int32
  priv mut string_return_phase : Int32
  priv mut string_start : Int32
  priv mut string_decoded_index : Int32
  priv mut string_candidates : Int32
  priv mut escape_state : Int32
  priv mut unicode_value : Int32
  priv mut unicode_digits : Int32
  priv mut high_surrogate : Int32
  priv mut pending_codepoint : Int32
  priv mut pending_width : Int32
  priv mut pending_index : Int32
  priv mut utf8_remaining : Int32
  priv mut utf8_codepoint : Int32
  priv mut utf8_minimum : Int32
  priv mut true_index : Int32
  priv mut streaming_response : Bool
  priv mut input_start : Int32
  priv mut input_length : Int32
  priv mut sampling_mode : Int32
  priv mut temperature : Double
  priv mut has_top_k : Bool
  priv mut top_k : Int32
  priv mut has_top_p : Bool
  priv mut top_p : Double
  priv mut sampling_seed_value : UInt64
  priv mut stop_token_count : Int32
  priv mut stop_string_count : Int32
  priv mut stop_string_byte_count : Int32
  priv mut current_stop_start : Int32
  priv mut current_stop_length : Int32
  priv mut number_return_phase : Int32
  priv mut number_target : Int32
  priv mut number_stage : Int32
  priv mut number_negative : Bool
  priv mut number_significand : UInt64
  priv mut number_fraction_digits : Int32
  priv mut number_exponent_negative : Bool
  priv mut number_exponent : Int32
  priv mut number_leading_zero : Bool
  priv mut number_value : Double
  priv mut number_scale : Int32
  priv mut render_message : Int32
  priv mut render_segment : Int32
  priv mut render_segment_index : Int32
  priv mut render_source_start : Int32
  priv mut render_source_length : Int32
  priv mut render_source_index : Int32
  priv mut prompt_length : Int32
  priv mut handoff_push_stage : Int32
  priv mut handoff_stop_index : Int32
  priv mut handoff_stop_byte : Int32
  priv mut failure_rule : Int32
  priv mut last_work_units : Int32
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
