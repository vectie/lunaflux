///|
const LUNA_PREPARATION_IDLE : Int32 = 0

///|
const LUNA_PREPARATION_TEXT_INPUT_COPY : Int32 = 1

///|
const LUNA_PREPARATION_TEXT_TOKENIZE : Int32 = 2

///|
const LUNA_PREPARATION_TEXT_BEGIN_COPY : Int32 = 3

///|
const LUNA_PREPARATION_TEXT_COPY : Int32 = 4

///|
const LUNA_PREPARATION_SEMANTIC_CACHE_MEASURE : Int32 = 5

///|
const LUNA_PREPARATION_SEMANTIC_BEGIN : Int32 = 6

///|
const LUNA_PREPARATION_SEMANTIC_TOKEN_COPY : Int32 = 7

///|
const LUNA_PREPARATION_SEMANTIC_STRING_MEASURE : Int32 = 8

///|
const LUNA_PREPARATION_SEMANTIC_STRING_HEADER : Int32 = 9

///|
const LUNA_PREPARATION_SEMANTIC_STRING_EMIT : Int32 = 10

///|
const LUNA_PREPARATION_SEMANTIC_CACHE_EMIT : Int32 = 11

///|
const LUNA_PREPARATION_SEMANTIC_FINISH : Int32 = 12

///|
const LUNA_PREPARATION_SEMANTIC_VALIDATE : Int32 = 13

///|
const LUNA_PREPARATION_SEMANTIC_TAKE : Int32 = 14

///|
const LUNA_PREPARATION_OUTPUT_SETUP : Int32 = 15

///|
const LUNA_PREPARATION_ASSEMBLE : Int32 = 16

///|
const LUNA_PREPARATION_READY : Int32 = 17

///|
const LUNA_PREPARATION_PREPARED : Int32 = 18

///|
const LUNA_PREPARATION_CLAIMED : Int32 = 19

///|
const LUNA_PREPARATION_FAILED : Int32 = 20

///|
const LUNA_PREPARATION_CANCEL_REQUESTED : Int32 = 21

///|
const LUNA_PREPARATION_FRAMED_RECEIVING : Int32 = 22

///|
const LUNA_PREPARATION_FRAMED_SCANNING : Int32 = 23

///|
const LUNA_PREPARATION_FRAMED_TAKE_VIEW : Int32 = 24

///|
const LUNA_PREPARATION_FRAMED_BIND_MODEL : Int32 = 25

///|
const LUNA_PREPARATION_FRAMED_BIND_SCALARS : Int32 = 26

///|
const LUNA_PREPARATION_FRAMED_INPUT_BEGIN : Int32 = 27

///|
const LUNA_PREPARATION_FRAMED_INPUT_COPY : Int32 = 28

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_BEGIN : Int32 = 29

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_TOKEN_COPY : Int32 = 30

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_STRING_HEADER : Int32 = 31

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_STRING_COPY : Int32 = 32

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_CACHE_COPY : Int32 = 33

///|
const LUNA_PREPARATION_FRAMED_SEMANTIC_FINISH : Int32 = 34

///|
const LUNA_PREPARATION_FRAMED_RELEASE_VIEW : Int32 = 35

///|
const LUNA_PREPARATION_TYPED_INPUT_BEGIN : Int32 = 36

///|
const LUNA_PREPARATION_TYPED_INPUT_COPY : Int32 = 37

///|
const LUNA_PREPARATION_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_PREPARATION_MAX_LANES : Int32 = 1024

///|
#valtype
pub struct LunaRequestPreparationStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaRequestPreparationStepBudget::new(
  work_units : Int32,
) -> LunaRequestPreparationStepBudget raise RequestAdmissionError {
  if work_units <= 0 || work_units > LUNA_PREPARATION_MAX_STEP_WORK_UNITS {
    raise Invalid(LunaPreparationWork)
  }
  { work_units, }
}

///|
pub fn LunaRequestPreparationStepBudget::work_units(
  self : LunaRequestPreparationStepBudget,
) -> Int32 {
  self.work_units
}

///|
#valtype
pub struct LunaRequestPreparationWorkLimit {
  priv work_units : UInt64
}

///|
pub fn LunaRequestPreparationWorkLimit::new(
  work_units : UInt64,
) -> LunaRequestPreparationWorkLimit raise RequestAdmissionError {
  if work_units == 0UL {
    raise Invalid(LunaPreparationWork)
  }
  { work_units, }
}

///|
pub fn LunaRequestPreparationWorkLimit::work_units(
  self : LunaRequestPreparationWorkLimit,
) -> UInt64 {
  self.work_units
}

///|
/// Logical startup-preallocated collection-cell ceilings. These are not a
/// claim about managed-object headers or allocator metadata.
pub struct LunaRequestPreparationStorageBudget {
  priv int_cells : UInt64
  priv byte_cells : UInt64
  priv reference_cells : UInt64
}

///|
pub fn LunaRequestPreparationStorageBudget::new(
  int_cells~ : UInt64,
  byte_cells~ : UInt64,
  reference_cells~ : UInt64,
) -> LunaRequestPreparationStorageBudget raise RequestAdmissionError {
  if int_cells == 0UL || byte_cells == 0UL || reference_cells == 0UL {
    raise Invalid(LunaPreparationStorage)
  }
  { int_cells, byte_cells, reference_cells, }
}

///|
#valtype
pub(all) enum LunaRequestPreparationAdmissionKind {
  LunaRequestPreparationAdmitted
  LunaRequestPreparationSaturated
  LunaRequestPreparationDraining
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaRequestPreparationWorkState {
  LunaRequestPreparationPending
  LunaRequestPreparationReady
  LunaRequestPreparationFailed
  LunaRequestPreparationCancelRequested
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaRequestPreparationPoolProgress {
  LunaRequestPreparationPoolIdle
  LunaRequestPreparationPoolAwaitingInput
  LunaRequestPreparationPoolAdvanced
} derive(Debug, Eq)

///|
#valtype
pub struct LunaRequestPreparationAdmission {
  priv pool : LunaRequestPreparationPool
  priv lane : Int32
  priv generation : UInt64
  priv disposition : Int32
  priv consumed : Int32
}

///|
#valtype
pub struct LunaRequestPreparationWork {
  priv pool : LunaRequestPreparationPool
  priv lane : Int32
  priv generation : UInt64
}

///|
priv struct LunaRequestPreparationLane {
  tokenizer : @tokenizer.LunaTokenizerWorker
  token_storage : @inference.LunaTokenBufferStorage
  semantic_storage : @inference.LunaRequestSemanticStorage
  output_workspace : @incremental_output.LunaIncrementalOutputWorkspace
  framed_workspace : @framed_wire.LunaFramedRequestWorkspace
  framed_work : Array[@framed_wire.LunaFramedRequestWork]
  framed_view : Array[@framed_wire.LunaFramedRequestView]
  typed_handoff : Array[@inference.LunaTextRequestHandoffClaim]
  mut generation : UInt64
  mut phase : Int32
  mut queued : Bool
  mut receipt_at_millis : UInt64
  mut request : @inference.GenerateRequest?
  mut deadline : @inference.AdmissionDeadline?
  mut hard_deadline : @inference.AdmissionDeadline?
  mut effective : @inference.EffectiveLimits?
  mut input_capacity : Int32
  mut tokenizer_input_write : @tokenizer.LunaTokenizerInputWrite?
  mut tokenizer_work : @tokenizer.LunaTokenizerWork?
  mut token_write : @inference.LunaTokenBufferWrite?
  mut token_lease : @inference.LunaTokenBufferLease?
  semantic_write : Array[@inference.LunaRequestSemanticWrite]
  semantic_work : Array[@inference.LunaRequestSemanticWork]
  semantic_lease : Array[@inference.LunaRequestSemanticLease]
  mut output_work : @incremental_output.LunaIncrementalOutputWork?
  mut output_lease : @incremental_output.LunaIncrementalOutputLease?
  mut tokens : @inference.TokenBuffer?
  mut copy_index : Int32
  mut semantic_item_index : Int32
  mut semantic_unit_offset : Int32
  mut semantic_byte_length : Int32
  mut semantic_codepoint : Int32
  mut semantic_codepoint_width : Int32
  mut semantic_codepoint_byte_index : Int32
  mut framed_direct : Bool
  mut typed_direct : Bool
  mut typed_request_id : @inference.RequestId
  mut typed_sampling : @inference.LunaValidatedSampling
  mut typed_sampling_seed : @inference.SamplingSeed
  mut typed_stream_preference : @inference.StreamPreference
  mut framed_receipt_complete : Bool
  mut framed_model_index : Int32
  mut framed_input_kind : Int32
  mut framed_input_count : Int32
  mut framed_stop_token_count : Int32
  mut framed_stop_string_count : Int32
  mut framed_cache_scope_length : Int32
  mut framed_request_id : UInt64
  mut framed_sampling_mode : Int32
  mut framed_temperature : Double
  mut framed_has_top_k : Bool
  mut framed_top_k : Int32
  mut framed_has_top_p : Bool
  mut framed_top_p : Double
  mut framed_sampling_seed : UInt64
  mut framed_streaming : Bool
  mut framed_token_ids_only : Bool
  mut trusted_receipt_bound : Bool
  mut trusted_deadline_millis : UInt64
  mut claim : LunaPreparedRequestClaim?
  mut failure : RequestAdmissionError?
  mut last_work_units : Int32
  mut total_work_units : UInt64
}

///|
pub struct LunaRequestPreparationPool {
  priv input_special_policy : @tokenizer.SpecialTokenPolicy
  priv tokenizer : @tokenizer.TokenizerSpec
  priv expected_tokenizer : @tokenizer.TokenizerDigest
  priv expected_model : @spec.ModelIdentity
  priv inference : @inference.InferenceLimits
  priv clock_read : () -> UInt64 raise RequestAdmissionError
  priv step_budget : LunaRequestPreparationStepBudget
  priv work_limit : LunaRequestPreparationWorkLimit
  priv lanes : FixedArray[LunaRequestPreparationLane]
  priv free_ring : FixedArray[Int32]
  priv active_ring : FixedArray[Int32]
  priv mut free_head : Int32
  priv mut free_count : Int32
  priv mut active_head : Int32
  priv mut active_count : Int32
  priv mut draining : Bool
}

///|
pub fn LunaRequestPreparationPool::lane_count(
  self : LunaRequestPreparationPool,
) -> Int32 {
  self.lanes.length()
}

///|
pub fn LunaRequestPreparationPool::free_lane_count(
  self : LunaRequestPreparationPool,
) -> Int32 {
  self.free_count
}
