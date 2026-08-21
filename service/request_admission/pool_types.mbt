///|
const LUNA_PREPARATION_IDLE : Int = 0

///|
const LUNA_PREPARATION_TEXT_INPUT_COPY : Int = 1

///|
const LUNA_PREPARATION_TEXT_TOKENIZE : Int = 2

///|
const LUNA_PREPARATION_TEXT_BEGIN_COPY : Int = 3

///|
const LUNA_PREPARATION_TEXT_COPY : Int = 4

///|
const LUNA_PREPARATION_OUTPUT_SETUP : Int = 5

///|
const LUNA_PREPARATION_ASSEMBLE : Int = 6

///|
const LUNA_PREPARATION_READY : Int = 7

///|
const LUNA_PREPARATION_PREPARED : Int = 8

///|
const LUNA_PREPARATION_CLAIMED : Int = 9

///|
const LUNA_PREPARATION_FAILED : Int = 10

///|
const LUNA_PREPARATION_CANCEL_REQUESTED : Int = 11

///|
const LUNA_PREPARATION_MAX_STEP_WORK_UNITS : Int = 65536

///|
const LUNA_PREPARATION_MAX_LANES : Int = 1024

///|
#valtype
pub struct LunaRequestPreparationStepBudget {
  priv work_units : Int
}

///|
pub fn LunaRequestPreparationStepBudget::new(
  work_units : Int,
) -> LunaRequestPreparationStepBudget raise RequestAdmissionError {
  if work_units <= 0 || work_units > LUNA_PREPARATION_MAX_STEP_WORK_UNITS {
    raise Invalid(LunaPreparationWork)
  }
  { work_units, }
}

///|
pub fn LunaRequestPreparationStepBudget::work_units(
  self : LunaRequestPreparationStepBudget,
) -> Int {
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
  { int_cells, byte_cells, reference_cells }
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
  LunaRequestPreparationPoolAdvanced
} derive(Debug, Eq)

///|
#valtype
pub struct LunaRequestPreparationAdmission {
  priv pool : LunaRequestPreparationPool
  priv lane : Int
  priv generation : UInt64
  priv disposition : Int
}

///|
#valtype
pub struct LunaRequestPreparationWork {
  priv pool : LunaRequestPreparationPool
  priv lane : Int
  priv generation : UInt64
}

///|
priv struct LunaRequestPreparationLane {
  tokenizer : @tokenizer.LunaTokenizerWorker
  token_storage : @inference.LunaTokenBufferStorage
  output_workspace : @incremental_output.LunaIncrementalOutputWorkspace
  mut generation : UInt64
  mut phase : Int
  mut queued : Bool
  mut receipt_at_millis : UInt64
  mut request : @inference.GenerateRequest?
  mut deadline : @inference.AdmissionDeadline?
  mut effective : @inference.EffectiveLimits?
  mut input_capacity : Int
  mut tokenizer_input_write : @tokenizer.LunaTokenizerInputWrite?
  mut tokenizer_work : @tokenizer.LunaTokenizerWork?
  mut token_write : @inference.LunaTokenBufferWrite?
  mut token_lease : @inference.LunaTokenBufferLease?
  mut output_work : @incremental_output.LunaIncrementalOutputWork?
  mut output_lease : @incremental_output.LunaIncrementalOutputLease?
  mut tokens : @inference.TokenBuffer?
  mut copy_index : Int
  mut claim : LunaPreparedRequestClaim?
  mut failure : RequestAdmissionError?
  mut last_work_units : Int
  mut total_work_units : UInt64
}

///|
pub struct LunaRequestPreparationPool {
  priv tokenizer : @tokenizer.TokenizerSpec
  priv expected_tokenizer : @tokenizer.TokenizerDigest
  priv expected_model : @spec.ModelIdentity
  priv inference : @inference.InferenceLimits
  priv clock_read : () -> UInt64 raise RequestAdmissionError
  priv step_budget : LunaRequestPreparationStepBudget
  priv work_limit : LunaRequestPreparationWorkLimit
  priv lanes : FixedArray[LunaRequestPreparationLane]
  priv free_ring : FixedArray[Int]
  priv active_ring : FixedArray[Int]
  priv mut free_head : Int
  priv mut free_count : Int
  priv mut active_head : Int
  priv mut active_count : Int
  priv mut draining : Bool
}

///|
pub fn LunaRequestPreparationPool::lane_count(
  self : LunaRequestPreparationPool,
) -> Int {
  self.lanes.length()
}

///|
pub fn LunaRequestPreparationPool::free_lane_count(
  self : LunaRequestPreparationPool,
) -> Int {
  self.free_count
}
