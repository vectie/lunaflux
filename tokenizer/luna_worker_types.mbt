///|
const LUNA_TOKENIZER_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_WORKER_IDLE : Int32 = 0

///|
const LUNA_WORKER_INITIALIZE : Int32 = 1

///|
const LUNA_WORKER_SPECIAL_MATCH : Int32 = 2

///|
const LUNA_WORKER_SELECT_ADJACENCY : Int32 = 3

///|
const LUNA_WORKER_SELECT_RULE : Int32 = 4

///|
const LUNA_WORKER_MERGE_BATCH : Int32 = 5

///|
const LUNA_WORKER_COMPACT : Int32 = 6

///|
const LUNA_WORKER_READY : Int32 = 7

///|
const LUNA_WORKER_INPUT_WRITE : Int32 = 8

///|
const LUNA_WORKER_SENTENCEPIECE_SCALAR : Int32 = 9

///|
const LUNA_WORKER_QWEN3_NORMALIZE : Int32 = 10

///|
const LUNA_WORKER_DIRECT_SEGMENT : Int32 = 11

///|
const LUNA_WORKER_DIRECT_LOOKUP : Int32 = 12

///|
const LUNA_WORKER_DIRECT_COMPARE : Int32 = 13

///|
const LUNA_REMOVED_NODE : Int32 = -2

///|
const LUNA_PRETOKEN_BOUNDARY : Int32 = -3

///|
pub(all) enum LunaTokenizerProgress {
  LunaTokenizerPending
  LunaTokenizerReady
} derive(Debug, Eq)

///|
/// Allocation-free scalar read of one ready Luna tokenizer result token.
#valtype
pub struct LunaTokenizerTokenStatus(Int32, Int32)

///|
pub fn LunaTokenizerTokenStatus::is_ready(
  self : LunaTokenizerTokenStatus,
) -> Bool {
  self.0 == 0
}

///|
/// Returns the token ID for a ready status and `-1` otherwise.
pub fn LunaTokenizerTokenStatus::token_id(
  self : LunaTokenizerTokenStatus,
) -> Int32 {
  if self.0 == 0 {
    self.1
  } else {
    -1
  }
}

///|
#valtype
pub struct LunaTokenizerStepBudget {
  priv work_units : Int32
}

///|
/// Constructs the maximum number of charged transitions allowed in one
/// `progress` call or one ranged result copy. One transition is exactly one
/// input-symbol append, special-piece byte/candidate transition, adjacency
/// setup, CSR binary-search comparison/advance, merge candidate, compaction
/// step/finalization, exact-pretoken byte comparison/index advance, or copied
/// token. No transition performs a collection
/// scan, grows storage, or allocates managed memory.
pub fn LunaTokenizerStepBudget::new(
  work_units : Int32,
) -> LunaTokenizerStepBudget raise TokenizerError {
  if work_units <= 0 || work_units > LUNA_TOKENIZER_MAX_STEP_WORK_UNITS {
    raise LunaInvalidStepBudget
  }
  { work_units, }
}

///|
pub fn LunaTokenizerStepBudget::work_units(
  self : LunaTokenizerStepBudget,
) -> Int32 {
  self.work_units
}

///|
/// Reusable startup-preallocated tokenizer owner. `begin_bytes` returns an
/// exact epoch credit; retained aliases become stale after abort or reuse.
pub struct LunaTokenizerWorker {
  priv spec : TokenizerSpec
  priv input_capacity : Int32
  priv node_capacity : Int32
  priv output_capacity : Int32
  priv budget : LunaTokenizerStepBudget
  priv tokens : FixedArray[Int32]
  priv previous : FixedArray[Int32]
  priv next : FixedArray[Int32]
  priv input : FixedArray[Byte]
  priv mut input_length : Int32
  priv mut written_input_bytes : Int32
  priv mut ordinary_ready : Bool
  priv mut segment_start : Bool
  priv mut post_processor_pending : Bool
  priv mut scalar_is_meta : Bool
  priv mut scalar_direct_token_id : Int32
  priv mut scalar_byte_offset : Int32
  priv mut scalar_byte_length : Int32
  priv mut scalar_source_advance : Int32
  priv mut normalized_length : Int32
  priv mut qwen_segment_end : Int32
  priv mut direct_start : Int32
  priv mut direct_end : Int32
  priv mut direct_node : Int32
  priv mut direct_byte : Int32
  priv mut direct_low : Int32
  priv mut direct_high : Int32
  priv mut direct_middle : Int32
  priv mut special_policy : SpecialTokenPolicy
  priv mut truncation : TruncationPolicy
  priv mut output_limit : Int32
  priv mut phase : Int32
  priv mut epoch : UInt64
  priv mut input_offset : Int32
  priv mut node_count : Int32
  priv mut head : Int32
  priv mut tail : Int32
  priv mut special_index : Int32
  priv mut special_piece_offset : Int32
  priv mut special_matches : Bool
  priv mut best_special_id : Int32
  priv mut best_special_bytes : Int32
  priv mut scan_left : Int32
  priv mut lookup_low : Int32
  priv mut lookup_high : Int32
  priv mut selected_rank : Int32
  priv mut compact_node : Int32
  priv mut compact_actual : Int32
  priv mut output_count : Int32
  priv mut truncated : Bool
  priv mut last_work_units : Int32
}

///|
/// Returns the exact number of startup-preallocated `Int` cells used by one
/// raw ByteLevel worker's token, previous-link, and next-link arrays.
/// Profile-polymorphic callers must use
/// `TokenizerSpec::luna_worker_required_int_cells` instead.
pub fn LunaTokenizerWorker::required_int_cells(
  input_capacity : Int32,
) -> UInt64 raise TokenizerError {
  if input_capacity <= 0 {
    raise LunaInvalidWorkerCapacity
  }
  input_capacity.to_uint64() * 3UL
}

///|
/// Conservative startup byte cells owned by one worker's input backing.
/// Unicode NFC can expand UTF-8 by at most three times under the generated
/// Unicode 17.0 canonical-decomposition tables; other profiles use the same
/// fixed envelope so profile-polymorphic pool sizing remains exact and simple.
pub fn LunaTokenizerWorker::required_byte_cells(
  input_capacity : Int32,
) -> UInt64 raise TokenizerError {
  if input_capacity <= 0 ||
    input_capacity > 1_073_741_824 / QWEN3_NFC_MAX_UTF8_EXPANSION {
    raise LunaInvalidWorkerCapacity
  }
  input_capacity.to_uint64() * QWEN3_NFC_MAX_UTF8_EXPANSION.to_uint64()
}

///|
/// Exact-generation authority for sequentially filling worker-owned input.
#valtype
pub struct LunaTokenizerInputWrite {
  priv worker : LunaTokenizerWorker
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTokenizerWork {
  priv worker : LunaTokenizerWorker
  priv epoch : UInt64
}

///|
pub extend LunaTokenizerProgress with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaTokenizerProgress with Eq::{not_equal, equal}
