///|
const LUNA_TOKENIZER_MAX_STEP_WORK_UNITS : Int = 65536

///|
const LUNA_WORKER_IDLE : Int = 0

///|
const LUNA_WORKER_INITIALIZE : Int = 1

///|
const LUNA_WORKER_SPECIAL_MATCH : Int = 2

///|
const LUNA_WORKER_SELECT_ADJACENCY : Int = 3

///|
const LUNA_WORKER_SELECT_RULE : Int = 4

///|
const LUNA_WORKER_MERGE_BATCH : Int = 5

///|
const LUNA_WORKER_COMPACT : Int = 6

///|
const LUNA_WORKER_READY : Int = 7

///|
const LUNA_REMOVED_NODE : Int = -2

///|
pub(all) enum LunaTokenizerProgress {
  LunaTokenizerPending
  LunaTokenizerReady
} derive(Debug, Eq)

///|
#valtype
pub struct LunaTokenizerStepBudget {
  priv work_units : Int
}

///|
/// Constructs the maximum number of charged transitions allowed in one
/// `progress` call or one ranged result copy. One transition is exactly one
/// input-symbol append, special-piece byte/candidate transition, adjacency
/// setup, CSR binary-search comparison/advance, merge candidate, compaction
/// step/finalization, or copied token. No transition performs a collection
/// scan, grows storage, or allocates managed memory.
pub fn LunaTokenizerStepBudget::new(
  work_units : Int,
) -> LunaTokenizerStepBudget raise TokenizerError {
  if work_units <= 0 || work_units > LUNA_TOKENIZER_MAX_STEP_WORK_UNITS {
    raise LunaInvalidStepBudget
  }
  { work_units, }
}

///|
pub fn LunaTokenizerStepBudget::work_units(
  self : LunaTokenizerStepBudget,
) -> Int {
  self.work_units
}

///|
/// Reusable startup-preallocated tokenizer owner. `begin_bytes` returns an
/// exact epoch credit; retained aliases become stale after abort or reuse.
pub struct LunaTokenizerWorker {
  priv spec : TokenizerSpec
  priv input_capacity : Int
  priv output_capacity : Int
  priv budget : LunaTokenizerStepBudget
  priv tokens : FixedArray[Int]
  priv previous : FixedArray[Int]
  priv next : FixedArray[Int]
  priv mut input : Bytes
  priv mut special_policy : SpecialTokenPolicy
  priv mut truncation : TruncationPolicy
  priv mut output_limit : Int
  priv mut phase : Int
  priv mut epoch : UInt64
  priv mut input_offset : Int
  priv mut node_count : Int
  priv mut head : Int
  priv mut tail : Int
  priv mut special_index : Int
  priv mut special_piece_offset : Int
  priv mut special_matches : Bool
  priv mut best_special_id : Int
  priv mut best_special_bytes : Int
  priv mut scan_left : Int
  priv mut lookup_low : Int
  priv mut lookup_high : Int
  priv mut selected_rank : Int
  priv mut compact_node : Int
  priv mut compact_actual : Int
  priv mut output_count : Int
  priv mut truncated : Bool
  priv mut last_work_units : Int
}

///|
#valtype
pub struct LunaTokenizerWork {
  priv worker : LunaTokenizerWorker
  priv epoch : UInt64
}
