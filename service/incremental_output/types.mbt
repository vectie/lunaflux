///|
/// A complete text must finish every scalar. A caller-imposed token budget may
/// cut the tokenizer's byte pieces inside the final scalar; publish only the
/// complete prefix in that case. Neither boundary accepts malformed bytes.
pub(all) enum OutputTextBoundary {
  CompleteText
  TokenLimit
}

///|
pub(all) enum IncrementalOutputRule {
  DecodeEnvelope
  Token
  Utf8
  Destination
  Lifecycle
  LunaIncrementalOutputWorkspace
  LunaIncrementalOutputEpoch
} derive(Debug, Eq)

///|
pub(all) enum IncrementalOutputIssue {
  Unsupported
  Invalid
  TooSmall
  Terminal
  Busy
  NotReady
  Stale
  Exhausted
} derive(Debug, Eq)

///|
pub(all) suberror IncrementalOutputError {
  Invalid(rule~ : IncrementalOutputRule, issue~ : IncrementalOutputIssue)
} derive(Debug, Eq)

///|
#valtype
pub struct IncrementalOutputStep(Int32, Bool)

///|
pub fn IncrementalOutputStep::copied_bytes(
  self : IncrementalOutputStep,
) -> Int32 {
  self.0
}

///|
pub fn IncrementalOutputStep::stop_matched(
  self : IncrementalOutputStep,
) -> Bool {
  self.1
}

///|
/// Scalar, allocation-free token decode status for owner-resident sessions.
#valtype
pub struct IncrementalOutputStatus(Int32, Int32, Bool)

///|
pub fn IncrementalOutputStatus::is_ready(
  self : IncrementalOutputStatus,
) -> Bool {
  self.0 == 0
}

///|
pub fn IncrementalOutputStatus::copied_bytes(
  self : IncrementalOutputStatus,
) -> Int32 {
  if self.0 != 0 {
    return 0
  }
  self.1
}

///|
pub fn IncrementalOutputStatus::stop_matched(
  self : IncrementalOutputStatus,
) -> Bool {
  self.0 == 0 && self.2
}

///|
pub fn IncrementalOutputStatus::rejected() -> IncrementalOutputStatus {
  IncrementalOutputStatus(1, 0, false)
}

///|
const OUTPUT_ACTIVE : Int32 = 0

///|
const OUTPUT_STOPPED : Int32 = 1

///|
const OUTPUT_FINISHED : Int32 = 2

///|
const LUNA_INCREMENTAL_OUTPUT_IDLE : Int32 = 0

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN : Int32 = 1

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_TEXT : Int32 = 2

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE : Int32 = 3

///|
const LUNA_INCREMENTAL_OUTPUT_READY : Int32 = 4

///|
const LUNA_INCREMENTAL_OUTPUT_LEASED : Int32 = 5

///|
const LUNA_INCREMENTAL_OUTPUT_FAILED : Int32 = 6

///|
const LUNA_INCREMENTAL_OUTPUT_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
#valtype
pub(all) enum LunaIncrementalOutputProgress {
  LunaIncrementalOutputPending
  LunaIncrementalOutputReady
} derive(Debug, Eq)

///|
#valtype
pub struct LunaIncrementalOutputStepBudget {
  priv work_units : Int32
}

///|
pub struct LunaIncrementalOutputWorkspace {
  priv maximum_decoded_piece_bytes : Int32
  priv tokenizer : @tokenizer.TokenizerSpec
  priv inference : @inference.InferenceLimits
  priv budget : LunaIncrementalOutputStepBudget
  priv stop_string_byte_capacity : Int32
  priv pattern_bytes : FixedArray[Byte]
  priv failure : FixedArray[Int32]
  priv pattern_lengths : FixedArray[Int32]
  priv states : FixedArray[Int32]
  priv next_states : FixedArray[Int32]
  priv pending : FixedArray[Byte]
  priv scratch : FixedArray[Byte]
  priv mut setup_stops : @inference.StopConditions?
  priv setup_semantics : Array[@inference.LunaRequestSemanticView]
  priv mut special_policy : @tokenizer.DecodeSpecialPolicy
  priv mut phase : Int32
  priv mut epoch : UInt64
  priv mut stop_count : Int32
  priv mut setup_pattern : Int32
  priv mut setup_unit_offset : Int32
  priv mut setup_emit_index : Int32
  priv mut setup_emit_length : Int32
  priv mut setup_codepoint : Int32
  priv mut setup_failure_index : Int32
  priv mut setup_failure_matched : Int32
  priv mut longest_pattern : Int32
  priv mut last_work_units : Int32
  priv mut setup_error_rule : IncrementalOutputRule
  priv mut setup_error_issue : IncrementalOutputIssue
  priv mut pending_count : Int32
  priv mut decoder_at_start : Bool
  priv mut utf8_remaining : Int32
  priv mut utf8_codepoint : Int32
  priv mut utf8_minimum : Int32
  priv mut utf8_seen : Int32
  priv mut lifecycle : Int32
}

///|
#valtype
pub struct LunaIncrementalOutputWork {
  priv workspace : LunaIncrementalOutputWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaIncrementalOutputLease {
  priv workspace : LunaIncrementalOutputWorkspace
  priv epoch : UInt64
}

///|
/// Synchronous compatibility owner backed by one Luna workspace lease.
pub struct IncrementalOutput {
  priv lease : LunaIncrementalOutputLease
}

///|
fn invalid(
  rule : IncrementalOutputRule,
  issue : IncrementalOutputIssue,
) -> IncrementalOutputError {
  Invalid(rule~, issue~)
}
