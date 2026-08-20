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
pub struct IncrementalOutputStep(Int, Bool)

///|
pub fn IncrementalOutputStep::copied_bytes(self : IncrementalOutputStep) -> Int {
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
pub struct IncrementalOutputStatus(Int, Int, Bool)

///|
pub fn IncrementalOutputStatus::is_ready(
  self : IncrementalOutputStatus,
) -> Bool {
  self.0 == 0
}

///|
pub fn IncrementalOutputStatus::copied_bytes(
  self : IncrementalOutputStatus,
) -> Int {
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
const OUTPUT_ACTIVE : Int = 0

///|
const OUTPUT_STOPPED : Int = 1

///|
const OUTPUT_FINISHED : Int = 2

///|
const LUNA_INCREMENTAL_OUTPUT_IDLE : Int = 0

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_PATTERN : Int = 1

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_TEXT : Int = 2

///|
const LUNA_INCREMENTAL_OUTPUT_SETUP_FAILURE : Int = 3

///|
const LUNA_INCREMENTAL_OUTPUT_READY : Int = 4

///|
const LUNA_INCREMENTAL_OUTPUT_LEASED : Int = 5

///|
const LUNA_INCREMENTAL_OUTPUT_MAX_STEP_WORK_UNITS : Int = 65536

///|
#valtype
pub(all) enum LunaIncrementalOutputProgress {
  LunaIncrementalOutputPending
  LunaIncrementalOutputReady
} derive(Debug, Eq)

///|
#valtype
pub struct LunaIncrementalOutputStepBudget {
  priv work_units : Int
}

///|
pub struct LunaIncrementalOutputWorkspace {
  priv tokenizer : @tokenizer.TokenizerSpec
  priv inference : @inference.InferenceLimits
  priv budget : LunaIncrementalOutputStepBudget
  priv stop_string_byte_capacity : Int
  priv pattern_bytes : FixedArray[Byte]
  priv failure : FixedArray[Int]
  priv pattern_lengths : FixedArray[Int]
  priv states : FixedArray[Int]
  priv next_states : FixedArray[Int]
  priv pending : FixedArray[Byte]
  priv scratch : FixedArray[Byte]
  priv mut setup_stops : @inference.StopConditions?
  priv mut special_policy : @tokenizer.DecodeSpecialPolicy
  priv mut phase : Int
  priv mut epoch : UInt64
  priv mut stop_count : Int
  priv mut setup_pattern : Int
  priv mut setup_unit_offset : Int
  priv mut setup_emit_index : Int
  priv mut setup_emit_length : Int
  priv mut setup_codepoint : Int
  priv mut setup_failure_index : Int
  priv mut setup_failure_matched : Int
  priv mut longest_pattern : Int
  priv mut last_work_units : Int
  priv mut pending_count : Int
  priv mut utf8_remaining : Int
  priv mut utf8_codepoint : Int
  priv mut utf8_minimum : Int
  priv mut utf8_seen : Int
  priv mut lifecycle : Int
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
