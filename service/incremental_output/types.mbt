///|
pub(all) enum IncrementalOutputRule {
  DecodeEnvelope
  Token
  Utf8
  Destination
  Lifecycle
} derive(Debug, Eq)

///|
pub(all) enum IncrementalOutputIssue {
  Unsupported
  Invalid
  TooSmall
  Terminal
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
pub struct IncrementalOutputStatus(Int, Int)

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
pub fn IncrementalOutputStatus::rejected() -> IncrementalOutputStatus {
  IncrementalOutputStatus(1, 0)
}

///|
priv struct StopPattern {
  bytes : Bytes
  failure : FixedArray[Int]
}

///|
const OUTPUT_ACTIVE : Int = 0

///|
const OUTPUT_STOPPED : Int = 1

///|
const OUTPUT_FINISHED : Int = 2

///|
/// One request's bounded incremental decoded-output owner. Stop text, pending
/// decoded bytes, and tokenizer authority are private and have no `Debug`.
pub struct IncrementalOutput {
  priv tokenizer : @tokenizer.TokenizerSpec
  priv special_policy : @tokenizer.DecodeSpecialPolicy
  priv patterns : Array[StopPattern]
  priv states : FixedArray[Int]
  priv next_states : FixedArray[Int]
  priv pending : FixedArray[Byte]
  priv scratch : FixedArray[Byte]
  priv mut pending_count : Int
  priv mut utf8_remaining : Int
  priv mut utf8_codepoint : Int
  priv mut utf8_minimum : Int
  priv mut utf8_seen : Int
  priv mut lifecycle : Int
}

///|
fn invalid(
  rule : IncrementalOutputRule,
  issue : IncrementalOutputIssue,
) -> IncrementalOutputError {
  Invalid(rule~, issue~)
}
