///|
pub(all) suberror OpenAIBaselineAdapterError {
  OpenAIBaselineInvalidIdentity
  OpenAIBaselineInvalidInputs
  OpenAIBaselineInputMismatch
  OpenAIBaselineProtocolIncompatible
  OpenAIBaselineInvalidLimits
  OpenAIBaselineInvalidLifecycle
  OpenAIBaselineInvalidResponseHead
  OpenAIBaselineEventTooLarge
  OpenAIBaselineInvalidSse
  OpenAIBaselineAmbiguousEvent
  OpenAIBaselineInvalidUsage
  OpenAIBaselineCollectorRejected
  OpenAIBaselineIncompleteTrial
} derive(Debug, Eq)

///|
/// Closed identity of the protocol lifecycle this package actually observes.
/// A declared request-protocol digest cannot change this semantic identity.
pub(all) enum OpenAIBaselineObservedProtocol {
  ChatCompletionsSseV1
  ResponsesSseV1
} derive(Debug, Eq)

///|
pub struct OpenAIBaselineIdentity {
  priv engine : @evidence.BenchmarkEngine
  priv revision : @evidence.BenchmarkDigest
  priv image : @evidence.BenchmarkDigest
  priv configuration : @evidence.BenchmarkDigest
}

///|
pub struct OpenAIBaselineInputs {
  priv profile : @evidence.BenchmarkProfile
  priv tokenizer : @evidence.BenchmarkDigest
  priv corpus : @evidence.BenchmarkDigest
  priv protocol : @evidence.BenchmarkDigest
  priv workload : @evidence.BenchmarkDigest
  priv request_input_tokens : FixedArray[Int64]
}

///|
pub struct OpenAIBaselineTrialContract {
  priv identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv ordinal : Int
  priv order_position : Int
  priv maximum_event_bytes : Int
  priv maximum_json_depth : Int
}

///|
const OPENAI_BASELINE_VACANT : Int = 0

///|
const OPENAI_BASELINE_SUBMITTED : Int = 1

///|
const OPENAI_BASELINE_ADMITTED : Int = 2

///|
const OPENAI_BASELINE_TERMINAL : Int = 3

///|
priv struct OpenAIBaselineLane {
  event_storage : FixedArray[Byte]
  mut event_length : Int
  mut phase : Int
  mut first_content_seen : Bool
  mut observed_tokens : Int64
  mut finish_seen : Bool
  mut usage_seen : Bool
  mut completion_tokens : Int64
}

///|
pub struct OpenAIBaselineTrialAdapter {
  priv identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv collector : @runner.BenchmarkTrialCollector
  priv lanes : FixedArray[OpenAIBaselineLane]
  priv maximum_json_depth : Int
  priv ordinal : Int
  priv order_position : Int
  priv mut submitted : Int
  priv mut finished : Bool
}

///|
pub struct OpenAIBaselineTrialEvidence {
  priv measurements : @evidence.BenchmarkSummaryEvidence
  priv source_identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv ordinal : Int
  priv order_position : Int
  priv identity : @evidence.BenchmarkEngineIdentity
  priv workload : @evidence.BenchmarkWorkload
  priv configuration : @evidence.BenchmarkDigest
  priv comparison_executable : @evidence.BenchmarkDigest
  priv tokenizer : @evidence.BenchmarkDigest
  priv corpus : @evidence.BenchmarkDigest
  priv declared_protocol : @evidence.BenchmarkDigest
  priv observed_protocol : OpenAIBaselineObservedProtocol
  priv observed_protocol_digest : @evidence.BenchmarkDigest
  priv canonical : Bytes
  priv digest : @evidence.BenchmarkDigest
}

///|
/// Comparison-wide pins not already carried by the common OpenAI input
/// declaration. This value owns no filesystem, process, network, or device.
pub struct OpenAIComparisonEnvironmentPins {
  priv hardware : @evidence.BenchmarkDigest
  priv driver : @evidence.BenchmarkDigest
  priv toolkit : @evidence.BenchmarkDigest
  priv model : @evidence.BenchmarkDigest
  priv randomization : @evidence.BenchmarkDigest
}

///|
/// A complete, protocol-homogeneous 81-trial measurement set. It deliberately
/// contains no `BenchmarkTrial` or correctness-passed authority.
pub struct OpenAIComparisonMeasurementSet {
  priv environment : @evidence.BenchmarkEnvironment
  priv engines : FixedArray[@evidence.BenchmarkEngineIdentity]
  priv workloads : FixedArray[@evidence.BenchmarkWorkload]
  priv trials : FixedArray[OpenAIBaselineTrialEvidence]
}

///|
/// Shared observation contract for the `/v1/responses` SSE lifecycle. Its
/// evidence remains structurally separate from the Chat Completions set.
pub struct OpenAIResponsesTrialContract {
  priv identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv ordinal : Int
  priv order_position : Int
  priv maximum_event_bytes : Int
  priv maximum_json_depth : Int
}

///|
const OPENAI_RESPONSES_NO_TERMINAL : Int = 0

///|
const OPENAI_RESPONSES_COMPLETED : Int = 1

///|
const OPENAI_RESPONSES_CANCELLED : Int = 2

///|
const OPENAI_RESPONSES_FAILED : Int = 3

///|
priv struct OpenAIResponsesLane {
  event_storage : FixedArray[Byte]
  mut event_length : Int
  mut phase : Int
  mut response_id : String?
  mut created_seen : Bool
  mut first_delta_seen : Bool
  mut observed_delta_events : Int64
  mut usage_seen : Bool
  mut completion_tokens : Int64
  mut terminal_kind : Int
}

///|
pub struct OpenAIResponsesTrialObserver {
  priv identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv collector : @runner.BenchmarkTrialCollector
  priv lanes : FixedArray[OpenAIResponsesLane]
  priv maximum_json_depth : Int
  priv ordinal : Int
  priv order_position : Int
  priv mut submitted : Int
  priv mut finished : Bool
  priv captured_mode : Bool
  priv mut captured_timestamp : @runner.CapturedMonotonicTimestamp?
}

///|
/// Authority-free measurements from the shared Responses SSE
/// lifecycle. This type has no conversion to `OpenAIComparisonMeasurementSet`
/// or correctness-passed benchmark authority.
pub struct OpenAIResponsesTrialEvidence {
  priv measurements : @evidence.BenchmarkSummaryEvidence
  priv source_identity : OpenAIBaselineIdentity
  priv inputs : OpenAIBaselineInputs
  priv ordinal : Int
  priv order_position : Int
  priv identity : @evidence.BenchmarkEngineIdentity
  priv workload : @evidence.BenchmarkWorkload
  priv configuration : @evidence.BenchmarkDigest
  priv comparison_executable : @evidence.BenchmarkDigest
  priv tokenizer : @evidence.BenchmarkDigest
  priv corpus : @evidence.BenchmarkDigest
  priv declared_protocol : @evidence.BenchmarkDigest
  priv observed_protocol : OpenAIBaselineObservedProtocol
  priv observed_protocol_digest : @evidence.BenchmarkDigest
  priv canonical : Bytes
  priv digest : @evidence.BenchmarkDigest
}

///|
/// A protocol-homogeneous 81-trial Responses measurement set. It deliberately
/// contains no correctness or comparison-passed authority.
pub struct OpenAIResponsesComparisonMeasurementSet {
  priv environment : @evidence.BenchmarkEnvironment
  priv engines : FixedArray[@evidence.BenchmarkEngineIdentity]
  priv workloads : FixedArray[@evidence.BenchmarkWorkload]
  priv trials : FixedArray[OpenAIResponsesTrialEvidence]
}
