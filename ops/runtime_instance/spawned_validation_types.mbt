///|
/// Fail-closed stages for the bounded spawned-worker correctness observation.
pub enum RuntimeSpawnedValidationRule {
  SpawnedValidationContract
  SpawnedValidationOwner
  SpawnedValidationIngress
  SpawnedValidationRejected
  SpawnedValidationObservation
  SpawnedValidationEvent
  SpawnedValidationPlan
  SpawnedValidationKv
  SpawnedValidationCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedValidationRule::code(
  self : RuntimeSpawnedValidationRule,
) -> String {
  match self {
    SpawnedValidationContract => "contract"
    SpawnedValidationOwner => "owner"
    SpawnedValidationIngress => "ingress"
    SpawnedValidationRejected => "rejected"
    SpawnedValidationObservation => "observation"
    SpawnedValidationEvent => "event"
    SpawnedValidationPlan => "plan"
    SpawnedValidationKv => "kv"
    SpawnedValidationCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedValidationError {
  RuntimeSpawnedValidationFailed(RuntimeSpawnedValidationRule)
} derive(Debug, Eq)

///|
/// Root-free sampling identity expected by a bounded physical qualification.
pub(all) enum RuntimeSpawnedSamplingExpectation {
  AnySpawnedSamplingRuntime
  HostSamplingRuntimeExpected
  EmbeddedCudaGreedyExpected
} derive(Debug, Eq)

///|
/// One exact two-token offline correctness request. The encoded request is
/// copied at construction so later caller mutation cannot substitute bytes.
pub struct RuntimeSpawnedValidationContract {
  priv frame : FixedArray[Byte]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int
  priv expected_second_token : Int
  priv input_tokens : Int
  priv context_ceiling : Int
  priv tokens_per_page : Int
  priv turn_limit : Int
  priv sampling_expectation : RuntimeSpawnedSamplingExpectation
  priv require_fused_runtime : Bool
}

///|
/// Opaque immutable proof values returned only after the child, stream,
/// scheduler, KV owner, and service have all closed.
pub struct RuntimeSpawnedValidationResult {
  priv first_token : Int
  priv second_token : Int
  priv prefill_plan_sequence : UInt64
  priv decode_plan_sequence : UInt64
  priv kv_page_count : Int
  priv graph_captured : Bool
  priv prefill_graph_hits : UInt64
  priv prefill_graph_misses : UInt64
  priv decode_graph_hits : UInt64
  priv decode_graph_misses : UInt64
}

///|
fn spawned_validation_error(
  rule : RuntimeSpawnedValidationRule,
) -> RuntimeSpawnedValidationError {
  RuntimeSpawnedValidationFailed(rule)
}

///|
fn spawned_validation_contract_values_valid(
  frame_length : Int,
  expected_first_token : Int,
  expected_second_token : Int,
  input_tokens : Int,
  context_ceiling : Int,
  tokens_per_page : Int,
  turn_limit : Int,
) -> Bool {
  frame_length > 0 &&
  frame_length <= 1048576 &&
  expected_first_token >= 0 &&
  expected_second_token >= 0 &&
  input_tokens > 0 &&
  input_tokens <= 1048574 &&
  context_ceiling >= input_tokens + 2 &&
  context_ceiling <= 1048576 &&
  tokens_per_page >= input_tokens + 2 &&
  tokens_per_page <= 1048576 &&
  turn_limit > 0 &&
  turn_limit <= 1048576
}

///|
pub fn RuntimeSpawnedValidationContract::new(
  frame : FixedArray[Byte],
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int,
  expected_second_token : Int,
  input_tokens : Int,
  context_ceiling : Int,
  tokens_per_page : Int,
  turn_limit : Int,
) -> RuntimeSpawnedValidationContract raise RuntimeSpawnedValidationError {
  if !spawned_validation_contract_values_valid(
      frame.length(),
      expected_first_token,
      expected_second_token,
      input_tokens,
      context_ceiling,
      tokens_per_page,
      turn_limit,
    ) {
    raise spawned_validation_error(SpawnedValidationContract)
  }
  let owned = FixedArray::make(frame.length(), b'\x00')
  for index in 0..<frame.length() {
    owned[index] = frame[index]
  }
  {
    frame: owned,
    expected_model,
    expected_first_token,
    expected_second_token,
    input_tokens,
    context_ceiling,
    tokens_per_page,
    turn_limit,
    sampling_expectation: AnySpawnedSamplingRuntime,
    require_fused_runtime: false,
  }
}

///|
/// Adds an exact execution-extension expectation without exposing startup or
/// worker authority.
pub fn RuntimeSpawnedValidationContract::with_execution_expectation(
  self : RuntimeSpawnedValidationContract,
  sampling : RuntimeSpawnedSamplingExpectation,
  require_fused_runtime : Bool,
) -> RuntimeSpawnedValidationContract {
  { ..self, sampling_expectation: sampling, require_fused_runtime }
}

///|
pub fn RuntimeSpawnedValidationResult::first_token(
  self : RuntimeSpawnedValidationResult,
) -> Int {
  self.first_token
}

///|
pub fn RuntimeSpawnedValidationResult::second_token(
  self : RuntimeSpawnedValidationResult,
) -> Int {
  self.second_token
}

///|
pub fn RuntimeSpawnedValidationResult::prefill_plan_sequence(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.prefill_plan_sequence
}

///|
pub fn RuntimeSpawnedValidationResult::decode_plan_sequence(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.decode_plan_sequence
}

///|
pub fn RuntimeSpawnedValidationResult::kv_page_count(
  self : RuntimeSpawnedValidationResult,
) -> Int {
  self.kv_page_count
}

///|
pub fn RuntimeSpawnedValidationResult::graph_captured(
  self : RuntimeSpawnedValidationResult,
) -> Bool {
  self.graph_captured
}

///|
pub fn RuntimeSpawnedValidationResult::prefill_graph_hits(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.prefill_graph_hits
}

///|
pub fn RuntimeSpawnedValidationResult::prefill_graph_misses(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.prefill_graph_misses
}

///|
pub fn RuntimeSpawnedValidationResult::decode_graph_hits(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.decode_graph_hits
}

///|
pub fn RuntimeSpawnedValidationResult::decode_graph_misses(
  self : RuntimeSpawnedValidationResult,
) -> UInt64 {
  self.decode_graph_misses
}
