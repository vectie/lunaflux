///|
/// Fail-closed stages for the owner-retained two-request prefix qualification.
pub enum RuntimeSpawnedPrefixValidationRule {
  SpawnedPrefixContract
  SpawnedPrefixOwner
  SpawnedPrefixIngress
  SpawnedPrefixRejected
  SpawnedPrefixEvent
  SpawnedPrefixPlan
  SpawnedPrefixUsage
  SpawnedPrefixTelemetry
  SpawnedPrefixCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedPrefixValidationRule::code(
  self : RuntimeSpawnedPrefixValidationRule,
) -> String {
  match self {
    SpawnedPrefixContract => "contract"
    SpawnedPrefixOwner => "owner"
    SpawnedPrefixIngress => "ingress"
    SpawnedPrefixRejected => "rejected"
    SpawnedPrefixEvent => "event"
    SpawnedPrefixPlan => "plan"
    SpawnedPrefixUsage => "usage"
    SpawnedPrefixTelemetry => "telemetry"
    SpawnedPrefixCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedPrefixValidationError {
  RuntimeSpawnedPrefixValidationFailed(RuntimeSpawnedPrefixValidationRule)
} derive(Debug, Eq)

///|
/// Exact two-request qualification input. The first prompt must fill complete
/// pages and the second adds one private-tail token. Both encoded frames are
/// copied so caller mutation cannot substitute request identity or cache scope.
pub struct RuntimeSpawnedPrefixValidationContract {
  priv first_frame : FixedArray[Byte]
  priv second_frame : FixedArray[Byte]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int
  priv expected_second_token : Int
  priv first_input_tokens : Int
  priv second_input_tokens : Int
  priv expected_reused_tokens : Int
  priv context_ceiling : Int
  priv tokens_per_page : Int
  priv turn_limit : Int
}

///|
/// Opaque scalar evidence returned only after both requests, stream epochs,
/// service, worker, device resources, and the runtime owner have closed.
pub struct RuntimeSpawnedPrefixValidationResult {
  priv first_token : Int
  priv second_token : Int
  priv first_terminal_plan_sequence : UInt64
  priv second_terminal_plan_sequence : UInt64
  priv prefix_lookups : UInt64
  priv prefix_hits : UInt64
  priv prefix_misses : UInt64
  priv prefix_publications : UInt64
  priv reused_tokens : UInt64
  priv retained_prefix_pages : Int
  priv initial_kv_pages_free : Int
  priv kv_pages_free : Int
}

///|
fn spawned_prefix_error(
  rule : RuntimeSpawnedPrefixValidationRule,
) -> RuntimeSpawnedPrefixValidationError {
  RuntimeSpawnedPrefixValidationFailed(rule)
}

///|
fn copy_spawned_prefix_frame(source : FixedArray[Byte]) -> FixedArray[Byte] {
  FixedArray::makei(source.length(), index => source[index])
}

///|
pub fn RuntimeSpawnedPrefixValidationContract::new(
  first_frame : FixedArray[Byte],
  second_frame : FixedArray[Byte],
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int,
  expected_second_token : Int,
  first_input_tokens : Int,
  second_input_tokens : Int,
  expected_reused_tokens : Int,
  context_ceiling : Int,
  tokens_per_page : Int,
  turn_limit : Int,
) -> RuntimeSpawnedPrefixValidationContract raise RuntimeSpawnedPrefixValidationError {
  let valid = first_frame.length() > 0 &&
    first_frame.length() <= 1048576 &&
    second_frame.length() > 0 &&
    second_frame.length() <= 1048576 &&
    expected_first_token >= 0 &&
    expected_second_token >= 0 &&
    first_input_tokens > 0 &&
    tokens_per_page > 0 &&
    first_input_tokens % tokens_per_page == 0 &&
    second_input_tokens == first_input_tokens + 1 &&
    expected_reused_tokens == first_input_tokens &&
    context_ceiling >= second_input_tokens + 1 &&
    context_ceiling <= 1048576 &&
    turn_limit > 0 &&
    turn_limit <= 1048576
  if !valid {
    raise spawned_prefix_error(SpawnedPrefixContract)
  }
  {
    first_frame: copy_spawned_prefix_frame(first_frame),
    second_frame: copy_spawned_prefix_frame(second_frame),
    expected_model,
    expected_first_token,
    expected_second_token,
    first_input_tokens,
    second_input_tokens,
    expected_reused_tokens,
    context_ceiling,
    tokens_per_page,
    turn_limit,
  }
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::first_token(
  self : RuntimeSpawnedPrefixValidationResult,
) -> Int {
  self.first_token
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::second_token(
  self : RuntimeSpawnedPrefixValidationResult,
) -> Int {
  self.second_token
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::first_terminal_plan_sequence(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.first_terminal_plan_sequence
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::second_terminal_plan_sequence(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.second_terminal_plan_sequence
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::prefix_lookups(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.prefix_lookups
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::prefix_hits(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.prefix_hits
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::prefix_misses(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.prefix_misses
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::prefix_publications(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.prefix_publications
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::reused_tokens(
  self : RuntimeSpawnedPrefixValidationResult,
) -> UInt64 {
  self.reused_tokens
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::retained_prefix_pages(
  self : RuntimeSpawnedPrefixValidationResult,
) -> Int {
  self.retained_prefix_pages
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::initial_kv_pages_free(
  self : RuntimeSpawnedPrefixValidationResult,
) -> Int {
  self.initial_kv_pages_free
}

///|
pub fn RuntimeSpawnedPrefixValidationResult::kv_pages_free(
  self : RuntimeSpawnedPrefixValidationResult,
) -> Int {
  self.kv_pages_free
}
