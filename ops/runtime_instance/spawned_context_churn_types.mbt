///|
/// Fail-closed stages for bounded same-child request and fresh-child context
/// churn qualification.
pub enum RuntimeSpawnedContextChurnRule {
  SpawnedContextChurnContract
  SpawnedContextChurnOwner
  SpawnedContextChurnIngress
  SpawnedContextChurnEvent
  SpawnedContextChurnPlan
  SpawnedContextChurnMetrics
  SpawnedContextChurnCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedContextChurnRule::code(
  self : RuntimeSpawnedContextChurnRule,
) -> String {
  match self {
    SpawnedContextChurnContract => "contract"
    SpawnedContextChurnOwner => "owner"
    SpawnedContextChurnIngress => "ingress"
    SpawnedContextChurnEvent => "event"
    SpawnedContextChurnPlan => "plan"
    SpawnedContextChurnMetrics => "metrics"
    SpawnedContextChurnCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedContextChurnError {
  RuntimeSpawnedContextChurnFailed(RuntimeSpawnedContextChurnRule)
} derive(Debug, Eq)

///|
/// Three exact disabled-cache requests for one already authenticated physical
/// child. Construction decodes and copies every frame before retaining it.
pub struct RuntimeSpawnedContextChurnContract {
  priv frames : FixedArray[FixedArray[Byte]]
  priv context_ceilings : FixedArray[Int]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int
  priv expected_second_token : Int
  priv input_tokens : Int
  priv turn_limit : Int
}

///|
/// Scalar evidence published only after all streams, the service, child, and
/// device-owned resources have closed.
pub struct RuntimeSpawnedContextChurnResult {
  priv request_count : Int
  priv event_count : Int
  priv plan_count : Int
  priv last_plan_sequence : UInt64
  priv initial_kv_pages_free : Int
  priv final_kv_pages_free : Int
  priv network_accepts : UInt64
  priv network_disconnects : UInt64
  priv network_rejections : UInt64
}

///|
fn spawned_context_churn_error(
  rule : RuntimeSpawnedContextChurnRule,
) -> RuntimeSpawnedContextChurnError {
  RuntimeSpawnedContextChurnFailed(rule)
}

///|
fn copy_context_churn_frame(source : FixedArray[Byte]) -> FixedArray[Byte] {
  FixedArray::makei(source.length(), index => source[index])
}

///|
fn context_churn_scalar_contract_valid(
  expected_first_token : Int,
  expected_second_token : Int,
  maximum_token_id : Int,
  input_tokens : Int,
  tokens_per_page : Int,
  first_context : Int,
  second_context : Int,
  third_context : Int,
  maximum_context : Int,
  turn_limit : Int,
) -> Bool {
  expected_first_token >= 0 &&
  expected_first_token <= maximum_token_id &&
  expected_second_token >= 0 &&
  expected_second_token <= maximum_token_id &&
  input_tokens > 0 &&
  tokens_per_page >= input_tokens + 2 &&
  first_context >= input_tokens + 2 &&
  first_context < second_context &&
  second_context < third_context &&
  third_context <= maximum_context &&
  turn_limit > 0 &&
  turn_limit <= 1048576
}

///|
pub fn RuntimeSpawnedContextChurnContract::new(
  first_frame : FixedArray[Byte],
  first_context : Int,
  second_frame : FixedArray[Byte],
  second_context : Int,
  third_frame : FixedArray[Byte],
  third_context : Int,
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int,
  expected_second_token : Int,
  input_tokens : Int,
  tokens_per_page : Int,
  inference_limits : @inference.InferenceLimits,
  maximum_frame_bytes : Int,
  turn_limit : Int,
) -> RuntimeSpawnedContextChurnContract raise RuntimeSpawnedContextChurnError {
  let framed_limits = @framed_wire.FramedWireLimits::new(
    inference_limits,
    max_frame_bytes=maximum_frame_bytes,
  ) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnContract)
  }
  if !context_churn_scalar_contract_valid(
      expected_first_token,
      expected_second_token,
      inference_limits.max_token_id(),
      input_tokens,
      tokens_per_page,
      first_context,
      second_context,
      third_context,
      inference_limits.max_context_tokens(),
      turn_limit,
    ) {
    raise spawned_context_churn_error(SpawnedContextChurnContract)
  }
  let sources = [first_frame, second_frame, third_frame]
  let contexts : FixedArray[Int] = [
    first_context, second_context, third_context,
  ]
  let owned : FixedArray[FixedArray[Byte]] = FixedArray::makei(3, index => {
    let source = sources[index]
    if source.length() <= 0 || source.length() > framed_limits.max_frame_bytes() {
      raise spawned_context_churn_error(SpawnedContextChurnContract)
    }
    let request = @framed_wire.RequestFrameBuffer::new(framed_limits)
      .load(source, source.length())
      .request() catch {
        _ => raise spawned_context_churn_error(SpawnedContextChurnContract)
      }
    if !spawned_serving_request_matches(
        request,
        expected_model,
        input_tokens,
        contexts[index],
      ) {
      raise spawned_context_churn_error(SpawnedContextChurnContract)
    }
    copy_context_churn_frame(source)
  })
  {
    frames: owned,
    context_ceilings: contexts,
    expected_model,
    expected_first_token,
    expected_second_token,
    input_tokens,
    turn_limit,
  }
}

///|
pub fn RuntimeSpawnedContextChurnResult::request_count(
  self : RuntimeSpawnedContextChurnResult,
) -> Int {
  self.request_count
}

///|
pub fn RuntimeSpawnedContextChurnResult::event_count(
  self : RuntimeSpawnedContextChurnResult,
) -> Int {
  self.event_count
}

///|
pub fn RuntimeSpawnedContextChurnResult::plan_count(
  self : RuntimeSpawnedContextChurnResult,
) -> Int {
  self.plan_count
}

///|
pub fn RuntimeSpawnedContextChurnResult::last_plan_sequence(
  self : RuntimeSpawnedContextChurnResult,
) -> UInt64 {
  self.last_plan_sequence
}

///|
pub fn RuntimeSpawnedContextChurnResult::initial_kv_pages_free(
  self : RuntimeSpawnedContextChurnResult,
) -> Int {
  self.initial_kv_pages_free
}

///|
pub fn RuntimeSpawnedContextChurnResult::final_kv_pages_free(
  self : RuntimeSpawnedContextChurnResult,
) -> Int {
  self.final_kv_pages_free
}

///|
pub fn RuntimeSpawnedContextChurnResult::network_accepts(
  self : RuntimeSpawnedContextChurnResult,
) -> UInt64 {
  self.network_accepts
}

///|
pub fn RuntimeSpawnedContextChurnResult::network_disconnects(
  self : RuntimeSpawnedContextChurnResult,
) -> UInt64 {
  self.network_disconnects
}

///|
pub fn RuntimeSpawnedContextChurnResult::network_rejections(
  self : RuntimeSpawnedContextChurnResult,
) -> UInt64 {
  self.network_rejections
}
