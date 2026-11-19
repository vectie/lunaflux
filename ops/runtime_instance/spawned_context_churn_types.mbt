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
  priv context_ceilings : FixedArray[Int32]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int32
  priv expected_second_token : Int32
  priv input_tokens : Int32
  priv turn_limit : Int32
}

///|
/// Scalar evidence published only after all streams, the service, child, and
/// device-owned resources have closed.
pub struct RuntimeSpawnedContextChurnResult {
  priv request_count : Int32
  priv event_count : Int32
  priv plan_count : Int32
  priv last_plan_sequence : UInt64
  priv initial_kv_pages_free : Int32
  priv final_kv_pages_free : Int32
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
  expected_first_token : Int32,
  expected_second_token : Int32,
  maximum_token_id : Int32,
  input_tokens : Int32,
  tokens_per_page : Int32,
  first_context : Int32,
  second_context : Int32,
  third_context : Int32,
  maximum_context : Int32,
  turn_limit : Int32,
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
  first_context : Int32,
  second_frame : FixedArray[Byte],
  second_context : Int32,
  third_frame : FixedArray[Byte],
  third_context : Int32,
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int32,
  expected_second_token : Int32,
  input_tokens : Int32,
  tokens_per_page : Int32,
  inference_limits : @inference.InferenceLimits,
  maximum_frame_bytes : Int32,
  turn_limit : Int32,
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
  let contexts : FixedArray[Int32] = [
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
) -> Int32 {
  self.request_count
}

///|
pub fn RuntimeSpawnedContextChurnResult::event_count(
  self : RuntimeSpawnedContextChurnResult,
) -> Int32 {
  self.event_count
}

///|
pub fn RuntimeSpawnedContextChurnResult::plan_count(
  self : RuntimeSpawnedContextChurnResult,
) -> Int32 {
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
) -> Int32 {
  self.initial_kv_pages_free
}

///|
pub fn RuntimeSpawnedContextChurnResult::final_kv_pages_free(
  self : RuntimeSpawnedContextChurnResult,
) -> Int32 {
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
