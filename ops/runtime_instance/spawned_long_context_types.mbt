///|
pub enum RuntimeSpawnedLongContextRule {
  SpawnedLongContextContract
  SpawnedLongContextOwner
  SpawnedLongContextIngress
  SpawnedLongContextEvent
  SpawnedLongContextPlan
  SpawnedLongContextKv
  SpawnedLongContextTelemetry
  SpawnedLongContextCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedLongContextRule::code(
  self : RuntimeSpawnedLongContextRule,
) -> String {
  match self {
    SpawnedLongContextContract => "contract"
    SpawnedLongContextOwner => "owner"
    SpawnedLongContextIngress => "ingress"
    SpawnedLongContextEvent => "event"
    SpawnedLongContextPlan => "plan"
    SpawnedLongContextKv => "kv"
    SpawnedLongContextTelemetry => "telemetry"
    SpawnedLongContextCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedLongContextError {
  RuntimeSpawnedLongContextFailed(RuntimeSpawnedLongContextRule)
} derive(Debug, Eq)

///|
/// Caller-pinned 8-token and 9-token requests for one direct authenticated
/// physical child. The outer campaign owns referee provenance; this neutral
/// runtime contract requires both requests to disable prefix reuse.
pub struct RuntimeSpawnedLongContextContract {
  priv first_frame : FixedArray[Byte]
  priv second_frame : FixedArray[Byte]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int32
  priv expected_second_token : Int32
  priv context_ceiling : Int32
  priv tokens_per_page : Int32
  priv turn_limit : Int32
}

///|
/// Scalar evidence published only after both requests and all owned physical
/// resources have closed.
pub struct RuntimeSpawnedLongContextResult {
  priv first_token : Int32
  priv second_token : Int32
  priv request_count : Int32
  priv event_count : Int32
  priv plan_count : Int32
  priv last_plan_sequence : UInt64
  priv input_tokens_processed : Int32
  priv initial_kv_pages_free : Int32
  priv final_kv_pages_free : Int32
}

///|
fn spawned_long_context_error(
  rule : RuntimeSpawnedLongContextRule,
) -> RuntimeSpawnedLongContextError {
  RuntimeSpawnedLongContextFailed(rule)
}

///|
fn long_context_request_matches(
  request : @inference.GenerateRequest,
  expected_model : @spec.ModelIdentity,
  expected_input_tokens : Int32,
  context_ceiling : Int32,
) -> Bool {
  let token_count = match request.input() {
    TokenIds(tokens) => tokens.length()
    Text(_) => -1
  }
  request.model_identity() == expected_model &&
  token_count == expected_input_tokens &&
  request.max_new_tokens() == 1 &&
  request.context_ceiling() == context_ceiling &&
  request.sampling().mode() == Greedy &&
  request.stream_preference() == Streaming &&
  request.cache().permission() == Disabled
}

///|
fn copy_long_context_frame(source : FixedArray[Byte]) -> FixedArray[Byte] {
  FixedArray::makei(source.length(), index => source[index])
}

///|
pub fn RuntimeSpawnedLongContextContract::new(
  first_frame : FixedArray[Byte],
  second_frame : FixedArray[Byte],
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int32,
  expected_second_token : Int32,
  context_ceiling : Int32,
  tokens_per_page : Int32,
  inference_limits : @inference.InferenceLimits,
  maximum_frame_bytes : Int32,
  turn_limit : Int32,
) -> RuntimeSpawnedLongContextContract raise RuntimeSpawnedLongContextError {
  let framed_limits = @framed_wire.FramedWireLimits::new(
    inference_limits,
    max_frame_bytes=maximum_frame_bytes,
  ) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextContract)
  }
  let scalars_valid = expected_first_token >= 0 &&
    expected_first_token <= inference_limits.max_token_id() &&
    expected_second_token >= 0 &&
    expected_second_token <= inference_limits.max_token_id() &&
    context_ceiling == inference_limits.max_context_tokens() &&
    inference_limits.max_input_tokens() == 9 &&
    inference_limits.max_new_tokens() == 1 &&
    tokens_per_page == 8 &&
    turn_limit > 0 &&
    turn_limit <= 1048576
  if !scalars_valid {
    raise spawned_long_context_error(SpawnedLongContextContract)
  }
  let first = @framed_wire.RequestFrameBuffer::new(framed_limits)
    .load(first_frame, first_frame.length())
    .request() catch {
      _ => raise spawned_long_context_error(SpawnedLongContextContract)
    }
  let second = @framed_wire.RequestFrameBuffer::new(framed_limits)
    .load(second_frame, second_frame.length())
    .request() catch {
      _ => raise spawned_long_context_error(SpawnedLongContextContract)
    }
  if !long_context_request_matches(first, expected_model, 8, context_ceiling) ||
    !long_context_request_matches(second, expected_model, 9, context_ceiling) {
    raise spawned_long_context_error(SpawnedLongContextContract)
  }
  {
    first_frame: copy_long_context_frame(first_frame),
    second_frame: copy_long_context_frame(second_frame),
    expected_model,
    expected_first_token,
    expected_second_token,
    context_ceiling,
    tokens_per_page,
    turn_limit,
  }
}

///|
pub fn RuntimeSpawnedLongContextResult::first_token(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.first_token
}

///|
pub fn RuntimeSpawnedLongContextResult::second_token(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.second_token
}

///|
pub fn RuntimeSpawnedLongContextResult::request_count(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.request_count
}

///|
pub fn RuntimeSpawnedLongContextResult::event_count(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.event_count
}

///|
pub fn RuntimeSpawnedLongContextResult::plan_count(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.plan_count
}

///|
pub fn RuntimeSpawnedLongContextResult::last_plan_sequence(
  self : RuntimeSpawnedLongContextResult,
) -> UInt64 {
  self.last_plan_sequence
}

///|
pub fn RuntimeSpawnedLongContextResult::input_tokens_processed(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.input_tokens_processed
}

///|
pub fn RuntimeSpawnedLongContextResult::initial_kv_pages_free(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.initial_kv_pages_free
}

///|
pub fn RuntimeSpawnedLongContextResult::final_kv_pages_free(
  self : RuntimeSpawnedLongContextResult,
) -> Int32 {
  self.final_kv_pages_free
}
