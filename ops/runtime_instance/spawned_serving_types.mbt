///|
/// Fail-closed stages for the bounded listener-level spawned validation.
pub enum RuntimeSpawnedServingValidationRule {
  SpawnedServingContract
  SpawnedServingOwner
  SpawnedServingHealth
  SpawnedServingReadiness
  SpawnedServingConnect
  SpawnedServingIngress
  SpawnedServingEvent
  SpawnedServingAccepted
  SpawnedServingFirstToken
  SpawnedServingSecondToken
  SpawnedServingUsage
  SpawnedServingCompleted
  SpawnedServingMetrics
  SpawnedServingCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedServingValidationRule::code(
  self : RuntimeSpawnedServingValidationRule,
) -> String {
  match self {
    SpawnedServingContract => "contract"
    SpawnedServingOwner => "owner"
    SpawnedServingHealth => "health"
    SpawnedServingReadiness => "readiness"
    SpawnedServingConnect => "connect"
    SpawnedServingIngress => "ingress"
    SpawnedServingEvent => "event"
    SpawnedServingAccepted => "accepted"
    SpawnedServingFirstToken => "first_token"
    SpawnedServingSecondToken => "second_token"
    SpawnedServingUsage => "usage"
    SpawnedServingCompleted => "completed"
    SpawnedServingMetrics => "metrics"
    SpawnedServingCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedServingValidationError {
  RuntimeSpawnedServingValidationFailed(RuntimeSpawnedServingValidationRule)
} derive(Debug, Eq)

///|
/// Exact native-framed serving request copied and authenticated at
/// construction. The contract admits only one streaming, greedy, cache-free,
/// two-token request so the operator method cannot become a general client.
pub struct RuntimeSpawnedServingValidationContract {
  priv frame : FixedArray[Byte]
  priv expected_model : @spec.ModelIdentity
  priv expected_first_token : Int
  priv expected_second_token : Int
  priv input_tokens : Int
  priv context_ceiling : Int
  priv framed_limits : @framed_wire.FramedWireLimits
  priv turn_limit : Int
  priv io_timeout_millis : Int
}

///|
/// Immutable proof scalars published only after client, listener, service,
/// child, scheduler, and KV authority have terminally closed.
pub struct RuntimeSpawnedServingValidationResult {
  priv first_token : Int
  priv second_token : Int
  priv event_count : Int
  priv kv_pages_free : Int
  priv network_accepts : UInt64
  priv network_disconnects : UInt64
}

///|
fn spawned_serving_error(
  rule : RuntimeSpawnedServingValidationRule,
) -> RuntimeSpawnedServingValidationError {
  RuntimeSpawnedServingValidationFailed(rule)
}

///|
fn spawned_serving_contract_scalars_valid(
  frame_length : Int,
  maximum_frame_length : Int,
  expected_first_token : Int,
  expected_second_token : Int,
  maximum_token_id : Int,
  input_tokens : Int,
  context_ceiling : Int,
  maximum_context_tokens : Int,
  tokens_per_page : Int,
  turn_limit : Int,
  io_timeout_millis : Int,
) -> Bool {
  frame_length > 0 &&
  frame_length <= maximum_frame_length &&
  expected_first_token >= 0 &&
  expected_first_token <= maximum_token_id &&
  expected_second_token >= 0 &&
  expected_second_token <= maximum_token_id &&
  input_tokens > 0 &&
  context_ceiling >= input_tokens + 2 &&
  context_ceiling <= maximum_context_tokens &&
  tokens_per_page >= input_tokens + 2 &&
  turn_limit > 0 &&
  turn_limit <= 1048576 &&
  io_timeout_millis > 0 &&
  io_timeout_millis <= 60000
}

///|
fn spawned_serving_request_matches(
  request : @inference.GenerateRequest,
  expected_model : @spec.ModelIdentity,
  input_tokens : Int,
  context_ceiling : Int,
) -> Bool {
  let token_count = match request.input() {
    TokenIds(tokens) => tokens.length()
    Text(_) => -1
  }
  request.model_identity() == expected_model &&
  token_count == input_tokens &&
  request.max_new_tokens() == 2 &&
  request.context_ceiling() == context_ceiling &&
  request.sampling().mode() == Greedy &&
  request.stream_preference() == Streaming &&
  request.cache().permission() == Disabled
}

///|
pub fn RuntimeSpawnedServingValidationContract::new(
  frame : FixedArray[Byte],
  expected_model : @spec.ModelIdentity,
  expected_first_token : Int,
  expected_second_token : Int,
  input_tokens : Int,
  context_ceiling : Int,
  tokens_per_page : Int,
  inference_limits : @inference.InferenceLimits,
  maximum_frame_bytes : Int,
  turn_limit : Int,
  io_timeout_millis : Int,
) -> RuntimeSpawnedServingValidationContract raise RuntimeSpawnedServingValidationError {
  let framed_limits = @framed_wire.FramedWireLimits::new(
    inference_limits,
    max_frame_bytes=maximum_frame_bytes,
  ) catch {
    _ => raise spawned_serving_error(SpawnedServingContract)
  }
  if !spawned_serving_contract_scalars_valid(
      frame.length(),
      framed_limits.max_frame_bytes(),
      expected_first_token,
      expected_second_token,
      inference_limits.max_token_id(),
      input_tokens,
      context_ceiling,
      inference_limits.max_context_tokens(),
      tokens_per_page,
      turn_limit,
      io_timeout_millis,
    ) {
    raise spawned_serving_error(SpawnedServingContract)
  }
  let request = @framed_wire.RequestFrameBuffer::new(framed_limits)
    .load(frame, frame.length())
    .request() catch {
      _ => raise spawned_serving_error(SpawnedServingContract)
    }
  if !spawned_serving_request_matches(
      request, expected_model, input_tokens, context_ceiling,
    ) {
    raise spawned_serving_error(SpawnedServingContract)
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
    framed_limits,
    turn_limit,
    io_timeout_millis,
  }
}

///|
pub fn RuntimeSpawnedServingValidationResult::first_token(
  self : RuntimeSpawnedServingValidationResult,
) -> Int {
  self.first_token
}

///|
pub fn RuntimeSpawnedServingValidationResult::second_token(
  self : RuntimeSpawnedServingValidationResult,
) -> Int {
  self.second_token
}

///|
pub fn RuntimeSpawnedServingValidationResult::event_count(
  self : RuntimeSpawnedServingValidationResult,
) -> Int {
  self.event_count
}

///|
pub fn RuntimeSpawnedServingValidationResult::kv_pages_free(
  self : RuntimeSpawnedServingValidationResult,
) -> Int {
  self.kv_pages_free
}

///|
pub fn RuntimeSpawnedServingValidationResult::network_accepts(
  self : RuntimeSpawnedServingValidationResult,
) -> UInt64 {
  self.network_accepts
}

///|
pub fn RuntimeSpawnedServingValidationResult::network_disconnects(
  self : RuntimeSpawnedServingValidationResult,
) -> UInt64 {
  self.network_disconnects
}
