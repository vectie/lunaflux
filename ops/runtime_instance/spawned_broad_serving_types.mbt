///|
/// Fail-closed stages for the qualification-only broad native serving probe.
pub enum RuntimeSpawnedBroadServingValidationRule {
  BroadServingContract
  BroadServingOwner
  BroadServingHealth
  BroadServingReadiness
  BroadServingConnect
  BroadServingIngress
  BroadServingConcurrency
  BroadServingCancellation
  BroadServingPairAccepted0
  BroadServingPairAccepted1
  BroadServingPairRemainingRead
  BroadServingPairRemainingAcceptedShape
  BroadServingPairRemainingToken0Shape
  BroadServingPairRemainingToken1Shape
  BroadServingPairRemainingTokenPosition
  BroadServingPairRemainingUsageShape
  BroadServingPairRemainingCompletedShape
  BroadServingPairRemainingUnexpectedEvent
  BroadServingPairRemainingCount
  BroadServingRecoveryAccepted
  BroadServingRecoveryToken0
  BroadServingRecoveryToken1
  BroadServingRecoveryUsage
  BroadServingRecoveryCompleted
  BroadServingBackpressure
  BroadServingRejection
  BroadServingMalformedTransition
  BroadServingMalformedUnexpectedOwnerFailure
  BroadServingMalformedRequestMetrics
  BroadServingMalformedNetworkMetrics
  BroadServingMalformedResourceMetrics
  BroadServingMalformedReuseAccepted
  BroadServingMalformedReuseToken0
  BroadServingMalformedReuseToken1
  BroadServingMalformedReuseUsage
  BroadServingMalformedReuseCompleted
  BroadServingMalformedReuseRequestMetrics
  BroadServingMalformedReuseNetworkMetrics
  BroadServingMalformedReuseResourceMetrics
  BroadServingRecovery
  BroadServingMetrics
  BroadServingCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedBroadServingValidationRule::code(
  self : RuntimeSpawnedBroadServingValidationRule,
) -> String {
  match self {
    BroadServingContract => "contract"
    BroadServingOwner => "owner"
    BroadServingHealth => "health"
    BroadServingReadiness => "readiness"
    BroadServingConnect => "connect"
    BroadServingIngress => "ingress"
    BroadServingConcurrency => "concurrency"
    BroadServingCancellation => "cancellation"
    BroadServingPairAccepted0 => "events_pair_accepted_0"
    BroadServingPairAccepted1 => "events_pair_accepted_1"
    BroadServingPairRemainingRead => "events_pair_remaining_read"
    BroadServingPairRemainingAcceptedShape =>
      "events_pair_remaining_accepted_shape"
    BroadServingPairRemainingToken0Shape =>
      "events_pair_remaining_token_0_shape"
    BroadServingPairRemainingToken1Shape =>
      "events_pair_remaining_token_1_shape"
    BroadServingPairRemainingTokenPosition =>
      "events_pair_remaining_token_position"
    BroadServingPairRemainingUsageShape => "events_pair_remaining_usage_shape"
    BroadServingPairRemainingCompletedShape =>
      "events_pair_remaining_completed_shape"
    BroadServingPairRemainingUnexpectedEvent =>
      "events_pair_remaining_unexpected_event"
    BroadServingPairRemainingCount => "events_pair_remaining_count"
    BroadServingRecoveryAccepted => "events_recovery_accepted"
    BroadServingRecoveryToken0 => "events_recovery_token_0"
    BroadServingRecoveryToken1 => "events_recovery_token_1"
    BroadServingRecoveryUsage => "events_recovery_usage"
    BroadServingRecoveryCompleted => "events_recovery_completed"
    BroadServingBackpressure => "backpressure"
    BroadServingRejection => "rejection"
    BroadServingMalformedTransition => "malformed_transition"
    BroadServingMalformedUnexpectedOwnerFailure =>
      "malformed_unexpected_owner_failure"
    BroadServingMalformedRequestMetrics => "malformed_metrics_request"
    BroadServingMalformedNetworkMetrics => "malformed_metrics_network"
    BroadServingMalformedResourceMetrics => "malformed_metrics_resources"
    BroadServingMalformedReuseAccepted => "malformed_reuse_events_accepted"
    BroadServingMalformedReuseToken0 => "malformed_reuse_events_token_0"
    BroadServingMalformedReuseToken1 => "malformed_reuse_events_token_1"
    BroadServingMalformedReuseUsage => "malformed_reuse_events_usage"
    BroadServingMalformedReuseCompleted => "malformed_reuse_events_completed"
    BroadServingMalformedReuseRequestMetrics =>
      "malformed_reuse_metrics_request"
    BroadServingMalformedReuseNetworkMetrics =>
      "malformed_reuse_metrics_network"
    BroadServingMalformedReuseResourceMetrics =>
      "malformed_reuse_metrics_resources"
    BroadServingRecovery => "recovery"
    BroadServingMetrics => "metrics"
    BroadServingCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedBroadServingValidationError {
  RuntimeSpawnedBroadServingValidationFailed(
    RuntimeSpawnedBroadServingValidationRule
  )
} derive(Debug, Eq)

///|
/// A copied, bounded qualification contract. The first two frames must be
/// cache-free one-token requests for the pinned model, the third is a retained
/// two-token saturation tail, and the recovery frame is a distinct two-token
/// probe. The final frame must be a well-formed request for a different model
/// identity.
pub struct RuntimeSpawnedBroadServingValidationContract {
  priv first_frame : FixedArray[Byte]
  priv second_frame : FixedArray[Byte]
  priv saturation_tail_frame : FixedArray[Byte]
  priv recovery_frame : FixedArray[Byte]
  priv foreign_frame : FixedArray[Byte]
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
/// The exact native ingress authority that produced the broad-serving
/// qualification result. The distinction keeps saturation evidence truthful:
/// a singleton listener observes scheduler backpressure, while a connection
/// pool proves saturation by deferring admission at its connection capacity.
pub(all) enum RuntimeSpawnedBroadServingIngressOwner {
  BroadServingSingletonServer
  BroadServingFramedConnectionPool
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedBroadServingIngressOwner::code(
  self : RuntimeSpawnedBroadServingIngressOwner,
) -> String {
  match self {
    BroadServingSingletonServer => "singleton-server"
    BroadServingFramedConnectionPool => "framed-connection-pool"
  }
}

///|
/// Immutable qualification scalars published only after the native listener,
/// service, scheduler, worker child, and KV authority have closed.
pub struct RuntimeSpawnedBroadServingValidationResult {
  priv ingress_owner : RuntimeSpawnedBroadServingIngressOwner
  priv request_count : UInt64
  priv completion_count : UInt64
  priv cancellation_count : UInt64
  priv event_count : Int
  priv backpressure_count : UInt64
  priv network_accepts : UInt64
  priv network_disconnects : UInt64
  priv network_rejections : UInt64
  priv kv_pages_free : Int
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::ingress_owner(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> RuntimeSpawnedBroadServingIngressOwner {
  self.ingress_owner
}

///|
fn broad_serving_error(
  rule : RuntimeSpawnedBroadServingValidationRule,
) -> RuntimeSpawnedBroadServingValidationError {
  RuntimeSpawnedBroadServingValidationFailed(rule)
}

///|
fn broad_serving_copy_frame(source : FixedArray[Byte]) -> FixedArray[Byte] {
  let owned = FixedArray::make(source.length(), b'\x00')
  for index in 0..<source.length() {
    owned[index] = source[index]
  }
  owned
}

///|
fn broad_serving_request_matches(
  request : @inference.GenerateRequest,
  expected_model : @spec.ModelIdentity,
  input_tokens : Int,
  context_ceiling : Int,
  maximum_new_tokens : Int,
) -> Bool {
  let token_count = match request.input() {
    TokenIds(tokens) => tokens.length()
    Text(_) => -1
  }
  request.model_identity() == expected_model &&
  token_count == input_tokens &&
  request.max_new_tokens() == maximum_new_tokens &&
  request.context_ceiling() == context_ceiling &&
  request.sampling().mode() == Greedy &&
  request.stream_preference() == Streaming &&
  request.cache().permission() == Disabled
}

///|
fn broad_serving_decode_request(
  frame : FixedArray[Byte],
  limits : @framed_wire.FramedWireLimits,
) -> @inference.GenerateRequest raise RuntimeSpawnedBroadServingValidationError {
  @framed_wire.RequestFrameBuffer::new(limits)
  .load(frame, frame.length())
  .request() catch {
    _ => raise broad_serving_error(BroadServingContract)
  }
}

///|
pub fn RuntimeSpawnedBroadServingValidationContract::new(
  first_frame : FixedArray[Byte],
  second_frame : FixedArray[Byte],
  saturation_tail_frame : FixedArray[Byte],
  recovery_frame : FixedArray[Byte],
  foreign_frame : FixedArray[Byte],
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
) -> RuntimeSpawnedBroadServingValidationContract raise RuntimeSpawnedBroadServingValidationError {
  let framed_limits = @framed_wire.FramedWireLimits::new(
    inference_limits,
    max_frame_bytes=maximum_frame_bytes,
  ) catch {
    _ => raise broad_serving_error(BroadServingContract)
  }
  if first_frame.length() <= 0 ||
    second_frame.length() <= 0 ||
    saturation_tail_frame.length() <= 0 ||
    recovery_frame.length() <= 0 ||
    foreign_frame.length() <= 0 ||
    first_frame.length() > framed_limits.max_frame_bytes() ||
    second_frame.length() > framed_limits.max_frame_bytes() ||
    saturation_tail_frame.length() > framed_limits.max_frame_bytes() ||
    recovery_frame.length() > framed_limits.max_frame_bytes() ||
    foreign_frame.length() > framed_limits.max_frame_bytes() ||
    expected_first_token < 0 ||
    expected_first_token > inference_limits.max_token_id() ||
    expected_second_token < 0 ||
    expected_second_token > inference_limits.max_token_id() ||
    input_tokens <= 0 ||
    context_ceiling < input_tokens + 2 ||
    context_ceiling > inference_limits.max_context_tokens() ||
    tokens_per_page < input_tokens + 2 ||
    turn_limit <= 0 ||
    turn_limit > 1048576 ||
    io_timeout_millis <= 0 ||
    io_timeout_millis > 60000 {
    raise broad_serving_error(BroadServingContract)
  }
  let first = broad_serving_decode_request(first_frame, framed_limits)
  let second = broad_serving_decode_request(second_frame, framed_limits)
  let saturation_tail = broad_serving_decode_request(
    saturation_tail_frame, framed_limits,
  )
  let recovery = broad_serving_decode_request(recovery_frame, framed_limits)
  let foreign = broad_serving_decode_request(foreign_frame, framed_limits)
  if !broad_serving_request_matches(
      first, expected_model, input_tokens, context_ceiling, 1,
    ) ||
    !broad_serving_request_matches(
      second, expected_model, input_tokens, context_ceiling, 1,
    ) ||
    !broad_serving_request_matches(
      saturation_tail, expected_model, input_tokens, context_ceiling, 2,
    ) ||
    !broad_serving_request_matches(
      recovery, expected_model, input_tokens, context_ceiling, 2,
    ) ||
    foreign.model_identity() == expected_model ||
    !broad_serving_request_matches(
      foreign,
      foreign.model_identity(),
      input_tokens,
      context_ceiling,
      1,
    ) {
    raise broad_serving_error(BroadServingContract)
  }
  {
    first_frame: broad_serving_copy_frame(first_frame),
    second_frame: broad_serving_copy_frame(second_frame),
    saturation_tail_frame: broad_serving_copy_frame(saturation_tail_frame),
    recovery_frame: broad_serving_copy_frame(recovery_frame),
    foreign_frame: broad_serving_copy_frame(foreign_frame),
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
pub fn RuntimeSpawnedBroadServingValidationResult::request_count(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.request_count
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::completion_count(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.completion_count
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::cancellation_count(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.cancellation_count
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::event_count(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> Int {
  self.event_count
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::backpressure_count(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.backpressure_count
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::network_accepts(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.network_accepts
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::network_disconnects(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.network_disconnects
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::network_rejections(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> UInt64 {
  self.network_rejections
}

///|
pub fn RuntimeSpawnedBroadServingValidationResult::kv_pages_free(
  self : RuntimeSpawnedBroadServingValidationResult,
) -> Int {
  self.kv_pages_free
}
