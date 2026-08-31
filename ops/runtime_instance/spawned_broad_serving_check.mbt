///|
priv struct BroadServingMetricSample {
  admissions : UInt64
  completions : UInt64
  cancellations : UInt64
  deadlines : UInt64
  failures : UInt64
  prompt_tokens : UInt64
  generated_tokens : UInt64
  worker_failures : UInt64
  backpressure : UInt64
  network_accepts : UInt64
  network_disconnects : UInt64
  network_rejections : UInt64
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
}

///|
priv struct BroadServingEventLedger {
  mut tail_accepts : Int
  mut first_tokens : Int
  mut second_tokens : Int
  mut cancelled_usage : Int
  mut survivor_usage : Int
  mut tail_usage : Int
  mut cancelled_completed : Int
  mut survivor_completed : Int
  mut tail_completed : Int
}

///|
fn BroadServingEventLedger::new() -> BroadServingEventLedger {
  {
    tail_accepts: 0,
    first_tokens: 0,
    second_tokens: 0,
    cancelled_usage: 0,
    survivor_usage: 0,
    tail_usage: 0,
    cancelled_completed: 0,
    survivor_completed: 0,
    tail_completed: 0,
  }
}

///|
fn capture_broad_serving_metrics(
  owner : RuntimeInstanceOwner,
) -> BroadServingMetricSample raise RuntimeSpawnedBroadServingValidationError {
  let snapshot = owner.metrics_snapshot() catch {
    _ => raise broad_serving_error(BroadServingMetrics)
  }
  {
    admissions: snapshot.counter_value(Admissions),
    completions: snapshot.counter_value(Completions),
    cancellations: snapshot.counter_value(Cancellations),
    deadlines: snapshot.counter_value(Deadlines),
    failures: snapshot.counter_value(Failures),
    prompt_tokens: snapshot.counter_value(PromptTokens),
    generated_tokens: snapshot.counter_value(GeneratedTokens),
    worker_failures: snapshot.counter_value(WorkerFailures),
    backpressure: snapshot.counter_value(Backpressure),
    network_accepts: snapshot.counter_value(NetworkAccepts),
    network_disconnects: snapshot.counter_value(NetworkDisconnects),
    network_rejections: snapshot.counter_value(NetworkRejections),
    queue_depth: snapshot.gauge_value(QueueDepth),
    active_requests: snapshot.gauge_value(ActiveRequests),
    kv_pages_used: snapshot.gauge_value(KvPagesUsed),
    kv_pages_free: snapshot.gauge_value(KvPagesFree),
  } catch {
    _ => raise broad_serving_error(BroadServingMetrics)
  }
}

///|
fn broad_serving_initial_metrics_valid(
  sample : BroadServingMetricSample,
) -> Bool {
  sample.admissions == 0UL &&
  sample.completions == 0UL &&
  sample.cancellations == 0UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == 0UL &&
  sample.generated_tokens == 0UL &&
  sample.worker_failures == 0UL &&
  sample.backpressure == 0UL &&
  sample.network_accepts == 0UL &&
  sample.network_disconnects == 0UL &&
  sample.network_rejections == 0UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free > 0
}

///|
fn broad_serving_pair_metrics_valid(
  sample : BroadServingMetricSample,
  input_tokens : Int,
  initial_kv_pages_free : Int,
  pooled? : Bool = false,
) -> Bool {
  sample.admissions == 3UL &&
  sample.completions == 2UL &&
  sample.cancellations == 1UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == (input_tokens * 3).to_uint64() &&
  sample.generated_tokens == 3UL &&
  sample.worker_failures == 0UL &&
  (pooled || sample.backpressure > 0UL) &&
  sample.network_accepts == (if pooled { 3UL } else { 1UL }) &&
  sample.network_disconnects == (if pooled { 3UL } else { 0UL }) &&
  sample.network_rejections == 0UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free == initial_kv_pages_free
}

///|
fn broad_serving_pool_cancel_boundary_valid(
  sample : BroadServingMetricSample,
) -> Bool {
  sample.admissions >= 1UL &&
  sample.admissions <= 2UL &&
  sample.active_requests + sample.queue_depth == 2 &&
  sample.completions == 0UL &&
  sample.cancellations == 0UL &&
  sample.generated_tokens == 0UL &&
  sample.failures == 0UL &&
  sample.deadlines == 0UL &&
  sample.worker_failures == 0UL &&
  sample.network_rejections == 0UL
}

///|
fn broad_serving_pool_cancel_progress_valid(
  sample : BroadServingMetricSample,
) -> Bool {
  sample.admissions >= 1UL &&
  sample.admissions <= 2UL &&
  sample.cancellations <= 1UL &&
  sample.completions <= 1UL &&
  sample.generated_tokens <= 2UL &&
  sample.failures == 0UL &&
  sample.deadlines == 0UL &&
  sample.worker_failures == 0UL &&
  sample.network_rejections == 0UL
}

///|
fn broad_serving_malformed_request_metrics_valid(
  sample : BroadServingMetricSample,
  input_tokens : Int,
  pooled? : Bool = false,
) -> Bool {
  sample.admissions == 4UL &&
  sample.completions == 3UL &&
  sample.cancellations == 1UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == (input_tokens * 4).to_uint64() &&
  sample.generated_tokens == 5UL &&
  sample.worker_failures == 0UL &&
  (pooled || sample.backpressure > 0UL)
}

///|
fn broad_serving_malformed_network_metrics_valid(
  sample : BroadServingMetricSample,
  pooled? : Bool = false,
) -> Bool {
  sample.network_accepts == (if pooled { 6UL } else { 4UL }) &&
  sample.network_disconnects == (if pooled { 6UL } else { 4UL }) &&
  sample.network_rejections == 2UL
}

///|
fn broad_serving_resource_metrics_valid(
  sample : BroadServingMetricSample,
  initial_kv_pages_free : Int,
) -> Bool {
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free == initial_kv_pages_free
}

///|
fn broad_serving_final_request_metrics_valid(
  sample : BroadServingMetricSample,
  input_tokens : Int,
  pooled? : Bool = false,
) -> Bool {
  sample.admissions == 5UL &&
  sample.completions == 4UL &&
  sample.cancellations == 1UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == (input_tokens * 5).to_uint64() &&
  sample.generated_tokens == 7UL &&
  sample.worker_failures == 0UL &&
  (pooled || sample.backpressure > 0UL)
}

///|
fn broad_serving_final_network_metrics_valid(
  sample : BroadServingMetricSample,
  pooled? : Bool = false,
) -> Bool {
  sample.network_accepts == (if pooled { 7UL } else { 5UL }) &&
  sample.network_disconnects == (if pooled { 7UL } else { 5UL }) &&
  sample.network_rejections == 2UL
}

///|
fn broad_serving_usage_valid(
  usage : @inference.Usage,
  input_tokens : Int,
  output_tokens : Int,
) -> Bool {
  usage.input_tokens() == input_tokens &&
  usage.cached_input_tokens() == 0 &&
  usage.output_tokens() == output_tokens &&
  usage.total_tokens() == input_tokens + output_tokens
}

///|
fn BroadServingEventLedger::record(
  self : BroadServingEventLedger,
  event : @inference.StreamEvent,
  contract : RuntimeSpawnedBroadServingValidationContract,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  match event {
    Accepted(accepted) if accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == 2 &&
      accepted.effective_limits().context_ceiling() == contract.context_ceiling =>
      self.tail_accepts += 1
    Accepted(_) =>
      raise broad_serving_error(BroadServingPairRemainingAcceptedShape)
    Token(token) if token.position() == 0 &&
      token.token_id() == contract.expected_first_token =>
      self.first_tokens += 1
    Token(token) if token.position() == 0 =>
      raise broad_serving_error(BroadServingPairRemainingToken0Shape)
    Token(token) if token.position() == 1 &&
      token.token_id() == contract.expected_second_token =>
      self.second_tokens += 1
    Token(token) if token.position() == 1 =>
      raise broad_serving_error(BroadServingPairRemainingToken1Shape)
    Token(_) =>
      raise broad_serving_error(BroadServingPairRemainingTokenPosition)
    Usage(usage) if broad_serving_usage_valid(usage, contract.input_tokens, 0) =>
      self.cancelled_usage += 1
    Usage(usage) if broad_serving_usage_valid(usage, contract.input_tokens, 1) =>
      self.survivor_usage += 1
    Usage(usage) if broad_serving_usage_valid(usage, contract.input_tokens, 2) =>
      self.tail_usage += 1
    Usage(_) => raise broad_serving_error(BroadServingPairRemainingUsageShape)
    Completed(completed) if completed.finish_reason() == Cancelled &&
      broad_serving_usage_valid(
        completed.final_usage(),
        contract.input_tokens,
        0,
      ) => self.cancelled_completed += 1
    Completed(completed) if completed.finish_reason() == TokenLimit &&
      broad_serving_usage_valid(
        completed.final_usage(),
        contract.input_tokens,
        1,
      ) => self.survivor_completed += 1
    Completed(completed) if completed.finish_reason() == TokenLimit &&
      broad_serving_usage_valid(
        completed.final_usage(),
        contract.input_tokens,
        2,
      ) => self.tail_completed += 1
    Completed(_) =>
      raise broad_serving_error(BroadServingPairRemainingCompletedShape)
    Failed(_) =>
      raise broad_serving_error(BroadServingPairRemainingUnexpectedEvent)
  }
}

///|
fn BroadServingEventLedger::is_exact(self : BroadServingEventLedger) -> Bool {
  self.tail_accepts == 1 &&
  self.first_tokens == 2 &&
  self.second_tokens == 1 &&
  self.cancelled_usage == 1 &&
  self.survivor_usage == 1 &&
  self.tail_usage == 1 &&
  self.cancelled_completed == 1 &&
  self.survivor_completed == 1 &&
  self.tail_completed == 1
}

///|
fn BroadServingEventLedger::require_exact(
  self : BroadServingEventLedger,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  if !self.is_exact() {
    raise broad_serving_error(BroadServingPairRemainingCount)
  }
}

///|
fn require_broad_accepted(
  event : @inference.StreamEvent,
  contract : RuntimeSpawnedBroadServingValidationContract,
  maximum_new_tokens : Int,
  failure_rule : RuntimeSpawnedBroadServingValidationRule,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  let valid = match event {
    Accepted(accepted) =>
      accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == maximum_new_tokens &&
      accepted.effective_limits().context_ceiling() == contract.context_ceiling
    _ => false
  }
  if !valid {
    raise broad_serving_error(failure_rule)
  }
}

///|
fn require_broad_token(
  event : @inference.StreamEvent,
  expected_token : Int,
  expected_position : Int,
  failure_rule : RuntimeSpawnedBroadServingValidationRule,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  match event {
    Token(token) if token.token_id() == expected_token &&
      token.position() == expected_position => ()
    _ => raise broad_serving_error(failure_rule)
  }
}

///|
fn require_broad_usage(
  event : @inference.StreamEvent,
  input_tokens : Int,
  output_tokens : Int,
  failure_rule : RuntimeSpawnedBroadServingValidationRule,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  match event {
    Usage(usage) if broad_serving_usage_valid(
        usage, input_tokens, output_tokens,
      ) => ()
    _ => raise broad_serving_error(failure_rule)
  }
}

///|
fn require_broad_completed(
  event : @inference.StreamEvent,
  input_tokens : Int,
  output_tokens : Int,
  reason : @inference.FinishReason,
  failure_rule : RuntimeSpawnedBroadServingValidationRule,
) -> Unit raise RuntimeSpawnedBroadServingValidationError {
  match event {
    Completed(completed) if completed.finish_reason() == reason &&
      broad_serving_usage_valid(
        completed.final_usage(),
        input_tokens,
        output_tokens,
      ) => ()
    _ => raise broad_serving_error(failure_rule)
  }
}
