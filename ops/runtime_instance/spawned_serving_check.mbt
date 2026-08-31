///|
priv struct SpawnedServingMetricSample {
  admissions : UInt64
  completions : UInt64
  cancellations : UInt64
  deadlines : UInt64
  failures : UInt64
  prompt_tokens : UInt64
  generated_tokens : UInt64
  worker_failures : UInt64
  network_accepts : UInt64
  network_disconnects : UInt64
  network_rejections : UInt64
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
}

///|
fn spawned_serving_prebind_state_valid(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
) -> Bool {
  phase == HealthyNotReady &&
  health == RuntimeHealthy &&
  readiness == RuntimeNotReady
}

///|
fn spawned_serving_bound_state_valid(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
) -> Bool {
  phase == Ready && health == RuntimeHealthy && readiness == RuntimeReady
}

///|
fn spawned_serving_ingress_shape_valid(
  server_count : Int,
  framed_pool_count : Int,
  openai_server_count : Int,
  openai_pool_count : Int,
) -> Bool {
  server_count + framed_pool_count == 1 &&
  openai_server_count == 0 &&
  openai_pool_count == 0
}

///|
fn spawned_serving_pool_live_state_valid(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
  pool_state : @online_tcp.LunaOnlineFramedConnectionPoolState,
  active_connections : Int,
) -> Bool {
  if phase != Ready ||
    health != RuntimeHealthy ||
    readiness != RuntimeReady ||
    active_connections < 0 {
    return false
  }
  match pool_state {
    LunaOnlineFramedConnectionPoolListening => active_connections == 0
    LunaOnlineFramedConnectionPoolActive => active_connections > 0
    LunaOnlineFramedConnectionPoolDraining
    | LunaOnlineFramedConnectionPoolFailed
    | LunaOnlineFramedConnectionPoolClosed => false
  }
}

///|
fn RuntimeInstanceOwner::spawned_serving_ingress_shape_valid(
  self : RuntimeInstanceOwner,
) -> Bool {
  spawned_serving_ingress_shape_valid(
    self.servers.length(),
    self.framed_pools.length(),
    self.openai_servers.length(),
    self.openai_pools.length(),
  )
}

///|
fn RuntimeInstanceOwner::spawned_serving_ingress_live(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.spawned_serving_ingress_shape_valid() {
    return false
  }
  if self.servers.length() == 1 {
    return spawned_serving_live_state_valid(
      self.phase(),
      self.health(),
      self.readiness(),
      self.servers[0].state(),
    )
  }
  spawned_serving_pool_live_state_valid(
    self.phase(),
    self.health(),
    self.readiness(),
    self.framed_pools[0].state(),
    self.framed_pools[0].active_connection_count(),
  )
}

///|
fn RuntimeInstanceOwner::spawned_serving_ingress_active(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.spawned_serving_ingress_live() {
    return false
  }
  if self.servers.length() == 1 {
    return self.servers[0].state() == LunaOnlineTcpServerConnected
  }
  self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolActive &&
  self.framed_pools[0].active_connection_count() == 1
}

///|
fn RuntimeInstanceOwner::spawned_serving_ingress_listening(
  self : RuntimeInstanceOwner,
) -> Bool {
  if !self.spawned_serving_ingress_live() {
    return false
  }
  if self.servers.length() == 1 {
    return self.servers[0].state() == LunaOnlineTcpServerListening
  }
  self.framed_pools[0].state() == LunaOnlineFramedConnectionPoolListening &&
  self.framed_pools[0].active_connection_count() == 0
}

///|
fn spawned_serving_terminal_ingress_state_valid(
  server_count : Int,
  framed_pool_count : Int,
  server_active : Bool,
  framed_pool_live : Bool,
) -> Bool {
  (server_count == 1 && framed_pool_count == 0 && server_active) ||
  (server_count == 0 && framed_pool_count == 1 && framed_pool_live)
}

///|
fn RuntimeInstanceOwner::spawned_serving_terminal_ingress_live(
  self : RuntimeInstanceOwner,
) -> Bool {
  spawned_serving_terminal_ingress_state_valid(
    self.servers.length(),
    self.framed_pools.length(),
    self.spawned_serving_ingress_active(),
    self.spawned_serving_ingress_live(),
  )
}

///|
fn RuntimeInstanceOwner::spawned_serving_terminal_disconnected(
  self : RuntimeInstanceOwner,
) -> Bool {
  self.servers.length() == 0 && self.framed_pools.length() == 1
}

///|
fn spawned_serving_live_state_valid(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
  server_state : @online_tcp.LunaOnlineTcpServerState,
) -> Bool {
  if phase != Ready || health != RuntimeHealthy {
    return false
  }
  match server_state {
    LunaOnlineTcpServerListening | LunaOnlineTcpServerAccepting =>
      readiness == RuntimeReady
    LunaOnlineTcpServerConnected => readiness == RuntimeNotReady
    LunaOnlineTcpServerRetiring => true
    LunaOnlineTcpServerDraining | LunaOnlineTcpServerClosed => false
  }
}

///|
fn spawned_serving_closed_state_valid(
  phase : RuntimeInstancePhase,
  health : RuntimeInstanceHealth,
  readiness : RuntimeInstanceReadiness,
  cleanup_complete : Bool,
) -> Bool {
  phase == Closed &&
  health == RuntimeClosed &&
  readiness == RuntimeNotReady &&
  cleanup_complete
}

///|
fn capture_spawned_serving_metrics(
  owner : RuntimeInstanceOwner,
) -> SpawnedServingMetricSample raise RuntimeSpawnedServingValidationError {
  let snapshot = owner.metrics_snapshot() catch {
    _ => raise spawned_serving_error(SpawnedServingMetrics)
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
    network_accepts: snapshot.counter_value(NetworkAccepts),
    network_disconnects: snapshot.counter_value(NetworkDisconnects),
    network_rejections: snapshot.counter_value(NetworkRejections),
    queue_depth: snapshot.gauge_value(QueueDepth),
    active_requests: snapshot.gauge_value(ActiveRequests),
    kv_pages_used: snapshot.gauge_value(KvPagesUsed),
    kv_pages_free: snapshot.gauge_value(KvPagesFree),
  } catch {
    _ => raise spawned_serving_error(SpawnedServingMetrics)
  }
}

///|
fn spawned_serving_initial_metrics_valid(
  sample : SpawnedServingMetricSample,
) -> Bool {
  sample.admissions == 0UL &&
  sample.completions == 0UL &&
  sample.cancellations == 0UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == 0UL &&
  sample.generated_tokens == 0UL &&
  sample.worker_failures == 0UL &&
  sample.network_accepts == 0UL &&
  sample.network_disconnects == 0UL &&
  sample.network_rejections == 0UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free > 0
}

///|
fn spawned_serving_terminal_metrics_valid(
  sample : SpawnedServingMetricSample,
  input_tokens : Int,
  initial_kv_pages_free : Int,
  disconnected : Bool,
) -> Bool {
  sample.admissions == 1UL &&
  sample.completions == 1UL &&
  sample.cancellations == 0UL &&
  sample.deadlines == 0UL &&
  sample.failures == 0UL &&
  sample.prompt_tokens == input_tokens.to_uint64() &&
  sample.generated_tokens == 2UL &&
  sample.worker_failures == 0UL &&
  sample.network_accepts == 1UL &&
  sample.network_disconnects == (if disconnected { 1UL } else { 0UL }) &&
  sample.network_rejections == 0UL &&
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free == initial_kv_pages_free
}

///|
fn require_spawned_serving_accepted(
  event : @inference.StreamEvent,
  contract : RuntimeSpawnedServingValidationContract,
) -> Unit raise RuntimeSpawnedServingValidationError {
  let valid = match event {
    Accepted(accepted) =>
      accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == 2 &&
      accepted.effective_limits().context_ceiling() == contract.context_ceiling
    _ => false
  }
  if !valid {
    raise spawned_serving_error(SpawnedServingAccepted)
  }
}

///|
fn require_spawned_serving_token(
  event : @inference.StreamEvent,
  expected_token : Int,
  expected_position : Int,
  rule : RuntimeSpawnedServingValidationRule,
) -> Int raise RuntimeSpawnedServingValidationError {
  match event {
    Token(token) if token.token_id() == expected_token &&
      token.position() == expected_position => token.token_id()
    _ => raise spawned_serving_error(rule)
  }
}

///|
fn spawned_serving_usage_valid(
  usage : @inference.Usage,
  input_tokens : Int,
) -> Bool {
  usage.input_tokens() == input_tokens &&
  usage.cached_input_tokens() == 0 &&
  usage.output_tokens() == 2 &&
  usage.total_tokens() == input_tokens + 2
}

///|
fn require_spawned_serving_usage(
  event : @inference.StreamEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedServingValidationError {
  match event {
    Usage(usage) if spawned_serving_usage_valid(usage, input_tokens) => ()
    _ => raise spawned_serving_error(SpawnedServingUsage)
  }
}

///|
fn require_spawned_serving_completed(
  event : @inference.StreamEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedServingValidationError {
  match event {
    Completed(completed) if completed.finish_reason() == TokenLimit &&
      spawned_serving_usage_valid(completed.final_usage(), input_tokens) => ()
    _ => raise spawned_serving_error(SpawnedServingCompleted)
  }
}
