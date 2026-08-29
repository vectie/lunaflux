///|
priv enum SpawnedValidationOutcome {
  SpawnedValidationSucceeded(RuntimeSpawnedValidationResult)
  SpawnedValidationRejectedOutcome(RuntimeSpawnedValidationRule)
}

///|
fn run_spawned_validation_service(
  service : @online_session.LunaOnlineFramedService,
  contract : RuntimeSpawnedValidationContract,
) -> RuntimeSpawnedValidationResult raise RuntimeSpawnedValidationError {
  let initial = service.luna_telemetry() catch {
    _ => raise spawned_validation_error(SpawnedValidationOwner)
  }
  let initial_plan = service.luna_plan_telemetry() catch {
    _ => raise spawned_validation_error(SpawnedValidationOwner)
  }
  if contract.input_tokens + 2 > contract.tokens_per_page ||
    initial.queue_depth() != 0 ||
    initial.active_requests() != 0 ||
    initial.kv_pages_used() != 0 ||
    initial.kv_pages_free() <= 0 ||
    initial_plan.last_plan_sequence() != 0UL ||
    initial_plan.last_plan_rows() != 0 ||
    initial_plan.last_plan_tokens() != 0 {
    raise spawned_validation_error(SpawnedValidationOwner)
  }
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    raise spawned_validation_error(SpawnedValidationOwner)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => raise spawned_validation_error(SpawnedValidationOwner)
  }
  offer_spawned_validation_frame(stream, contract.frame, contract.turn_limit)
  require_spawned_accepted(
    next_spawned_validation_event(stream, contract.turn_limit),
    contract,
  )
  let (first_token, prefill) = require_spawned_token(
    service,
    next_spawned_validation_event(stream, contract.turn_limit),
    contract.expected_first_token,
    0,
    0UL,
    initial.kv_pages_free(),
  )
  let prefill_graph = service.luna_graph_runtime_telemetry() catch {
    _ => raise spawned_validation_error(SpawnedValidationPlan)
  }
  if prefill_graph.sequence_value() != prefill ||
    prefill_graph.graph_hits() + prefill_graph.graph_misses() != 1UL {
    raise spawned_validation_error(SpawnedValidationPlan)
  }
  let (second_token, decode) = require_spawned_token(
    service,
    next_spawned_validation_event(stream, contract.turn_limit),
    contract.expected_second_token,
    1,
    prefill,
    initial.kv_pages_free(),
  )
  let decode_graph = service.luna_graph_runtime_telemetry() catch {
    _ => raise spawned_validation_error(SpawnedValidationPlan)
  }
  if decode_graph.sequence_value() != decode ||
    decode_graph.selected_path() != prefill_graph.selected_path() ||
    decode_graph.graph_hits() + decode_graph.graph_misses() != 2UL {
    raise spawned_validation_error(SpawnedValidationPlan)
  }
  if prefill == 0xffffffffffffffffUL || decode != prefill + 1UL {
    raise spawned_validation_error(SpawnedValidationPlan)
  }
  require_spawned_usage(
    next_spawned_validation_event(stream, contract.turn_limit),
    contract.input_tokens,
  )
  require_spawned_completed(
    next_spawned_validation_event(stream, contract.turn_limit),
    contract.input_tokens,
  )
  stream.disconnect() catch {
    _ => raise spawned_validation_error(SpawnedValidationCleanup)
  }
  {
    first_token,
    second_token,
    prefill_plan_sequence: prefill,
    decode_plan_sequence: decode,
    kv_page_count: 1,
    graph_captured: decode_graph.selected_path() == WorkerGraphCaptured,
    prefill_graph_hits: prefill_graph.graph_hits(),
    prefill_graph_misses: prefill_graph.graph_misses(),
    decode_graph_hits: decode_graph.graph_hits(),
    decode_graph_misses: decode_graph.graph_misses(),
  }
}

///|
fn RuntimeInstanceOwner::spawned_execution_expectation_matches(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedValidationContract,
) -> Bool {
  let sampling_matches = match contract.sampling_expectation {
    AnySpawnedSamplingRuntime => true
    HostSamplingRuntimeExpected =>
      self.release_admission.matches_spawned_execution_extensions(
        true,
        contract.require_fused_runtime,
      )
    EmbeddedCudaGreedyExpected =>
      self.release_admission.matches_spawned_execution_extensions(
        false,
        contract.require_fused_runtime,
      )
  }
  sampling_matches
}

///|
fn close_spawned_validation_owner(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Bool {
  if owner.services.length() != 1 {
    owner.phase = Failed
    return false
  }
  let service = owner.services[0]
  service.begin_drain()
  for _ in 0..<turn_limit {
    if service.is_closed() {
      owner.services.clear()
      owner.phase = Closed
      return owner.cleanup_complete()
    }
    ignore(service.progress() catch { _ => LunaOnlineFramedServiceDraining })
  }
  if service.is_closed() {
    owner.services.clear()
    owner.phase = Closed
    owner.cleanup_complete()
  } else {
    owner.phase = Failed
    false
  }
}

///|
/// Executes one exact, predeclared two-token correctness request through the
/// child-backed framed service without binding a listener. The method returns
/// no service, scheduler, worker, device, logits, or mutable owner authority.
/// Success is published only after deterministic stream disconnect, service
/// drain, device/resource close, and child reap.
pub fn RuntimeInstanceOwner::validate_spawned_physical_request(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedValidationContract,
) -> RuntimeSpawnedValidationResult raise RuntimeSpawnedValidationError {
  self.require_spawned_physical_handoff() catch {
    _ => raise spawned_validation_error(SpawnedValidationOwner)
  }
  if !self.spawned_execution_expectation_matches(contract) {
    raise spawned_validation_error(SpawnedValidationContract)
  }
  let service = self.preparations[0].take_ready() catch {
    _ => raise spawned_validation_error(SpawnedValidationOwner)
  }
  self.preparations.clear()
  self.services.push(service)
  let outcome = try run_spawned_validation_service(service, contract) catch {
    RuntimeSpawnedValidationFailed(rule) =>
      SpawnedValidationRejectedOutcome(rule)
  } noraise {
    result => SpawnedValidationSucceeded(result)
  }
  let cleaned = close_spawned_validation_owner(self, contract.turn_limit)
  if !cleaned {
    raise spawned_validation_error(SpawnedValidationCleanup)
  }
  match outcome {
    SpawnedValidationSucceeded(result) => result
    SpawnedValidationRejectedOutcome(rule) =>
      raise spawned_validation_error(rule)
  }
}
