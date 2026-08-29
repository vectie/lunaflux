///|
priv enum SpawnedContextChurnOutcome {
  SpawnedContextChurnSucceeded(RuntimeSpawnedContextChurnResult)
  SpawnedContextChurnRejected(RuntimeSpawnedContextChurnRule)
}

///|
fn offer_context_churn_frame(
  stream : @online_session.LunaOnlineFramedStream,
  frame : FixedArray[Byte],
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedContextChurnError {
  offer_spawned_validation_frame(stream, frame, turn_limit) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnIngress)
  }
}

///|
fn next_context_churn_event(
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> @online_session.LunaOnlineFramedSemanticEvent raise RuntimeSpawnedContextChurnError {
  next_spawned_validation_event(stream, turn_limit) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}

///|
fn retire_context_churn_stream(
  service : @online_session.LunaOnlineFramedService,
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedContextChurnError {
  if !retire_spawned_direct_stream(service, stream, turn_limit) {
    raise spawned_context_churn_error(SpawnedContextChurnCleanup)
  }
}

///|
fn run_one_context_churn_request(
  service : @online_session.LunaOnlineFramedService,
  frame : FixedArray[Byte],
  context_ceiling : Int,
  contract : RuntimeSpawnedContextChurnContract,
  previous_plan_sequence : UInt64,
  total_kv_pages : Int,
) -> UInt64 raise RuntimeSpawnedContextChurnError {
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    raise spawned_context_churn_error(SpawnedContextChurnOwner)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnOwner)
  }
  offer_context_churn_frame(stream, frame, contract.turn_limit)
  require_context_churn_accepted(
    next_context_churn_event(stream, contract.turn_limit),
    contract,
    context_ceiling,
  )
  let prefill = require_context_churn_token(
    service,
    next_context_churn_event(stream, contract.turn_limit),
    contract.expected_first_token,
    0,
    previous_plan_sequence,
    total_kv_pages,
  )
  let decode = require_context_churn_token(
    service,
    next_context_churn_event(stream, contract.turn_limit),
    contract.expected_second_token,
    1,
    prefill,
    total_kv_pages,
  )
  require_context_churn_usage(
    next_context_churn_event(stream, contract.turn_limit),
    contract.input_tokens,
  )
  require_context_churn_completed(
    next_context_churn_event(stream, contract.turn_limit),
    contract.input_tokens,
  )
  retire_context_churn_stream(service, stream, contract.turn_limit)
  decode
}

///|
fn run_spawned_context_churn_service(
  owner : RuntimeInstanceOwner,
  service : @online_session.LunaOnlineFramedService,
  contract : RuntimeSpawnedContextChurnContract,
) -> RuntimeSpawnedContextChurnResult raise RuntimeSpawnedContextChurnError {
  if !spawned_direct_validation_owner_valid(owner) {
    raise spawned_context_churn_error(SpawnedContextChurnOwner)
  }
  let initial = capture_spawned_direct_resources(service) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnMetrics)
  }
  if !spawned_direct_resources_valid(initial, initial.kv_pages_free, 0UL) ||
    initial.kv_pages_free <= 0 {
    raise spawned_context_churn_error(SpawnedContextChurnMetrics)
  }
  let initial_plan = service.luna_plan_telemetry() catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnPlan)
  }
  if initial_plan.last_plan_sequence() != 0UL ||
    initial_plan.last_plan_rows() != 0 ||
    initial_plan.last_plan_tokens() != 0 {
    raise spawned_context_churn_error(SpawnedContextChurnPlan)
  }
  let mut last_plan_sequence = 0UL
  for index in 0..<contract.frames.length() {
    last_plan_sequence = run_one_context_churn_request(
      service,
      contract.frames[index],
      contract.context_ceilings[index],
      contract,
      last_plan_sequence,
      initial.kv_pages_free,
    )
    let completed_requests = index + 1
    let after = capture_spawned_direct_resources(service) catch {
      _ => raise spawned_context_churn_error(SpawnedContextChurnMetrics)
    }
    if !spawned_direct_resources_valid(
        after,
        initial.kv_pages_free,
        (completed_requests * contract.input_tokens).to_uint64(),
      ) ||
      last_plan_sequence != (completed_requests * 2).to_uint64() {
      raise spawned_context_churn_error(SpawnedContextChurnMetrics)
    }
  }
  let terminal = capture_spawned_direct_resources(service) catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnMetrics)
  }
  {
    request_count: contract.frames.length(),
    event_count: contract.frames.length() * 5,
    plan_count: contract.frames.length() * 2,
    last_plan_sequence,
    initial_kv_pages_free: initial.kv_pages_free,
    final_kv_pages_free: terminal.kv_pages_free,
    network_accepts: 0UL,
    network_disconnects: 0UL,
    network_rejections: 0UL,
  }
}

///|
/// Runs three sequential requests through one authenticated child and closes
/// the complete runtime owner. Repeating fresh-child cycles remains a campaign
/// concern so this method never owns deployment or restart policy.
pub fn RuntimeInstanceOwner::validate_spawned_context_churn(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedContextChurnContract,
) -> RuntimeSpawnedContextChurnResult raise RuntimeSpawnedContextChurnError {
  self.require_spawned_physical_handoff() catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnOwner)
  }
  let service = self.preparations[0].take_ready() catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnOwner)
  }
  self.preparations.clear()
  self.services.push(service)
  let outcome = try
    run_spawned_context_churn_service(self, service, contract)
  catch {
    RuntimeSpawnedContextChurnFailed(rule) => SpawnedContextChurnRejected(rule)
  } noraise {
    result => SpawnedContextChurnSucceeded(result)
  }
  let cleaned = close_spawned_validation_owner(self, contract.turn_limit)
  if !cleaned {
    raise spawned_context_churn_error(SpawnedContextChurnCleanup)
  }
  match outcome {
    SpawnedContextChurnSucceeded(result) => result
    SpawnedContextChurnRejected(rule) => raise spawned_context_churn_error(rule)
  }
}
