///|
priv enum SpawnedLongContextOutcome {
  SpawnedLongContextSucceeded(RuntimeSpawnedLongContextResult)
  SpawnedLongContextRejected(RuntimeSpawnedLongContextRule)
}

///|
fn offer_long_context_frame(
  stream : @online_session.LunaOnlineFramedStream,
  frame : FixedArray[Byte],
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedLongContextError {
  offer_spawned_validation_frame(stream, frame, turn_limit) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextIngress)
  }
}

///|
fn next_long_context_event(
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> @online_session.LunaOnlineFramedSemanticEvent raise RuntimeSpawnedLongContextError {
  next_spawned_validation_event(stream, turn_limit) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}

///|
fn run_one_long_context_request(
  service : @online_session.LunaOnlineFramedService,
  frame : FixedArray[Byte],
  contract : RuntimeSpawnedLongContextContract,
  expected_token : Int,
  input_tokens : Int,
  previous_plan_sequence : UInt64,
  total_kv_pages : Int,
) -> (Int, UInt64) raise RuntimeSpawnedLongContextError {
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    raise spawned_long_context_error(SpawnedLongContextOwner)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextOwner)
  }
  offer_long_context_frame(stream, frame, contract.turn_limit)
  require_long_context_accepted(
    next_long_context_event(stream, contract.turn_limit),
    contract,
  )
  let token = require_long_context_token(
    service,
    next_long_context_event(stream, contract.turn_limit),
    expected_token,
    previous_plan_sequence,
    input_tokens,
    total_kv_pages,
    (input_tokens + contract.tokens_per_page - 1) / contract.tokens_per_page,
  )
  require_long_context_usage(
    next_long_context_event(stream, contract.turn_limit),
    input_tokens,
  )
  require_long_context_completed(
    next_long_context_event(stream, contract.turn_limit),
    input_tokens,
  )
  if !retire_spawned_direct_stream(service, stream, contract.turn_limit) {
    raise spawned_long_context_error(SpawnedLongContextCleanup)
  }
  token
}

///|
fn run_spawned_long_context_service(
  owner : RuntimeInstanceOwner,
  service : @online_session.LunaOnlineFramedService,
  contract : RuntimeSpawnedLongContextContract,
) -> RuntimeSpawnedLongContextResult raise RuntimeSpawnedLongContextError {
  if !spawned_direct_validation_owner_valid(owner) {
    raise spawned_long_context_error(SpawnedLongContextOwner)
  }
  let initial = capture_spawned_direct_resources(service) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  if initial.kv_pages_free <= 1 ||
    !spawned_direct_resources_valid(initial, initial.kv_pages_free, 0UL) {
    raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  let initial_plan = service.luna_plan_telemetry() catch {
    _ => raise spawned_long_context_error(SpawnedLongContextPlan)
  }
  if initial_plan.last_plan_sequence() != 0UL ||
    initial_plan.last_plan_rows() != 0 ||
    initial_plan.last_plan_tokens() != 0 {
    raise spawned_long_context_error(SpawnedLongContextPlan)
  }
  let first = run_one_long_context_request(
    service,
    contract.first_frame,
    contract,
    contract.expected_first_token,
    8,
    0UL,
    initial.kv_pages_free,
  )
  let after_first = capture_spawned_direct_resources(service) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  if first.1 != 8UL ||
    !spawned_direct_resources_valid(after_first, initial.kv_pages_free, 8UL) {
    raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  let second = run_one_long_context_request(
    service,
    contract.second_frame,
    contract,
    contract.expected_second_token,
    9,
    first.1,
    initial.kv_pages_free,
  )
  let terminal = capture_spawned_direct_resources(service) catch {
    _ => raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  if second.1 != 17UL ||
    !spawned_direct_resources_valid(terminal, initial.kv_pages_free, 17UL) {
    raise spawned_long_context_error(SpawnedLongContextTelemetry)
  }
  {
    first_token: first.0,
    second_token: second.0,
    request_count: 2,
    event_count: 8,
    plan_count: 17,
    last_plan_sequence: second.1,
    input_tokens_processed: 17,
    initial_kv_pages_free: initial.kv_pages_free,
    final_kv_pages_free: terminal.kv_pages_free,
  }
}

///|
/// Executes the independently refereed 8-token and 9-token disabled-cache
/// prompts through one physical child, then drains and reaps the whole owner.
pub fn RuntimeInstanceOwner::validate_spawned_long_context(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedLongContextContract,
) -> RuntimeSpawnedLongContextResult raise RuntimeSpawnedLongContextError {
  self.require_spawned_physical_handoff() catch {
    _ => raise spawned_long_context_error(SpawnedLongContextOwner)
  }
  let service = self.preparations[0].take_ready() catch {
    _ => raise spawned_long_context_error(SpawnedLongContextOwner)
  }
  self.preparations.clear()
  self.services.push(service)
  let outcome = try
    run_spawned_long_context_service(self, service, contract)
  catch {
    RuntimeSpawnedLongContextFailed(rule) => SpawnedLongContextRejected(rule)
  } noraise {
    result => SpawnedLongContextSucceeded(result)
  }
  let cleaned = close_spawned_validation_owner(self, contract.turn_limit)
  if !cleaned {
    raise spawned_long_context_error(SpawnedLongContextCleanup)
  }
  match outcome {
    SpawnedLongContextSucceeded(result) => result
    SpawnedLongContextRejected(rule) => raise spawned_long_context_error(rule)
  }
}
