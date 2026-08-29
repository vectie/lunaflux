///|
priv enum SpawnedPrefixValidationOutcome {
  SpawnedPrefixValidationSucceeded(RuntimeSpawnedPrefixValidationResult)
  SpawnedPrefixValidationRejectedOutcome(RuntimeSpawnedPrefixValidationRule)
}

///|
fn offer_spawned_prefix_frame(
  stream : @online_session.LunaOnlineFramedStream,
  frame : FixedArray[Byte],
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedPrefixValidationError {
  offer_spawned_validation_frame(stream, frame, turn_limit) catch {
    RuntimeSpawnedValidationFailed(SpawnedValidationRejected) =>
      raise spawned_prefix_error(SpawnedPrefixRejected)
    _ => raise spawned_prefix_error(SpawnedPrefixIngress)
  }
}

///|
fn next_spawned_prefix_event(
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> @online_session.LunaOnlineFramedSemanticEvent raise RuntimeSpawnedPrefixValidationError {
  next_spawned_validation_event(stream, turn_limit) catch {
    RuntimeSpawnedValidationFailed(SpawnedValidationRejected) =>
      raise spawned_prefix_error(SpawnedPrefixRejected)
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
}

///|
fn retire_spawned_prefix_stream(
  service : @online_session.LunaOnlineFramedService,
  stream : @online_session.LunaOnlineFramedStream,
  turn_limit : Int,
) -> Unit raise RuntimeSpawnedPrefixValidationError {
  stream.disconnect() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixCleanup)
  }
  for _ in 0..<turn_limit {
    if service.readiness() == LunaOnlineFramedServiceReady {
      return
    }
    try stream.progress() catch {
      _ => {
        if service.readiness() == LunaOnlineFramedServiceReady {
          return
        }
        raise spawned_prefix_error(SpawnedPrefixCleanup)
      }
    } noraise {
      _ => ()
    }
  }
  if service.readiness() != LunaOnlineFramedServiceReady {
    raise spawned_prefix_error(SpawnedPrefixCleanup)
  }
}

///|
fn run_one_spawned_prefix_request(
  service : @online_session.LunaOnlineFramedService,
  frame : FixedArray[Byte],
  contract : RuntimeSpawnedPrefixValidationContract,
  input_tokens : Int,
  cached_input_tokens : Int,
  expected_token : Int,
  previous_plan_sequence : UInt64,
  exact_plan_tokens : Int,
) -> (Int, UInt64) raise RuntimeSpawnedPrefixValidationError {
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    raise spawned_prefix_error(SpawnedPrefixOwner)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => raise spawned_prefix_error(SpawnedPrefixOwner)
  }
  offer_spawned_prefix_frame(stream, frame, contract.turn_limit)
  require_spawned_prefix_accepted(
    next_spawned_prefix_event(stream, contract.turn_limit),
    contract,
  )
  let token = require_spawned_prefix_token(
    service,
    next_spawned_prefix_event(stream, contract.turn_limit),
    expected_token,
    previous_plan_sequence,
    exact_plan_tokens,
  )
  require_spawned_prefix_usage(
    next_spawned_prefix_event(stream, contract.turn_limit),
    input_tokens,
    cached_input_tokens,
  )
  require_spawned_prefix_completed(
    next_spawned_prefix_event(stream, contract.turn_limit),
    input_tokens,
    cached_input_tokens,
  )
  retire_spawned_prefix_stream(service, stream, contract.turn_limit)
  token
}

///|
fn run_spawned_prefix_validation_service(
  service : @online_session.LunaOnlineFramedService,
  contract : RuntimeSpawnedPrefixValidationContract,
) -> RuntimeSpawnedPrefixValidationResult raise RuntimeSpawnedPrefixValidationError {
  let initial = capture_spawned_prefix_telemetry(service)
  if !spawned_prefix_initial_telemetry_valid(initial) {
    raise spawned_prefix_error(SpawnedPrefixTelemetry)
  }
  let page_count = contract.expected_reused_tokens / contract.tokens_per_page
  if page_count <= 0 || page_count >= initial.kv_pages_free {
    raise spawned_prefix_error(SpawnedPrefixContract)
  }
  let first = run_one_spawned_prefix_request(
    service,
    contract.first_frame,
    contract,
    contract.first_input_tokens,
    0,
    contract.expected_first_token,
    0UL,
    0,
  )
  let after_first = capture_spawned_prefix_telemetry(service)
  if !spawned_prefix_after_first_valid(
      after_first,
      initial.kv_pages_free,
      contract.first_input_tokens,
      page_count,
    ) {
    raise spawned_prefix_error(SpawnedPrefixTelemetry)
  }
  let second = run_one_spawned_prefix_request(
    service,
    contract.second_frame,
    contract,
    contract.second_input_tokens,
    contract.expected_reused_tokens,
    contract.expected_second_token,
    first.1,
    1,
  )
  let after_second = capture_spawned_prefix_telemetry(service)
  if !spawned_prefix_after_second_valid(
      after_second,
      initial.kv_pages_free,
      contract.first_input_tokens,
      contract.second_input_tokens,
      contract.expected_reused_tokens,
      page_count,
    ) {
    raise spawned_prefix_error(SpawnedPrefixTelemetry)
  }
  {
    first_token: first.0,
    second_token: second.0,
    first_terminal_plan_sequence: first.1,
    second_terminal_plan_sequence: second.1,
    prefix_lookups: after_second.prefix_lookups,
    prefix_hits: after_second.prefix_hits,
    prefix_misses: after_second.prefix_misses,
    prefix_publications: after_second.prefix_publications,
    reused_tokens: after_second.prefix_tokens_reused,
    retained_prefix_pages: after_second.prefix_pages,
    initial_kv_pages_free: initial.kv_pages_free,
    kv_pages_free: after_second.kv_pages_free,
  }
}

///|
/// Runs the exact write-then-read two-request prefix qualification through the
/// already prepared child-backed service. It creates no scheduler, worker, or
/// device owner and returns only after deterministic service/resource cleanup.
pub fn RuntimeInstanceOwner::validate_spawned_prefix_reuse(
  self : RuntimeInstanceOwner,
  contract : RuntimeSpawnedPrefixValidationContract,
) -> RuntimeSpawnedPrefixValidationResult raise RuntimeSpawnedPrefixValidationError {
  self.require_spawned_physical_handoff() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixOwner)
  }
  let service = self.preparations[0].take_ready() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixOwner)
  }
  self.preparations.clear()
  self.services.push(service)
  let outcome = try
    run_spawned_prefix_validation_service(service, contract)
  catch {
    RuntimeSpawnedPrefixValidationFailed(rule) =>
      SpawnedPrefixValidationRejectedOutcome(rule)
  } noraise {
    result => SpawnedPrefixValidationSucceeded(result)
  }
  let cleaned = close_spawned_validation_owner(self, contract.turn_limit)
  if !cleaned {
    raise spawned_prefix_error(SpawnedPrefixCleanup)
  }
  match outcome {
    SpawnedPrefixValidationSucceeded(result) => result
    SpawnedPrefixValidationRejectedOutcome(rule) =>
      raise spawned_prefix_error(rule)
  }
}
