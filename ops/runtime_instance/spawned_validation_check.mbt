///|
fn spawned_validation_token_matches(
  observed_token : Int,
  observed_position : Int,
  expected_token : Int,
  expected_position : Int,
) -> Bool {
  observed_token == expected_token && observed_position == expected_position
}

///|
fn spawned_validation_kv_state_valid(
  active_requests : Int,
  kv_pages_used : Int,
  kv_pages_free : Int,
  total_kv_pages : Int,
) -> Bool {
  // Scheduler telemetry is sampled after the semantic event was prepared.
  // Depending on event-consumption timing, it exposes either the single live
  // page or the fully retired request with the original free-page balance.
  (
    active_requests == 1 &&
    kv_pages_used == 1 &&
    kv_pages_free == total_kv_pages - 1
  ) ||
  (
    active_requests == 0 &&
    kv_pages_used == 0 &&
    kv_pages_free == total_kv_pages
  )
}

///|
fn require_spawned_accepted(
  semantic : @online_session.LunaOnlineFramedSemanticEvent,
  contract : RuntimeSpawnedValidationContract,
) -> Unit raise RuntimeSpawnedValidationError {
  let view = deliver_spawned_validation_event(semantic)
  let valid = try {
    if view.kind() != LunaAcceptedEventKind {
      false
    } else {
      let accepted = view.accepted()
      accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == 2 &&
      accepted.effective_limits().context_ceiling() == contract.context_ceiling
    }
  } catch {
    _ => false
  }
  finish_spawned_validation_event(semantic)
  if !valid {
    raise spawned_validation_error(SpawnedValidationEvent)
  }
}

///|
fn require_spawned_token(
  service : @online_session.LunaOnlineFramedService,
  semantic : @online_session.LunaOnlineFramedSemanticEvent,
  expected_token : Int,
  expected_position : Int,
  previous_plan_sequence : UInt64,
  total_kv_pages : Int,
) -> (Int, UInt64) raise RuntimeSpawnedValidationError {
  let view = deliver_spawned_validation_event(semantic)
  let observed = try {
    if view.kind() != LunaTokenEventKind {
      (-1, -1)
    } else {
      let token = view.token()
      (token.token_id(), token.position())
    }
  } catch {
    _ => (-1, -1)
  }
  let plan = service.luna_plan_telemetry() catch {
    _ => {
      finish_spawned_validation_event(semantic) catch {
        _ => ()
      }
      raise spawned_validation_error(SpawnedValidationPlan)
    }
  }
  let plan_sequence = plan.last_plan_sequence()
  let plan_valid = plan_sequence > previous_plan_sequence &&
    plan.last_plan_rows() == 1 &&
    plan.last_plan_tokens() == 1
  let telemetry = service.luna_telemetry() catch {
    _ => {
      finish_spawned_validation_event(semantic) catch {
        _ => ()
      }
      raise spawned_validation_error(SpawnedValidationKv)
    }
  }
  let kv_valid = spawned_validation_kv_state_valid(
    telemetry.active_requests(),
    telemetry.kv_pages_used(),
    telemetry.kv_pages_free(),
    total_kv_pages,
  )
  finish_spawned_validation_event(semantic)
  if !spawned_validation_token_matches(
      observed.0,
      observed.1,
      expected_token,
      expected_position,
    ) {
    raise spawned_validation_error(SpawnedValidationEvent)
  }
  if !plan_valid {
    raise spawned_validation_error(SpawnedValidationPlan)
  }
  if !kv_valid {
    raise spawned_validation_error(SpawnedValidationKv)
  }
  (observed.0, plan_sequence)
}

///|
fn require_spawned_usage(
  semantic : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedValidationError {
  let view = deliver_spawned_validation_event(semantic)
  let valid = try {
    if view.kind() != LunaUsageEventKind {
      false
    } else {
      let usage = view.usage()
      usage.input_tokens() == input_tokens &&
      usage.cached_input_tokens() == 0 &&
      usage.output_tokens() == 2 &&
      usage.total_tokens() == input_tokens + 2
    }
  } catch {
    _ => false
  }
  finish_spawned_validation_event(semantic)
  if !valid {
    raise spawned_validation_error(SpawnedValidationEvent)
  }
}

///|
fn require_spawned_completed(
  semantic : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedValidationError {
  let view = deliver_spawned_validation_event(semantic)
  let valid = try {
    if view.kind() != LunaCompletedEventKind {
      false
    } else {
      let completed = view.completed()
      completed.finish_reason() == TokenLimit &&
      completed.input_tokens() == input_tokens &&
      completed.cached_input_tokens() == 0 &&
      completed.output_tokens() == 2 &&
      completed.total_tokens() == input_tokens + 2
    }
  } catch {
    _ => false
  }
  finish_spawned_validation_event(semantic)
  if !valid {
    raise spawned_validation_error(SpawnedValidationEvent)
  }
}
