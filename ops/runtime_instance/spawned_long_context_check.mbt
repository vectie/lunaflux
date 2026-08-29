///|
fn spawned_long_context_kv_valid(
  active_requests : Int,
  kv_pages_used : Int,
  kv_pages_free : Int,
  total_kv_pages : Int,
  expected_live_pages : Int,
) -> Bool {
  (
    active_requests == 1 &&
    kv_pages_used == expected_live_pages &&
    kv_pages_free == total_kv_pages - expected_live_pages
  ) ||
  (
    active_requests == 0 &&
    kv_pages_used == 0 &&
    kv_pages_free == total_kv_pages
  )
}

///|
fn long_context_event_view(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> @luna_event.LunaEventView raise RuntimeSpawnedLongContextError {
  event.view() catch {
    _ => raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}

///|
fn finish_long_context_event(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> Unit raise RuntimeSpawnedLongContextError {
  event.delivered() catch {
    _ => raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}

///|
fn require_long_context_accepted(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  contract : RuntimeSpawnedLongContextContract,
) -> Unit raise RuntimeSpawnedLongContextError {
  let view = long_context_event_view(event)
  let valid = try {
    if view.kind() != LunaAcceptedEventKind {
      false
    } else {
      let accepted = view.accepted()
      accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == 1 &&
      accepted.effective_limits().context_ceiling() == contract.context_ceiling
    }
  } catch {
    _ => false
  }
  finish_long_context_event(event)
  if !valid {
    raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}

///|
fn require_long_context_token(
  service : @online_session.LunaOnlineFramedService,
  event : @online_session.LunaOnlineFramedSemanticEvent,
  expected_token : Int,
  previous_plan_sequence : UInt64,
  input_tokens : Int,
  total_kv_pages : Int,
  expected_live_pages : Int,
) -> (Int, UInt64) raise RuntimeSpawnedLongContextError {
  let view = long_context_event_view(event)
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
      finish_long_context_event(event) catch {
        _ => ()
      }
      raise spawned_long_context_error(SpawnedLongContextPlan)
    }
  }
  let telemetry = service.luna_telemetry() catch {
    _ => {
      finish_long_context_event(event) catch {
        _ => ()
      }
      raise spawned_long_context_error(SpawnedLongContextKv)
    }
  }
  let plan_sequence = plan.last_plan_sequence()
  let valid = observed == (expected_token, 0) &&
    plan_sequence == previous_plan_sequence + input_tokens.to_uint64() &&
    plan.last_plan_rows() == 1 &&
    plan.last_plan_tokens() == 1 &&
    spawned_long_context_kv_valid(
      telemetry.active_requests(),
      telemetry.kv_pages_used(),
      telemetry.kv_pages_free(),
      total_kv_pages,
      expected_live_pages,
    )
  finish_long_context_event(event)
  if !valid {
    raise spawned_long_context_error(SpawnedLongContextPlan)
  }
  (observed.0, plan_sequence)
}

///|
fn require_long_context_usage(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedLongContextError {
  let view = long_context_event_view(event)
  let valid = try {
    if view.kind() != LunaUsageEventKind {
      false
    } else {
      let usage = view.usage()
      usage.input_tokens() == input_tokens &&
      usage.cached_input_tokens() == 0 &&
      usage.output_tokens() == 1 &&
      usage.total_tokens() == input_tokens + 1
    }
  } catch {
    _ => false
  }
  finish_long_context_event(event)
  if !valid {
    raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}

///|
fn require_long_context_completed(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedLongContextError {
  let view = long_context_event_view(event)
  let valid = try {
    if view.kind() != LunaCompletedEventKind {
      false
    } else {
      let completed = view.completed()
      completed.finish_reason() == TokenLimit &&
      completed.input_tokens() == input_tokens &&
      completed.cached_input_tokens() == 0 &&
      completed.output_tokens() == 1 &&
      completed.total_tokens() == input_tokens + 1
    }
  } catch {
    _ => false
  }
  finish_long_context_event(event)
  if !valid {
    raise spawned_long_context_error(SpawnedLongContextEvent)
  }
}
