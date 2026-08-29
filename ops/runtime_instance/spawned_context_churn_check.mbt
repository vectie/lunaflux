///|
fn require_context_churn_event_view(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> @luna_event.LunaEventView raise RuntimeSpawnedContextChurnError {
  event.view() catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}

///|
fn finish_context_churn_event(
  event : @online_session.LunaOnlineFramedSemanticEvent,
) -> Unit raise RuntimeSpawnedContextChurnError {
  event.delivered() catch {
    _ => raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}

///|
fn require_context_churn_accepted(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  contract : RuntimeSpawnedContextChurnContract,
  context_ceiling : Int,
) -> Unit raise RuntimeSpawnedContextChurnError {
  let view = require_context_churn_event_view(event)
  let valid = try {
    if view.kind() != LunaAcceptedEventKind {
      false
    } else {
      let accepted = view.accepted()
      accepted.model_identity() == contract.expected_model &&
      accepted.effective_limits().max_new_tokens() == 2 &&
      accepted.effective_limits().context_ceiling() == context_ceiling
    }
  } catch {
    _ => false
  }
  finish_context_churn_event(event)
  if !valid {
    raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}

///|
fn require_context_churn_token(
  service : @online_session.LunaOnlineFramedService,
  event : @online_session.LunaOnlineFramedSemanticEvent,
  expected_token : Int,
  expected_position : Int,
  previous_plan_sequence : UInt64,
  total_kv_pages : Int,
) -> UInt64 raise RuntimeSpawnedContextChurnError {
  let view = require_context_churn_event_view(event)
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
      finish_context_churn_event(event) catch {
        _ => ()
      }
      raise spawned_context_churn_error(SpawnedContextChurnPlan)
    }
  }
  let telemetry = service.luna_telemetry() catch {
    _ => {
      finish_context_churn_event(event) catch {
        _ => ()
      }
      raise spawned_context_churn_error(SpawnedContextChurnMetrics)
    }
  }
  let plan_sequence = plan.last_plan_sequence()
  let valid = observed == (expected_token, expected_position) &&
    plan_sequence == previous_plan_sequence + 1UL &&
    plan.last_plan_rows() == 1 &&
    plan.last_plan_tokens() == 1 &&
    spawned_validation_kv_state_valid(
      telemetry.active_requests(),
      telemetry.kv_pages_used(),
      telemetry.kv_pages_free(),
      total_kv_pages,
    )
  finish_context_churn_event(event)
  if !valid {
    raise spawned_context_churn_error(SpawnedContextChurnPlan)
  }
  plan_sequence
}

///|
fn require_context_churn_usage(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedContextChurnError {
  let view = require_context_churn_event_view(event)
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
  finish_context_churn_event(event)
  if !valid {
    raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}

///|
fn require_context_churn_completed(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
) -> Unit raise RuntimeSpawnedContextChurnError {
  let view = require_context_churn_event_view(event)
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
  finish_context_churn_event(event)
  if !valid {
    raise spawned_context_churn_error(SpawnedContextChurnEvent)
  }
}
