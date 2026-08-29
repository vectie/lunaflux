///|
priv struct SpawnedPrefixTelemetrySample {
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
  prefix_lookups : UInt64
  prefix_hits : UInt64
  prefix_misses : UInt64
  prefix_publications : UInt64
  prefix_tokens_reused : UInt64
  prefix_tokens_computed : UInt64
  prefix_entries : Int
  prefix_pages : Int
}

///|
fn capture_spawned_prefix_telemetry(
  service : @online_session.LunaOnlineFramedService,
) -> SpawnedPrefixTelemetrySample raise RuntimeSpawnedPrefixValidationError {
  let observed = service.luna_telemetry() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixTelemetry)
  }
  {
    queue_depth: observed.queue_depth(),
    active_requests: observed.active_requests(),
    kv_pages_used: observed.kv_pages_used(),
    kv_pages_free: observed.kv_pages_free(),
    prefix_lookups: observed.prefix_lookups(),
    prefix_hits: observed.prefix_hits(),
    prefix_misses: observed.prefix_misses(),
    prefix_publications: observed.prefix_publications(),
    prefix_tokens_reused: observed.prefix_tokens_reused(),
    prefix_tokens_computed: observed.prefix_tokens_computed(),
    prefix_entries: observed.prefix_entries(),
    prefix_pages: observed.prefix_pages(),
  }
}

///|
fn spawned_prefix_initial_telemetry_valid(
  sample : SpawnedPrefixTelemetrySample,
) -> Bool {
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == 0 &&
  sample.kv_pages_free > 0 &&
  sample.prefix_lookups == 0UL &&
  sample.prefix_hits == 0UL &&
  sample.prefix_misses == 0UL &&
  sample.prefix_publications == 0UL &&
  sample.prefix_tokens_reused == 0UL &&
  sample.prefix_tokens_computed == 0UL &&
  sample.prefix_entries == 0 &&
  sample.prefix_pages == 0
}

///|
fn spawned_prefix_after_first_valid(
  sample : SpawnedPrefixTelemetrySample,
  initial_free : Int,
  input_tokens : Int,
  page_count : Int,
) -> Bool {
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == page_count &&
  sample.kv_pages_free == initial_free - page_count &&
  sample.prefix_lookups == 1UL &&
  sample.prefix_hits == 0UL &&
  sample.prefix_misses == 1UL &&
  sample.prefix_publications == 1UL &&
  sample.prefix_tokens_reused == 0UL &&
  sample.prefix_tokens_computed == input_tokens.to_uint64() &&
  sample.prefix_entries == 1 &&
  sample.prefix_pages == page_count
}

///|
fn spawned_prefix_after_second_valid(
  sample : SpawnedPrefixTelemetrySample,
  initial_free : Int,
  first_input_tokens : Int,
  second_input_tokens : Int,
  reused_tokens : Int,
  page_count : Int,
) -> Bool {
  let newly_computed = second_input_tokens - reused_tokens
  sample.queue_depth == 0 &&
  sample.active_requests == 0 &&
  sample.kv_pages_used == page_count &&
  sample.kv_pages_free == initial_free - page_count &&
  sample.prefix_lookups == 2UL &&
  sample.prefix_hits == 1UL &&
  sample.prefix_misses == 1UL &&
  sample.prefix_publications == 1UL &&
  sample.prefix_tokens_reused == reused_tokens.to_uint64() &&
  sample.prefix_tokens_computed ==
  (first_input_tokens + newly_computed).to_uint64() &&
  sample.prefix_entries == 1 &&
  sample.prefix_pages == page_count
}

///|
fn require_spawned_prefix_accepted(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  contract : RuntimeSpawnedPrefixValidationContract,
) -> Unit raise RuntimeSpawnedPrefixValidationError {
  let view = event.view() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
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
  event.delivered() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  if !valid {
    raise spawned_prefix_error(SpawnedPrefixEvent)
  }
}

///|
fn require_spawned_prefix_token(
  service : @online_session.LunaOnlineFramedService,
  event : @online_session.LunaOnlineFramedSemanticEvent,
  expected_token : Int,
  previous_plan_sequence : UInt64,
  exact_plan_tokens : Int,
) -> (Int, UInt64) raise RuntimeSpawnedPrefixValidationError {
  let view = event.view() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
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
      event.delivered() catch {
        _ => ()
      }
      raise spawned_prefix_error(SpawnedPrefixPlan)
    }
  }
  event.delivered() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  if observed.0 != expected_token || observed.1 != 0 {
    raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  if plan.last_plan_sequence() <= previous_plan_sequence ||
    plan.last_plan_rows() != 1 ||
    plan.last_plan_tokens() <= 0 ||
    (exact_plan_tokens > 0 && plan.last_plan_tokens() != exact_plan_tokens) {
    raise spawned_prefix_error(SpawnedPrefixPlan)
  }
  (observed.0, plan.last_plan_sequence())
}

///|
fn require_spawned_prefix_usage(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
  cached_input_tokens : Int,
) -> Unit raise RuntimeSpawnedPrefixValidationError {
  let view = event.view() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  let valid = try {
    if view.kind() != LunaUsageEventKind {
      false
    } else {
      let usage = view.usage()
      usage.input_tokens() == input_tokens &&
      usage.cached_input_tokens() == cached_input_tokens &&
      usage.output_tokens() == 1 &&
      usage.total_tokens() == input_tokens + 1
    }
  } catch {
    _ => false
  }
  event.delivered() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  if !valid {
    raise spawned_prefix_error(SpawnedPrefixUsage)
  }
}

///|
fn require_spawned_prefix_completed(
  event : @online_session.LunaOnlineFramedSemanticEvent,
  input_tokens : Int,
  cached_input_tokens : Int,
) -> Unit raise RuntimeSpawnedPrefixValidationError {
  let view = event.view() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  let valid = try {
    if view.kind() != LunaCompletedEventKind {
      false
    } else {
      let completed = view.completed()
      completed.finish_reason() == TokenLimit &&
      completed.input_tokens() == input_tokens &&
      completed.cached_input_tokens() == cached_input_tokens &&
      completed.output_tokens() == 1 &&
      completed.total_tokens() == input_tokens + 1
    }
  } catch {
    _ => false
  }
  event.delivered() catch {
    _ => raise spawned_prefix_error(SpawnedPrefixEvent)
  }
  if !valid {
    raise spawned_prefix_error(SpawnedPrefixUsage)
  }
}
