///|
fn wb_scheduler_sample(
  plan_sequence~ : UInt64,
  plan_rows~ : Int,
  plan_tokens~ : Int,
  prefix_lookups~ : UInt64,
  prefix_hits~ : UInt64,
  prefix_misses~ : UInt64,
  prefix_evictions~ : UInt64,
  prefix_tokens_reused~ : UInt64,
  prefix_tokens_computed~ : UInt64,
  prefix_publications~ : UInt64,
  kv_pages_used? : Int = 3,
  kv_pages_free? : Int = 5,
  prefix_entries? : Int = 2,
  prefix_pages? : Int = 4,
) -> LunaServerSchedulerSample {
  {
    queue_depth: 1,
    active_requests: 2,
    kv_pages_used,
    kv_pages_free,
    prefix_lookups,
    prefix_hits,
    prefix_misses,
    prefix_evictions,
    prefix_tokens_reused,
    prefix_tokens_computed,
    prefix_publications,
    prefix_entries,
    prefix_pages,
    plan_sequence,
    plan_rows,
    plan_tokens,
  }
}

///|
fn wb_histogram_count(
  metrics : @metrics.LunaInstanceMetrics,
  histogram : @metrics.LunaInstanceHistogram,
) -> UInt64 {
  let mut total = 0UL
  for index in 0..<@metrics.luna_instance_histogram_bucket_count() {
    total += try! metrics.histogram_bucket_value(histogram, index)
  }
  total
}

///|
fn wb_observation_sample(
  kind : @online_session.LunaOnlineFramedObservationKind,
  latency_kind : @online_session.LunaOnlineFramedLatencyKind,
  latency_millis? : Int = 0,
  input_tokens? : Int = 0,
  output_tokens? : Int = 0,
) -> LunaServerObservationSample {
  {
    kind: luna_server_observation_kind_code(kind),
    latency_kind: luna_server_latency_kind_code(latency_kind),
    latency_millis,
    input_tokens,
    output_tokens,
  }
}

///|
fn wb_assert_prefix_totals(
  bridge : LunaServerTelemetryBridge,
  lookups~ : UInt64,
  hits~ : UInt64,
  misses~ : UInt64,
  evictions~ : UInt64,
  reused~ : UInt64,
  computed~ : UInt64,
  publications~ : UInt64,
  entries~ : Int,
  pages~ : Int,
) -> Unit raise {
  assert_eq(bridge.metrics.counter_value(PrefixLookups), lookups)
  assert_eq(bridge.metrics.counter_value(PrefixHits), hits)
  assert_eq(bridge.metrics.counter_value(PrefixMisses), misses)
  assert_eq(bridge.metrics.counter_value(PrefixEvictions), evictions)
  assert_eq(bridge.metrics.counter_value(PrefixTokensReused), reused)
  assert_eq(bridge.metrics.counter_value(PrefixTokensComputed), computed)
  assert_eq(bridge.metrics.counter_value(PrefixPublications), publications)
  assert_eq(bridge.metrics.gauge_value(PrefixEntries), entries)
  assert_eq(bridge.metrics.gauge_value(PrefixPages), pages)
}

///|
test "native telemetry delegate applies prefix totals exactly once" {
  let bridge = try! LunaServerTelemetryBridge::new(8)
  let initial = wb_scheduler_sample(
    plan_sequence=1UL,
    plan_rows=2,
    plan_tokens=7,
    prefix_lookups=5UL,
    prefix_hits=3UL,
    prefix_misses=2UL,
    prefix_evictions=1UL,
    prefix_tokens_reused=11UL,
    prefix_tokens_computed=13UL,
    prefix_publications=2UL,
  )
  assert_true(luna_native_server_refresh_sample(bridge, initial))
  assert_true(luna_native_server_refresh_sample(bridge, initial))
  wb_assert_prefix_totals(
    bridge,
    lookups=5UL,
    hits=3UL,
    misses=2UL,
    evictions=1UL,
    reused=11UL,
    computed=13UL,
    publications=2UL,
    entries=2,
    pages=4,
  )
  assert_eq(wb_histogram_count(bridge.metrics, BatchRows), 1UL)
  assert_eq(wb_histogram_count(bridge.metrics, BatchTokens), 1UL)
  bridge.close_gauges()
  assert_eq(bridge.metrics.gauge_value(QueueDepth), 0)
  assert_eq(bridge.metrics.gauge_value(ActiveRequests), 0)
  assert_eq(bridge.metrics.gauge_value(KvPagesUsed), 0)
  assert_eq(bridge.metrics.gauge_value(KvPagesFree), 8)
  assert_eq(bridge.metrics.gauge_value(PrefixEntries), 0)
  assert_eq(bridge.metrics.gauge_value(PrefixPages), 0)
}

///|
test "OpenAI telemetry delegate matches native delta and gauge semantics" {
  let bridge = try! LunaServerTelemetryBridge::new(8)
  let initial = wb_scheduler_sample(
    plan_sequence=1UL,
    plan_rows=2,
    plan_tokens=7,
    prefix_lookups=5UL,
    prefix_hits=3UL,
    prefix_misses=2UL,
    prefix_evictions=1UL,
    prefix_tokens_reused=11UL,
    prefix_tokens_computed=13UL,
    prefix_publications=2UL,
  )
  let next = wb_scheduler_sample(
    plan_sequence=2UL,
    plan_rows=3,
    plan_tokens=9,
    prefix_lookups=9UL,
    prefix_hits=6UL,
    prefix_misses=3UL,
    prefix_evictions=2UL,
    prefix_tokens_reused=19UL,
    prefix_tokens_computed=17UL,
    prefix_publications=4UL,
    prefix_entries=3,
    prefix_pages=6,
  )
  assert_true(luna_openai_server_refresh_sample(bridge, initial))
  assert_true(luna_openai_server_refresh_sample(bridge, next))
  assert_true(luna_openai_server_refresh_sample(bridge, next))
  wb_assert_prefix_totals(
    bridge,
    lookups=9UL,
    hits=6UL,
    misses=3UL,
    evictions=2UL,
    reused=19UL,
    computed=17UL,
    publications=4UL,
    entries=3,
    pages=6,
  )
  assert_eq(wb_histogram_count(bridge.metrics, BatchRows), 2UL)
  assert_eq(wb_histogram_count(bridge.metrics, BatchTokens), 2UL)
}

///|
test "telemetry drift is sticky and zero KV total is not relearned" {
  let prefix = try! LunaServerTelemetryBridge::new(4)
  let initial = wb_scheduler_sample(
    plan_sequence=0UL,
    plan_rows=0,
    plan_tokens=0,
    prefix_lookups=2UL,
    prefix_hits=1UL,
    prefix_misses=1UL,
    prefix_evictions=0UL,
    prefix_tokens_reused=3UL,
    prefix_tokens_computed=5UL,
    prefix_publications=1UL,
    kv_pages_used=0,
    kv_pages_free=0,
  )
  assert_true(luna_native_server_refresh_sample(prefix, initial))
  let rollback = wb_scheduler_sample(
    plan_sequence=0UL,
    plan_rows=0,
    plan_tokens=0,
    prefix_lookups=1UL,
    prefix_hits=1UL,
    prefix_misses=0UL,
    prefix_evictions=0UL,
    prefix_tokens_reused=3UL,
    prefix_tokens_computed=5UL,
    prefix_publications=1UL,
    kv_pages_used=0,
    kv_pages_free=0,
  )
  assert_false(luna_native_server_refresh_sample(prefix, rollback))
  assert_false(luna_native_server_refresh_sample(prefix, initial))

  let kv = try! LunaServerTelemetryBridge::new(4)
  assert_true(luna_openai_server_refresh_sample(kv, initial))
  let changed_total = wb_scheduler_sample(
    plan_sequence=0UL,
    plan_rows=0,
    plan_tokens=0,
    prefix_lookups=2UL,
    prefix_hits=1UL,
    prefix_misses=1UL,
    prefix_evictions=0UL,
    prefix_tokens_reused=3UL,
    prefix_tokens_computed=5UL,
    prefix_publications=1UL,
    kv_pages_used=0,
    kv_pages_free=1,
  )
  assert_false(luna_openai_server_refresh_sample(kv, changed_total))
}

///|
test "telemetry rejects zero-sequence shape and same-sequence plan drift" {
  let zero = try! LunaServerTelemetryBridge::new(4)
  let invalid_zero = wb_scheduler_sample(
    plan_sequence=0UL,
    plan_rows=1,
    plan_tokens=0,
    prefix_lookups=0UL,
    prefix_hits=0UL,
    prefix_misses=0UL,
    prefix_evictions=0UL,
    prefix_tokens_reused=0UL,
    prefix_tokens_computed=0UL,
    prefix_publications=0UL,
  )
  assert_false(luna_native_server_refresh_sample(zero, invalid_zero))
  let plan = try! LunaServerTelemetryBridge::new(4)
  let first = wb_scheduler_sample(
    plan_sequence=4UL,
    plan_rows=2,
    plan_tokens=8,
    prefix_lookups=0UL,
    prefix_hits=0UL,
    prefix_misses=0UL,
    prefix_evictions=0UL,
    prefix_tokens_reused=0UL,
    prefix_tokens_computed=0UL,
    prefix_publications=0UL,
  )
  let drift = { ..first, plan_rows: 3 }
  assert_true(luna_openai_server_refresh_sample(plan, first))
  assert_false(luna_openai_server_refresh_sample(plan, drift))
}

///|
test "observation ACK retry guard suppresses duplicate bridge mutation" {
  assert_true(luna_server_observation_requires_record(false))
  assert_false(luna_server_observation_requires_record(true))
}

///|
test "pipeline pending accounting covers every admitted terminal kind" {
  let admission = @online_session.LunaOnlineFramedAdmissionObservation
  assert_true(luna_online_tcp_pipeline_admission_observation(admission))
  assert_false(luna_online_tcp_pipeline_terminal_observation(admission))
  let mut pending = luna_online_tcp_pipeline_pending_after_record(
    0,
    luna_online_tcp_pipeline_admission_observation(admission),
    luna_online_tcp_pipeline_terminal_observation(admission),
  )
  assert_eq(pending, 1)
  pending = luna_online_tcp_pipeline_pending_after_record(pending, true, false)
  assert_eq(pending, 2)
  let terminals = [
    @online_session.LunaOnlineFramedCompletionObservation,
    LunaOnlineFramedCancellationObservation,
    LunaOnlineFramedDeadlineObservation,
    LunaOnlineFramedRequestFailureObservation,
    LunaOnlineFramedWorkerRequestTerminalObservation,
  ]
  for kind in terminals {
    assert_false(luna_online_tcp_pipeline_admission_observation(kind))
    assert_true(luna_online_tcp_pipeline_terminal_observation(kind))
  }
  pending = luna_online_tcp_pipeline_pending_after_record(
    pending,
    false,
    luna_online_tcp_pipeline_terminal_observation(
      LunaOnlineFramedCancellationObservation,
    ),
  )
  assert_eq(pending, 1)
  pending = luna_online_tcp_pipeline_pending_after_record(
    pending,
    false,
    luna_online_tcp_pipeline_terminal_observation(
      LunaOnlineFramedWorkerRequestTerminalObservation,
    ),
  )
  assert_eq(pending, 0)
}

///|
test "pipeline pending accounting rejects zero and overflow without mutation" {
  assert_eq(luna_online_tcp_pipeline_pending_after_record(0, false, true), -1)
  assert_eq(
    luna_online_tcp_pipeline_pending_after_record(0x7fffffff, true, false),
    -1,
  )
  assert_eq(luna_online_tcp_pipeline_pending_after_record(-1, false, false), -1)
  assert_eq(luna_online_tcp_pipeline_pending_after_record(3, true, true), -1)
  let nonterminals = [
    @online_session.LunaOnlineFramedTokenObservation,
    LunaOnlineFramedUsageObservation,
    LunaOnlineFramedWorkerFailureObservation,
    LunaOnlineFramedWorkerRestartObservation,
  ]
  for kind in nonterminals {
    assert_false(luna_online_tcp_pipeline_admission_observation(kind))
    assert_false(luna_online_tcp_pipeline_terminal_observation(kind))
    assert_eq(luna_online_tcp_pipeline_pending_after_record(3, false, false), 3)
  }
}

///|
test "bridge records authenticated latency scalars into exact vocabularies" {
  let bridge = try! LunaServerTelemetryBridge::new(4)
  assert_true(
    bridge.record_latency_sample_code(
      luna_server_latency_kind_code(LunaOnlineFramedFirstTokenLatency),
      12,
    ),
  )
  assert_true(
    bridge.record_latency_sample_code(
      luna_server_latency_kind_code(LunaOnlineFramedInterTokenLatency),
      7,
    ),
  )
  assert_true(
    bridge.record_latency_sample_code(
      luna_server_latency_kind_code(LunaOnlineFramedRequestLatency),
      31,
    ),
  )
  assert_true(
    bridge.record_latency_sample_code(
      luna_server_latency_kind_code(LunaOnlineFramedNoLatency),
      0,
    ),
  )
  assert_false(
    bridge.record_latency_sample_code(
      luna_server_latency_kind_code(LunaOnlineFramedRequestLatency),
      -1,
    ),
  )
  assert_eq(wb_histogram_count(bridge.metrics, FirstTokenLatencyMillis), 1UL)
  assert_eq(wb_histogram_count(bridge.metrics, InterTokenLatencyMillis), 1UL)
  assert_eq(wb_histogram_count(bridge.metrics, RequestLatencyMillis), 1UL)
}

///|
test "cold-start telemetry commits first valid sample exactly once" {
  let bridge = try! LunaServerTelemetryBridge::new(4)
  assert_eq(bridge.record_cold_start_latency_millis(-1), -1)
  assert_eq(bridge.record_cold_start_latency_millis(86400001), -1)
  assert_eq(wb_histogram_count(bridge.metrics, ColdStartLatencyMillis), 0UL)
  assert_eq(bridge.record_cold_start_latency_millis(321), 1)
  assert_eq(bridge.record_cold_start_latency_millis(654), 0)
  assert_eq(bridge.record_cold_start_latency_millis(-1), -1)
  assert_eq(wb_histogram_count(bridge.metrics, ColdStartLatencyMillis), 1UL)
}

///|
fn wb_count_log_event(
  bridge : LunaServerTelemetryBridge,
  event : @instance_log.LunaInstanceLogEvent,
) -> Int raise {
  let snapshot = bridge.log.snapshot()
  let mut count = 0
  for index in 0..<snapshot.length() {
    if snapshot.event_at(index) == event {
      count += 1
    }
  }
  count
}

///|
test "worker loss is durable before Usage disconnect and terminal is count-free" {
  let bridge = try! LunaServerTelemetryBridge::new(8)
  let no_latency : @online_session.LunaOnlineFramedLatencyKind = LunaOnlineFramedNoLatency
  assert_eq(
    bridge.record_observation_sample(
      wb_observation_sample(LunaOnlineFramedAdmissionObservation, no_latency),
    ),
    0,
  )
  let mut worker_recorded = false
  if luna_server_observation_requires_record(worker_recorded) {
    assert_eq(
      bridge.record_observation_sample(
        wb_observation_sample(
          LunaOnlineFramedWorkerFailureObservation,
          no_latency,
        ),
      ),
      0,
    )
    worker_recorded = true
  }
  // An ACK retry must not recount the durable worker/request failure pulse.
  assert_false(luna_server_observation_requires_record(worker_recorded))
  assert_eq(
    bridge.record_observation_sample(
      wb_observation_sample(
        LunaOnlineFramedWorkerRestartObservation,
        no_latency,
      ),
    ),
    0,
  )
  assert_eq(
    bridge.record_observation_sample(
      wb_observation_sample(
        LunaOnlineFramedUsageObservation,
        no_latency,
        input_tokens=3,
        output_tokens=1,
      ),
    ),
    0,
  )
  // Model a peer disconnect here: the terminal pulse is deliberately absent.
  assert_eq(bridge.metrics.counter_value(WorkerFailures), 1UL)
  assert_eq(bridge.metrics.counter_value(Failures), 1UL)
  assert_eq(wb_count_log_event(bridge, LunaWorkerFailed), 1)
  assert_eq(wb_count_log_event(bridge, LunaRequestFailed), 1)

  let terminal = wb_observation_sample(
    LunaOnlineFramedWorkerRequestTerminalObservation,
    LunaOnlineFramedRequestLatency,
    latency_millis=37,
  )
  let mut terminal_recorded = false
  if luna_server_observation_requires_record(terminal_recorded) {
    assert_eq(bridge.record_observation_sample(terminal), 1)
    terminal_recorded = true
  }
  assert_false(luna_server_observation_requires_record(terminal_recorded))
  assert_eq(bridge.metrics.counter_value(WorkerFailures), 1UL)
  assert_eq(bridge.metrics.counter_value(Failures), 1UL)
  assert_eq(wb_count_log_event(bridge, LunaWorkerFailed), 1)
  assert_eq(wb_count_log_event(bridge, LunaRequestFailed), 1)
  assert_eq(wb_histogram_count(bridge.metrics, RequestLatencyMillis), 1UL)
}

///|
test "non-worker request failure remains counted and terminal" {
  let bridge = try! LunaServerTelemetryBridge::new(4)
  assert_eq(
    bridge.record_observation_sample(
      wb_observation_sample(
        LunaOnlineFramedRequestFailureObservation,
        LunaOnlineFramedRequestLatency,
        latency_millis=19,
      ),
    ),
    1,
  )
  assert_eq(bridge.metrics.counter_value(Failures), 1UL)
  assert_eq(bridge.metrics.counter_value(WorkerFailures), 0UL)
  assert_eq(wb_count_log_event(bridge, LunaRequestFailed), 1)
  assert_eq(wb_histogram_count(bridge.metrics, RequestLatencyMillis), 1UL)
}
