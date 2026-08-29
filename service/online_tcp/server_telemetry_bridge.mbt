///|
priv struct LunaServerSchedulerSample {
  queue_depth : Int
  active_requests : Int
  kv_pages_used : Int
  kv_pages_free : Int
  prefix_lookups : UInt64
  prefix_hits : UInt64
  prefix_misses : UInt64
  prefix_evictions : UInt64
  prefix_tokens_reused : UInt64
  prefix_tokens_computed : UInt64
  prefix_publications : UInt64
  prefix_entries : Int
  prefix_pages : Int
  plan_sequence : UInt64
  plan_rows : Int
  plan_tokens : Int
}

///|
priv struct LunaServerTelemetryBridge {
  metrics : @metrics.LunaInstanceMetrics
  log : @instance_log.LunaInstanceLog
  clock : @monotonic_clock.MonotonicClock
  mut last_log_timestamp : UInt64
  mut total_kv_pages : Int
  mut kv_total_initialized : Bool
  mut last_plan_sequence : UInt64
  mut last_plan_rows : Int
  mut last_plan_tokens : Int
  mut prefix_lookups : UInt64
  mut prefix_hits : UInt64
  mut prefix_misses : UInt64
  mut prefix_evictions : UInt64
  mut prefix_tokens_reused : UInt64
  mut prefix_tokens_computed : UInt64
  mut prefix_publications : UInt64
  mut graph_hits : UInt64
  mut graph_misses : UInt64
  mut graph_initialized : Bool
  mut cold_start_recorded : Bool
  mut degraded : Bool
}

///|
fn luna_server_scheduler_sample(
  telemetry : @online_session.LunaOnlineFramedTelemetry,
  plan : @online_session.LunaOnlineFramedPlanTelemetry,
) -> LunaServerSchedulerSample {
  {
    queue_depth: telemetry.queue_depth(),
    active_requests: telemetry.active_requests(),
    kv_pages_used: telemetry.kv_pages_used(),
    kv_pages_free: telemetry.kv_pages_free(),
    prefix_lookups: telemetry.prefix_lookups(),
    prefix_hits: telemetry.prefix_hits(),
    prefix_misses: telemetry.prefix_misses(),
    prefix_evictions: telemetry.prefix_evictions(),
    prefix_tokens_reused: telemetry.prefix_tokens_reused(),
    prefix_tokens_computed: telemetry.prefix_tokens_computed(),
    prefix_publications: telemetry.prefix_publications(),
    prefix_entries: telemetry.prefix_entries(),
    prefix_pages: telemetry.prefix_pages(),
    plan_sequence: plan.last_plan_sequence(),
    plan_rows: plan.last_plan_rows(),
    plan_tokens: plan.last_plan_tokens(),
  }
}

///|
fn LunaServerTelemetryBridge::new(
  log_capacity : Int,
) -> LunaServerTelemetryBridge raise LunaOnlineTcpError {
  let log = @instance_log.LunaInstanceLog::new(log_capacity) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  {
    metrics: @metrics.LunaInstanceMetrics::new(),
    log,
    clock: @monotonic_clock.MonotonicClock::system(),
    last_log_timestamp: 0UL,
    total_kv_pages: 0,
    kv_total_initialized: false,
    last_plan_sequence: 0UL,
    last_plan_rows: 0,
    last_plan_tokens: 0,
    prefix_lookups: 0UL,
    prefix_hits: 0UL,
    prefix_misses: 0UL,
    prefix_evictions: 0UL,
    prefix_tokens_reused: 0UL,
    prefix_tokens_computed: 0UL,
    prefix_publications: 0UL,
    graph_hits: 0UL,
    graph_misses: 0UL,
    graph_initialized: false,
    cold_start_recorded: false,
    degraded: false,
  }
}

///|
fn LunaServerTelemetryBridge::refresh_graph_runtime(
  self : LunaServerTelemetryBridge,
  report : @worker_wire.WorkerGraphTelemetry,
) -> Bool {
  if self.degraded {
    return false
  }
  let hits = report.graph_hits()
  let misses = report.graph_misses()
  if (
      self.graph_initialized &&
      (hits < self.graph_hits || misses < self.graph_misses)
    ) ||
    (hits == 0UL && misses == 0UL) {
    self.degraded = true
    return false
  }
  self.metrics.record_counter_u64(GraphHits, hits - self.graph_hits)
  self.metrics.record_counter_u64(GraphMisses, misses - self.graph_misses)
  self.graph_hits = hits
  self.graph_misses = misses
  self.graph_initialized = true
  true
}

///|
/// Returns -1 for an invalid duration, 0 after the once-only slot was already
/// committed, and 1 after recording the first valid startup interval.
fn LunaServerTelemetryBridge::record_cold_start_latency_millis(
  self : LunaServerTelemetryBridge,
  duration_millis : Int,
) -> Int {
  if duration_millis < 0 || duration_millis > 86400000 {
    return -1
  }
  if self.cold_start_recorded {
    return 0
  }
  try! self.metrics.record_histogram(ColdStartLatencyMillis, duration_millis)
  self.cold_start_recorded = true
  1
}

///|
fn LunaServerTelemetryBridge::valid_sample(
  self : LunaServerTelemetryBridge,
  sample : LunaServerSchedulerSample,
) -> Bool {
  if sample.queue_depth < 0 ||
    sample.active_requests < 0 ||
    sample.kv_pages_used < 0 ||
    sample.kv_pages_free < 0 ||
    sample.prefix_entries < 0 ||
    sample.prefix_pages < 0 ||
    sample.plan_rows < 0 ||
    sample.plan_tokens < 0 ||
    sample.kv_pages_used > 0x7fffffff - sample.kv_pages_free {
    return false
  }
  let total = sample.kv_pages_used + sample.kv_pages_free
  if self.kv_total_initialized && self.total_kv_pages != total {
    return false
  }
  if sample.plan_sequence == 0UL &&
    (sample.plan_rows != 0 || sample.plan_tokens != 0) {
    return false
  }
  if sample.plan_sequence < self.last_plan_sequence ||
    (
      sample.plan_sequence == self.last_plan_sequence &&
      (
        sample.plan_rows != self.last_plan_rows ||
        sample.plan_tokens != self.last_plan_tokens
      )
    ) {
    return false
  }
  sample.prefix_lookups >= self.prefix_lookups &&
  sample.prefix_hits >= self.prefix_hits &&
  sample.prefix_misses >= self.prefix_misses &&
  sample.prefix_evictions >= self.prefix_evictions &&
  sample.prefix_tokens_reused >= self.prefix_tokens_reused &&
  sample.prefix_tokens_computed >= self.prefix_tokens_computed &&
  sample.prefix_publications >= self.prefix_publications
}

///|
fn LunaServerTelemetryBridge::refresh_sample(
  self : LunaServerTelemetryBridge,
  sample : LunaServerSchedulerSample,
) -> Bool {
  if self.degraded || !self.valid_sample(sample) {
    self.degraded = true
    return false
  }
  let first_total = !self.kv_total_initialized
  let total = sample.kv_pages_used + sample.kv_pages_free
  let next_plan = sample.plan_sequence > self.last_plan_sequence
  let lookups_delta = sample.prefix_lookups - self.prefix_lookups
  let hits_delta = sample.prefix_hits - self.prefix_hits
  let misses_delta = sample.prefix_misses - self.prefix_misses
  let evictions_delta = sample.prefix_evictions - self.prefix_evictions
  let reused_delta = sample.prefix_tokens_reused - self.prefix_tokens_reused
  let computed_delta = sample.prefix_tokens_computed -
    self.prefix_tokens_computed
  let publications_delta = sample.prefix_publications - self.prefix_publications
  try! self.metrics.set_gauge(QueueDepth, sample.queue_depth)
  try! self.metrics.set_gauge(ActiveRequests, sample.active_requests)
  try! self.metrics.set_gauge(KvPagesUsed, sample.kv_pages_used)
  try! self.metrics.set_gauge(KvPagesFree, sample.kv_pages_free)
  try! self.metrics.set_gauge(PrefixEntries, sample.prefix_entries)
  try! self.metrics.set_gauge(PrefixPages, sample.prefix_pages)
  self.metrics.record_counter_u64(PrefixLookups, lookups_delta)
  self.metrics.record_counter_u64(PrefixHits, hits_delta)
  self.metrics.record_counter_u64(PrefixMisses, misses_delta)
  self.metrics.record_counter_u64(PrefixEvictions, evictions_delta)
  self.metrics.record_counter_u64(PrefixTokensReused, reused_delta)
  self.metrics.record_counter_u64(PrefixTokensComputed, computed_delta)
  self.metrics.record_counter_u64(PrefixPublications, publications_delta)
  if next_plan {
    try! self.metrics.record_histogram(BatchRows, sample.plan_rows)
    try! self.metrics.record_histogram(BatchTokens, sample.plan_tokens)
  }
  if first_total {
    self.total_kv_pages = total
    self.kv_total_initialized = true
  }
  self.last_plan_sequence = sample.plan_sequence
  self.last_plan_rows = sample.plan_rows
  self.last_plan_tokens = sample.plan_tokens
  self.prefix_lookups = sample.prefix_lookups
  self.prefix_hits = sample.prefix_hits
  self.prefix_misses = sample.prefix_misses
  self.prefix_evictions = sample.prefix_evictions
  self.prefix_tokens_reused = sample.prefix_tokens_reused
  self.prefix_tokens_computed = sample.prefix_tokens_computed
  self.prefix_publications = sample.prefix_publications
  true
}

///|
fn LunaServerTelemetryBridge::refresh(
  self : LunaServerTelemetryBridge,
  telemetry : @online_session.LunaOnlineFramedTelemetry,
  plan : @online_session.LunaOnlineFramedPlanTelemetry,
) -> Bool {
  self.refresh_sample(luna_server_scheduler_sample(telemetry, plan))
}

///|
fn luna_native_server_refresh_sample(
  bridge : LunaServerTelemetryBridge,
  sample : LunaServerSchedulerSample,
) -> Bool {
  bridge.refresh_sample(sample)
}

///|
fn luna_openai_server_refresh_sample(
  bridge : LunaServerTelemetryBridge,
  sample : LunaServerSchedulerSample,
) -> Bool {
  bridge.refresh_sample(sample)
}

///|
fn LunaServerTelemetryBridge::next_log_timestamp(
  self : LunaServerTelemetryBridge,
) -> UInt64 {
  let observed = self.clock.now_millis() catch {
    _ => return self.last_log_timestamp
  }
  if observed < self.last_log_timestamp {
    return self.last_log_timestamp
  }
  self.last_log_timestamp = observed
  observed
}

///|
fn LunaServerTelemetryBridge::record_log(
  self : LunaServerTelemetryBridge,
  event : @instance_log.LunaInstanceLogEvent,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  try! self.log.record(
    event,
    reason,
    timestamp_millis=self.next_log_timestamp(),
    count=1,
    duration_millis=0,
  )
}

///|
fn LunaServerTelemetryBridge::record_network_accept(
  self : LunaServerTelemetryBridge,
) -> Unit {
  try! self.metrics.record_counter(NetworkAccepts)
  self.record_log(LunaNetworkAccepted, LunaLogNoReason)
}

///|
fn LunaServerTelemetryBridge::record_network_disconnect(
  self : LunaServerTelemetryBridge,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  try! self.metrics.record_counter(NetworkDisconnects)
  self.record_log(LunaNetworkDisconnected, reason)
}

///|
fn LunaServerTelemetryBridge::record_backpressure(
  self : LunaServerTelemetryBridge,
) -> Unit {
  try! self.metrics.record_counter(Backpressure)
  self.record_log(LunaBackpressureObserved, LunaLogCapacity)
}

///|
fn LunaServerTelemetryBridge::record_network_rejection(
  self : LunaServerTelemetryBridge,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  try! self.metrics.record_counter(NetworkRejections)
  self.record_log(LunaNetworkRejected, reason)
  self.record_log(LunaAdmissionRejected, reason)
}

///|
fn LunaServerTelemetryBridge::record_network_rejection_only(
  self : LunaServerTelemetryBridge,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  try! self.metrics.record_counter(NetworkRejections)
  self.record_log(LunaNetworkRejected, reason)
}

///|
fn LunaServerTelemetryBridge::record_admission_rejection(
  self : LunaServerTelemetryBridge,
  reason : @instance_log.LunaInstanceLogReason,
) -> Unit {
  self.record_log(LunaAdmissionRejected, reason)
}

///|
fn LunaServerTelemetryBridge::close_gauges(
  self : LunaServerTelemetryBridge,
) -> Unit {
  try! self.metrics.set_gauge(QueueDepth, 0)
  try! self.metrics.set_gauge(ActiveRequests, 0)
  try! self.metrics.set_gauge(KvPagesUsed, 0)
  try! self.metrics.set_gauge(KvPagesFree, self.total_kv_pages)
  try! self.metrics.set_gauge(PrefixEntries, 0)
  try! self.metrics.set_gauge(PrefixPages, 0)
}
