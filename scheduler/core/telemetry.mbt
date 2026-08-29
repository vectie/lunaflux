///|
pub fn Scheduler::luna_telemetry(self : Scheduler) -> LunaSchedulerTelemetry {
  let free = self.page_allocator.free_page_count()
  {
    queue_depth: self.waiting_count,
    active_requests: self.active_count,
    kv_pages_used: self.capacity.total_page_count() - free,
    kv_pages_free: free,
    prefix_lookups: self.prefix_lookups,
    prefix_hits: self.prefix_hits,
    prefix_misses: self.prefix_misses,
    prefix_evictions: self.prefix_evictions,
    prefix_tokens_reused: self.prefix_tokens_reused,
    prefix_tokens_computed: self.prefix_tokens_computed,
    prefix_publications: self.prefix_publications,
    prefix_entries: match self.prefix_index {
      Some(index) => index.entry_count()
      None => 0
    },
    prefix_pages: match self.prefix_index {
      Some(index) => index.page_count()
      None => 0
    },
  }
}

///|
pub fn LunaSchedulerTelemetry::prefix_lookups(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_lookups
}

///|
pub fn LunaSchedulerTelemetry::prefix_hits(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_hits
}

///|
pub fn LunaSchedulerTelemetry::prefix_misses(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_misses
}

///|
pub fn LunaSchedulerTelemetry::prefix_evictions(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_evictions
}

///|
pub fn LunaSchedulerTelemetry::prefix_tokens_reused(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_tokens_reused
}

///|
pub fn LunaSchedulerTelemetry::prefix_tokens_computed(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_tokens_computed
}

///|
pub fn LunaSchedulerTelemetry::prefix_publications(
  self : LunaSchedulerTelemetry,
) -> UInt64 {
  self.prefix_publications
}

///|
pub fn LunaSchedulerTelemetry::prefix_entries(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.prefix_entries
}

///|
pub fn LunaSchedulerTelemetry::prefix_pages(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.prefix_pages
}

///|
pub fn Scheduler::luna_plan_telemetry(
  self : Scheduler,
) -> LunaSchedulerPlanTelemetry {
  {
    plan_sequence: self.telemetry_plan_sequence,
    plan_rows: self.telemetry_plan_rows,
    plan_tokens: self.telemetry_plan_tokens,
  }
}

///|
pub fn LunaSchedulerTelemetry::queue_depth(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.queue_depth
}

///|
pub fn LunaSchedulerTelemetry::active_requests(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.active_requests
}

///|
pub fn LunaSchedulerTelemetry::kv_pages_used(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.kv_pages_used
}

///|
pub fn LunaSchedulerTelemetry::kv_pages_free(
  self : LunaSchedulerTelemetry,
) -> Int {
  self.kv_pages_free
}

///|
pub fn LunaSchedulerPlanTelemetry::last_plan_sequence(
  self : LunaSchedulerPlanTelemetry,
) -> UInt64 {
  self.plan_sequence
}

///|
pub fn LunaSchedulerPlanTelemetry::last_plan_rows(
  self : LunaSchedulerPlanTelemetry,
) -> Int {
  self.plan_rows
}

///|
pub fn LunaSchedulerPlanTelemetry::last_plan_tokens(
  self : LunaSchedulerPlanTelemetry,
) -> Int {
  self.plan_tokens
}
