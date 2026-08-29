///|
fn LunaOnlineInstance::luna_scheduler_telemetry(
  self : LunaOnlineInstance,
) -> @core.LunaSchedulerTelemetry raise LunaOnlineInstanceError {
  self.lease_owner().luna_scheduler_telemetry() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
}

///|
fn LunaOnlineInstance::luna_scheduler_plan_telemetry(
  self : LunaOnlineInstance,
) -> @core.LunaSchedulerPlanTelemetry raise LunaOnlineInstanceError {
  self.lease_owner().luna_scheduler_plan_telemetry() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
}

///|
fn LunaOnlineInstance::luna_graph_runtime_telemetry(
  self : LunaOnlineInstance,
) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineInstanceError {
  self.lease_owner().luna_graph_runtime_telemetry() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
}

///|
fn LunaOnlineInstance::has_luna_graph_runtime_telemetry(
  self : LunaOnlineInstance,
) -> Bool raise LunaOnlineInstanceError {
  self.lease_owner().has_luna_graph_runtime_telemetry() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
}

///|
pub fn LunaOnlineFramedService::luna_telemetry(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedTelemetry raise LunaOnlineFramedCoordinatorError {
  let lower = self.instance.luna_scheduler_telemetry() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
  let scheduler_queue = lower.queue_depth()
  let prepared_count = self.prepared.length()
  if scheduler_queue < 0 ||
    self.queue_count < 0 ||
    prepared_count < 0 ||
    self.queue_count > 0x7fffffff - prepared_count ||
    scheduler_queue > 0x7fffffff - self.queue_count - prepared_count {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  let queued = scheduler_queue + self.queue_count + prepared_count
  {
    queue_depth: queued,
    active_requests: lower.active_requests(),
    kv_pages_used: lower.kv_pages_used(),
    kv_pages_free: lower.kv_pages_free(),
    prefix_lookups: lower.prefix_lookups(),
    prefix_hits: lower.prefix_hits(),
    prefix_misses: lower.prefix_misses(),
    prefix_evictions: lower.prefix_evictions(),
    prefix_tokens_reused: lower.prefix_tokens_reused(),
    prefix_tokens_computed: lower.prefix_tokens_computed(),
    prefix_publications: lower.prefix_publications(),
    prefix_entries: lower.prefix_entries(),
    prefix_pages: lower.prefix_pages(),
  }
}

///|
pub fn LunaOnlineFramedService::luna_plan_telemetry(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedPlanTelemetry raise LunaOnlineFramedCoordinatorError {
  let lower = self.instance.luna_scheduler_plan_telemetry() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
  {
    plan_sequence: lower.last_plan_sequence(),
    plan_rows: lower.last_plan_rows(),
    plan_tokens: lower.last_plan_tokens(),
  }
}

///|
/// Reports the authenticated startup-selected graph/eager path, optional
/// manifest shape class, and replacement-stable completed hit/miss totals.
pub fn LunaOnlineFramedService::luna_graph_runtime_telemetry(
  self : LunaOnlineFramedService,
) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineFramedCoordinatorError {
  self.instance.luna_graph_runtime_telemetry() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
}

///|
pub fn LunaOnlineFramedService::has_luna_graph_runtime_telemetry(
  self : LunaOnlineFramedService,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  self.instance.has_luna_graph_runtime_telemetry() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::luna_telemetry(
  self : LunaOnlineFramedServicePreparation,
) -> LunaOnlineFramedTelemetry raise LunaOnlineInstancePreparationError {
  let service = self.require_ready_service()
  service.luna_telemetry() catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::luna_plan_telemetry(
  self : LunaOnlineFramedServicePreparation,
) -> LunaOnlineFramedPlanTelemetry raise LunaOnlineInstancePreparationError {
  let service = self.require_ready_service()
  service.luna_plan_telemetry() catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::luna_graph_runtime_telemetry(
  self : LunaOnlineFramedServicePreparation,
) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineInstancePreparationError {
  let service = self.require_ready_service()
  service.luna_graph_runtime_telemetry() catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
}

///|
pub fn LunaOnlineFramedServicePreparation::has_luna_graph_runtime_telemetry(
  self : LunaOnlineFramedServicePreparation,
) -> Bool raise LunaOnlineInstancePreparationError {
  let service = self.require_ready_service()
  service.has_luna_graph_runtime_telemetry() catch {
    _ => raise luna_coordinator_preparation_failure(LunaOnlineFramedPreparation)
  }
}

///|
pub fn LunaOnlineFramedTelemetry::queue_depth(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.queue_depth
}

///|
pub fn LunaOnlineFramedTelemetry::active_requests(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.active_requests
}

///|
pub fn LunaOnlineFramedTelemetry::kv_pages_used(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.kv_pages_used
}

///|
pub fn LunaOnlineFramedTelemetry::kv_pages_free(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.kv_pages_free
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_lookups(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_lookups
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_hits(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_hits
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_misses(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_misses
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_evictions(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_evictions
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_tokens_reused(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_tokens_reused
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_tokens_computed(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_tokens_computed
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_publications(
  self : LunaOnlineFramedTelemetry,
) -> UInt64 {
  self.prefix_publications
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_entries(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.prefix_entries
}

///|
pub fn LunaOnlineFramedTelemetry::prefix_pages(
  self : LunaOnlineFramedTelemetry,
) -> Int {
  self.prefix_pages
}

///|
pub fn LunaOnlineFramedPlanTelemetry::last_plan_sequence(
  self : LunaOnlineFramedPlanTelemetry,
) -> UInt64 {
  self.plan_sequence
}

///|
pub fn LunaOnlineFramedPlanTelemetry::last_plan_rows(
  self : LunaOnlineFramedPlanTelemetry,
) -> Int {
  self.plan_rows
}

///|
pub fn LunaOnlineFramedPlanTelemetry::last_plan_tokens(
  self : LunaOnlineFramedPlanTelemetry,
) -> Int {
  self.plan_tokens
}
