///|
/// Returns a scalar-only scheduler snapshot after authenticating this exact
/// production online lease epoch.
pub fn OnlineWorkerLease::luna_scheduler_telemetry(
  self : OnlineWorkerLease,
) -> @core.LunaSchedulerTelemetry raise WorkerServiceError {
  self.require_current().scheduler.luna_telemetry()
}

///|
pub fn OnlineWorkerLease::luna_scheduler_plan_telemetry(
  self : OnlineWorkerLease,
) -> @core.LunaSchedulerPlanTelemetry raise WorkerServiceError {
  self.require_current().scheduler.luna_plan_telemetry()
}
