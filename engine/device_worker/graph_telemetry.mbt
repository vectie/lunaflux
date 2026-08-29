///|
fn retained_graph_runtime_telemetry(
  resources : DeviceWorkerResources,
) -> @device_step.PagedGraphRuntimeTelemetry raise DeviceWorkerError {
  guard resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  executor.graph_runtime_telemetry()
}

///|
/// Reports the private executor's actual startup-selected graph/eager base,
/// authenticated shape class, and completed hit/miss counters. The returned
/// value is scalar-only and does not expose executor or cleanup authority.
/// Terminal fault/close state remains reportable while the retained executor
/// owner exists.
pub fn DeviceWorkerOwner::graph_runtime_telemetry(
  self : DeviceWorkerOwner,
) -> @device_step.PagedGraphRuntimeTelemetry raise DeviceWorkerError {
  retained_graph_runtime_telemetry(self.resources)
}
