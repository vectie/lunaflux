///|
/// Reports only the retained executor's actual startup selection and completed
/// execution counters. It does not expose execution or cleanup authority.
pub fn I8DeviceWorkerOwner::graph_runtime_telemetry(
  self : I8DeviceWorkerOwner,
) -> @device_step.PagedGraphRuntimeTelemetry raise I8DeviceWorkerError {
  guard self.resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  executor.graph_runtime_telemetry()
}
