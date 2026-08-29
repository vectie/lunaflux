///|
/// Reports only the retained executor's actual startup selection and completed
/// execution counters. It does not expose execution or cleanup authority.
pub fn Fp8DeviceWorkerOwnerV3::graph_runtime_telemetry(
  self : Self,
) -> @device_step.PagedGraphRuntimeTelemetry raise Fp8DeviceWorkerV3Error {
  guard self.resources.executor is Some(Complete(executor)) else {
    raise Invalid(ResourceState)
  }
  executor.graph_runtime_telemetry()
}
