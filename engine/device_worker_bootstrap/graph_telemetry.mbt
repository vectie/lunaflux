///|
/// Projects the concrete child worker's private executor telemetry without
/// exposing which numeric worker implementation owns it.
pub fn DeviceWorkerBootstrapOwner::graph_runtime_telemetry(
  self : DeviceWorkerBootstrapOwner,
) -> @device_step.PagedGraphRuntimeTelemetry raise DeviceWorkerBootstrapError {
  match self.worker {
    Legacy(owner) =>
      owner.graph_runtime_telemetry() catch {
        error => raise Worker(error)
      }
    I8(owner) =>
      owner.graph_runtime_telemetry() catch {
        error => raise I8Worker(error)
      }
    Fp8(owner) =>
      owner.graph_runtime_telemetry() catch {
        error => raise Fp8Worker(error)
      }
  }
}
