///|
/// Returns the allocation-free logical graph report for the exact last
/// completed rank plan. Collectives are currently interleaved outside the
/// immutable kernel queue, so the TP owner is prepared eager-only and has no
/// admitted graph shape class; both facts are reported rather than inferred.
pub fn TensorParallelDeviceWorkerOwner::graph_runtime_telemetry(
  self : TensorParallelDeviceWorkerOwner,
  sequence : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise TensorParallelDeviceWorkerError {
  if self.lifecycle != WorkerReady ||
    sequence == 0UL ||
    sequence != self.last_graph_sequence ||
    (self.graph_hits == 0UL && self.graph_misses == 0UL) {
    raise Invalid(ResourceState)
  }
  match self.resources.executor.resource.execution_mode() {
    OrderedKernelCaptured =>
      @worker_wire.WorkerGraphTelemetry::without_shape(
        sequence~,
        path=WorkerGraphCaptured,
        graph_hits=self.graph_hits,
        graph_misses=self.graph_misses,
      ) catch {
        error => raise Wire(error)
      }
    OrderedKernelEager =>
      @worker_wire.WorkerGraphTelemetry::without_shape(
        sequence~,
        path=WorkerGraphOrderedEager,
        graph_hits=self.graph_hits,
        graph_misses=self.graph_misses,
      ) catch {
        error => raise Wire(error)
      }
  }
}
