///|
/// Resolves the authenticated bootstrap sampling placement against the exact
/// admitted graph. Absence is represented on the wire as host sampling;
/// explicit CUDA greedy never falls back when its embedded LM-head reducer is
/// unavailable or mismatched.
pub fn admit_bootstrap_sampling_policy(
  source : @worker_wire.EncodedBootstrapSource,
  model_plan : @plan.ModelPlan,
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> @device_step.PagedSamplingExecutionPolicy raise DeviceWorkerError {
  match source.kernel().sampling_runtime() {
    HostSamplingRuntime => PagedHostSampling
    EmbeddedCudaGreedyV1 =>
      @device_step.admit_paged_embedded_cuda_greedy_sampling(
        model_plan,
        execution.blueprint(),
        execution.artifacts(),
      ) catch {
        error => raise Executor(error)
      }
  }
}
