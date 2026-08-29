///|
/// Exact post-join transfer authority for one schema-v4 I8 executor build.
/// Before this value exists, the caller retains the loaded-weight owner. Once
/// admitted, this value is the sole close authority until executor preparation
/// consumes it.
pub struct I8PagedExecutorTransferV4 {
  priv admission : I8PagedExecutionAdmissionV4
  priv loaded : @device_step.I8LoadedExecutionBlueprint
}

///|
pub fn I8PagedExecutionAdmissionV4::bind_loaded_executor_transfer(
  self : I8PagedExecutionAdmissionV4,
  loaded : @device_step.I8LoadedExecutionBlueprint,
) -> I8PagedExecutorTransferV4 raise ExecutionManifestFileError {
  if !physical_equal(loaded.blueprint(), self.blueprint()) ||
    loaded.bootstrap_manifest() != self.bootstrap_manifest() ||
    loaded.weight_artifact_digest() != self.weight_artifact_digest() {
    raise AdmissionRejected(Blueprint)
  }
  { admission: self, loaded }
}

///|
pub fn I8PagedExecutorTransferV4::close(
  self : I8PagedExecutorTransferV4,
) -> Unit raise ExecutionManifestFileError {
  self.loaded.close() catch {
    _ => raise AdmissionRejected(Blueprint)
  }
}

///|
/// Prepares the production-I8 software executor from the exact schema-v4
/// transfer authority. The lower constructor takes loaded-weight ownership at
/// function entry on every success and failure path.
///
/// This is host-side construction and serialized launch authority, not a
/// physical CUDA correctness/readiness claim. Release publication remains
/// gated on the repository's physical canary, sanitizer, leak, and benchmark
/// evidence.
pub fn prepare_paged_i8_v4_executor(
  context : @device.Context,
  transfer : I8PagedExecutorTransferV4,
  model_generation : @worker_protocol.ModelPlanGeneration,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
) -> @device_step.PagedGraphPreparation raise ExecutionManifestFileError {
  let admission = transfer.admission
  @device_step.prepare_i8_paged_graph_executor_v1(
    context,
    admission.model_plan(),
    model_generation,
    admission.device_numeric_capability(),
    transfer.loaded,
    has_accepted_predecessor~,
    accepted_predecessor_value~,
  ) catch {
    _ => raise AdmissionRejected(Blueprint)
  }
}
