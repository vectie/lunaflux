///|
pub(all) enum DeviceWorkerRule {
  StartupContract
  ModelIdentity
  WeightEnvelope
  DeviceOrdinal
  DeviceTarget
  ResourceState
} derive(Debug, Eq)

///|
/// Stable device-worker lifecycle. Only `WorkerReady` may emit readiness.
pub(all) enum DeviceWorkerLifecycle {
  WorkerReady
  WorkerFaulted
  WorkerClosingFailed
  WorkerClosed
} derive(Debug, Eq)

///|
/// Payload-safe device-worker preparation and cleanup failures.
pub(all) suberror DeviceWorkerError {
  Invalid(DeviceWorkerRule)
  Bootstrap(@device_step.DeviceBootstrapError)
  Protocol(@worker_protocol.WorkerProtocolError)
  Wire(@worker_wire.WorkerWireError)
  Device(@device.DeviceError)
  Weights(@device_materialize.DeviceWeightError)
  NumericWeights(@numeric_weight_file.NumericWeightFileError)
  Executor(@device_step.DeviceStepError)
  CleanupContext(@device.DeviceError)
  CleanupWeights(@device_materialize.DeviceWeightError)
  CleanupNumericWeights(@numeric_weight_file.NumericWeightFileError)
  CleanupExecutor(@device_step.DeviceStepError)
  CompletionAbortFailed(
    primary~ : DeviceWorkerError,
    abort~ : @worker_wire.WorkerWireError
  )
  InvalidLifecycle(DeviceWorkerLifecycle)
} derive(Debug, Eq)

///|
priv enum ExecutorOwner {
  Complete(@device_step.PagedGraphExecutor)
  Failed(@device_step.FailedPagedGraphPreparation)
}

///|
priv enum WeightOwner {
  Complete(@device_materialize.DeviceWeights)
  Failed(@device_materialize.FailedDeviceWeightPreparation)
  NumericComplete(@numeric_weight_file.NumericDeviceWeights)
  NumericFailed(@numeric_weight_file.FailedNumericDeviceWeightPreparation)
}

///|
priv struct DeviceWorkerResources {
  mut executor : ExecutorOwner?
  mut weights : WeightOwner?
  mut context : @device.Context?
  mut numeric_weights_owned_by_executor : Bool
}

///|
/// Opaque immutable admission for one exact device-worker bootstrap. It owns
/// no device or native resource and exposes no mutable execution owner.
pub struct DeviceWorkerPlan {
  priv startup : @worker_wire.WorkerStartupContract
  priv model_plan : @plan.ModelPlan
  priv weight_inspection : @device_materialize.DeviceWeightFileInspection?
  priv numeric_weight_inspection : @numeric_weight_file.NumericWeightFileInspection?
  priv execution : @execution_manifest_file.PagedExecutionAdmission
  priv manifest : @device_step.DeviceWorkerBootstrapManifest
  priv sampling_policy : @device_step.PagedSamplingExecutionPolicy
  priv fused_residual_span : @device_step.ApprovedFusedResidualRmsNormSpan?
  priv fused_qkv_readonly_artifacts : @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts?
  priv fused_qkv_readonly_source : @luna_paged_composite_artifact_admission.DigestPinnedFusedProductionRuntime?
}

///|
/// Thread-confined aggregate owner of one loaded model and prepared paged
/// executor. It exposes no context, allocation, weight, executor, module,
/// function, stream, or native handle.
pub struct DeviceWorkerOwner {
  priv startup : @worker_wire.WorkerStartupContract
  priv resources : DeviceWorkerResources
  priv mut lifecycle : DeviceWorkerLifecycle
}

///|
pub enum DeviceWorkerPreparation {
  DeviceWorkerReady(DeviceWorkerOwner)
  DeviceWorkerCleanupRequired(FailedDeviceWorkerPreparation)
}

///|
/// Aggregate cleanup authority retained after preparation and its first
/// dependency-ordered cleanup both fail.
pub struct FailedDeviceWorkerPreparation {
  priv resources : DeviceWorkerResources
  priv cause : DeviceWorkerError
  priv mut cleanup_failure : DeviceWorkerError
  priv mut lifecycle : DeviceWorkerLifecycle
}

///|
pub fn DeviceWorkerOwner::lifecycle(
  self : DeviceWorkerOwner,
) -> DeviceWorkerLifecycle {
  self.lifecycle
}

///|
pub fn FailedDeviceWorkerPreparation::cause(
  self : FailedDeviceWorkerPreparation,
) -> DeviceWorkerError {
  self.cause
}

///|
pub fn FailedDeviceWorkerPreparation::cleanup_failure(
  self : FailedDeviceWorkerPreparation,
) -> DeviceWorkerError {
  self.cleanup_failure
}

///|
pub fn FailedDeviceWorkerPreparation::is_closed(
  self : FailedDeviceWorkerPreparation,
) -> Bool {
  self.lifecycle == WorkerClosed
}
