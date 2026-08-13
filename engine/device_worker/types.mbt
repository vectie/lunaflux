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
  WorkerClosingFailed
  WorkerClosed
} derive(Debug, Eq)

///|
/// Payload-safe device-worker preparation and cleanup failures.
pub(all) suberror DeviceWorkerError {
  Invalid(DeviceWorkerRule)
  Bootstrap(@device_step.DeviceBootstrapError)
  Protocol(@worker_protocol.WorkerProtocolError)
  Device(@device.DeviceError)
  Weights(@device_materialize.DeviceWeightError)
  Executor(@device_step.DeviceStepError)
  CleanupContext(@device.DeviceError)
  CleanupWeights(@device_materialize.DeviceWeightError)
  CleanupExecutor(@device_step.DeviceStepError)
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
}

///|
priv struct DeviceWorkerResources {
  mut executor : ExecutorOwner?
  mut weights : WeightOwner?
  mut context : @device.Context?
}

///|
/// Opaque immutable admission for one exact device-worker bootstrap. It owns
/// no device or native resource and exposes no mutable execution owner.
pub struct DeviceWorkerPlan {
  priv startup : @worker_wire.WorkerStartupContract
  priv model : @spec.LlamaModelMetadata
  priv model_path : String
  priv weight_file_limits : @device_materialize.DeviceWeightFileLimits
  priv weight_alignment_bytes : Int64
  priv max_weight_arena_bytes : Int64
  priv model_plan : @plan.ModelPlan
  priv layout : @device_layout.DeviceKvLayout
  priv device_step_limits : @device_step.DeviceStepLimits
  priv static_plan : @device_plan.StaticDevicePlan
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv profile_id : @launch_contract.KernelProfileId
  priv contracts : @launch_contract.PagedKvLaunchContractSet
  priv artifacts : @artifact.KernelArtifactBundle
  priv manifest : @device_step.DeviceWorkerBootstrapManifest
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
pub(all) enum DeviceWorkerPreparation {
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
