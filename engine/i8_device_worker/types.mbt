///|
pub enum I8DeviceWorkerRule {
  StartupContract
  ModelIdentity
  NumericWeightEnvelope
  DeviceOrdinal
  DeviceTarget
  DeviceNumericCapability
  ResourceState
} derive(Debug, Eq)

///|
pub enum I8DeviceWorkerLifecycle {
  WorkerReady
  WorkerFaulted
  WorkerClosingFailed
  WorkerClosed
} derive(Debug, Eq)

///|
pub suberror I8DeviceWorkerError {
  Invalid(I8DeviceWorkerRule)
  Bootstrap(@device_step.DeviceBootstrapError)
  Protocol(@worker_protocol.WorkerProtocolError)
  Wire(@worker_wire.WorkerWireError)
  Device(@device.DeviceError)
  NumericWeights(@numeric_weight_file.NumericWeightFileError)
  ExecutionManifest(@execution_manifest_file.ExecutionManifestFileError)
  Executor(@device_step.DeviceStepError)
  CleanupContext(@device.DeviceError)
  CleanupNumericWeights(@numeric_weight_file.NumericWeightFileError)
  CleanupExecutionManifest(@execution_manifest_file.ExecutionManifestFileError)
  CleanupExecutor(@device_step.DeviceStepError)
  CompletionAbortFailed(
    primary~ : I8DeviceWorkerError,
    abort~ : @worker_wire.WorkerWireError
  )
  InvalidLifecycle(I8DeviceWorkerLifecycle)
} derive(Debug, Eq)

///|
priv enum I8ExecutorOwner {
  Complete(@device_step.PagedGraphExecutor)
  Failed(@device_step.FailedPagedGraphPreparation)
}

///|
priv enum I8NumericOwner {
  Raw(@numeric_weight_file.NumericDeviceWeights)
  Authenticated(@numeric_weight_file.AuthenticatedNumericWeightAuthority)
  Loaded(@device_step.I8LoadedExecutionBlueprint)
  ExecutorTransfer(@execution_manifest_file.I8PagedExecutorTransferV4)
  Failed(@numeric_weight_file.FailedNumericDeviceWeightPreparation)
}

///|
priv enum I8OwnerSlotPhase {
  OwnerSlotsEmpty
  RawNumericOwner
  AuthenticatedNumericOwner
  LoadedNumericOwner
  ExecutorTransferOwner
  ExecutorConstructionOwner
  ReadyExecutorOwner
  FailedNumericOwner
  FailedExecutorOwner
} derive(Eq)

///|
priv struct I8DeviceWorkerResources {
  mut executor : I8ExecutorOwner?
  mut numeric : I8NumericOwner?
  mut context : @device.Context?
  mut owner_phase : I8OwnerSlotPhase
}

///|
/// Opaque inert admission for one exact production-I8 worker. It owns no
/// filesystem, device, allocation, module, function, or execution resource.
pub struct I8DeviceWorkerPlan {
  priv startup : @worker_wire.WorkerStartupContract
  priv model_plan : @plan.ModelPlan
  priv inspection : @numeric_weight_file.NumericWeightFileInspection
  priv execution : @execution_manifest_file.I8PagedExecutionAdmissionV4
  priv expected_total_memory_bytes : Int64
}

///|
/// Thread-confined aggregate owner. No context, allocation, loaded-weight
/// authority, module, function, stream, or executor handle is exposed.
pub struct I8DeviceWorkerOwner {
  priv startup : @worker_wire.WorkerStartupContract
  priv resources : I8DeviceWorkerResources
  priv mut lifecycle : I8DeviceWorkerLifecycle
}

///|
pub enum I8DeviceWorkerPreparation {
  I8DeviceWorkerReady(I8DeviceWorkerOwner)
  I8DeviceWorkerCleanupRequired(FailedI8DeviceWorkerPreparation)
}

///|
/// Retryable aggregate cleanup authority after preparation and the first
/// dependency-ordered cleanup both fail.
pub struct FailedI8DeviceWorkerPreparation {
  priv resources : I8DeviceWorkerResources
  priv cause : I8DeviceWorkerError
  priv mut cleanup_failure : I8DeviceWorkerError
  priv mut lifecycle : I8DeviceWorkerLifecycle
}

///|
pub fn I8DeviceWorkerOwner::lifecycle(
  self : I8DeviceWorkerOwner,
) -> I8DeviceWorkerLifecycle {
  self.lifecycle
}

///|
pub fn FailedI8DeviceWorkerPreparation::cause(
  self : FailedI8DeviceWorkerPreparation,
) -> I8DeviceWorkerError {
  self.cause
}

///|
pub fn FailedI8DeviceWorkerPreparation::cleanup_failure(
  self : FailedI8DeviceWorkerPreparation,
) -> I8DeviceWorkerError {
  self.cleanup_failure
}

///|
pub fn FailedI8DeviceWorkerPreparation::is_closed(
  self : FailedI8DeviceWorkerPreparation,
) -> Bool {
  self.lifecycle == WorkerClosed
}
