///|
pub enum Fp8DeviceWorkerV3Rule {
  StartupContract
  SourceRecipe
  ModelIdentity
  ReleaseIdentity
  DeviceOrdinal
  DeviceTarget
  DeviceNumericCapability
  ResourceState
} derive(Debug, Eq)

///|
pub enum Fp8DeviceWorkerV3Lifecycle {
  WorkerReady
  WorkerFaulted
  WorkerClosingFailed
  WorkerClosed
} derive(Debug, Eq)

///|
pub suberror Fp8DeviceWorkerV3Error {
  Invalid(Fp8DeviceWorkerV3Rule)
  Protocol(@worker_protocol.WorkerProtocolError)
  Wire(@worker_wire.WorkerWireError)
  Device(@device.DeviceError)
  NumericWeights(@numeric_weight_file.NumericWeightFileError)
  Executor(@device_step.DeviceStepError)
  CleanupContext(@device.DeviceError)
  CleanupNumericWeights(@numeric_weight_file.NumericWeightFileError)
  CleanupExecutor(@device_step.DeviceStepError)
  CompletionAbortFailed(
    primary~ : Fp8DeviceWorkerV3Error,
    abort~ : @worker_wire.WorkerWireError
  )
  InvalidLifecycle(Fp8DeviceWorkerV3Lifecycle)
} derive(Debug, Eq)

///|
priv enum Fp8ExecutorOwnerV3 {
  Complete(@device_step.PagedGraphExecutor)
  Failed(@device_step.FailedPagedGraphPreparation)
}

///|
priv enum Fp8NumericOwnerV3 {
  Raw(@numeric_weight_file.NumericDeviceWeights)
  Authenticated(@numeric_weight_file.AuthenticatedNumericWeightAuthority)
  Failed(@numeric_weight_file.FailedNumericDeviceWeightPreparation)
}

///|
priv struct Fp8DeviceWorkerV3Resources {
  mut executor : Fp8ExecutorOwnerV3?
  mut numeric : Fp8NumericOwnerV3?
  mut context : @device.Context?
}

///|
/// Inert exact plan; construction retains no live device resource.
pub struct Fp8DeviceWorkerPlanV3 {
  priv startup : @worker_wire.WorkerStartupContract
  priv execution : @execution_manifest_file.Fp8PagedExecutionAdmissionV5
  priv expected_total_memory_bytes : Int64
}

///|
/// Thread-confined owner of the reusable mixed PagedV4 executor.
pub struct Fp8DeviceWorkerOwnerV3 {
  priv startup : @worker_wire.WorkerStartupContract
  priv resources : Fp8DeviceWorkerV3Resources
  priv mut lifecycle : Fp8DeviceWorkerV3Lifecycle
}

///|
pub enum Fp8DeviceWorkerPreparationV3 {
  Fp8DeviceWorkerReadyV3(Fp8DeviceWorkerOwnerV3)
  Fp8DeviceWorkerCleanupRequiredV3(FailedFp8DeviceWorkerPreparationV3)
}

///|
pub struct FailedFp8DeviceWorkerPreparationV3 {
  priv resources : Fp8DeviceWorkerV3Resources
  priv cause : Fp8DeviceWorkerV3Error
  priv mut cleanup_failure : Fp8DeviceWorkerV3Error
  priv mut lifecycle : Fp8DeviceWorkerV3Lifecycle
}

///|
pub fn FailedFp8DeviceWorkerPreparationV3::cause(
  self : Self,
) -> Fp8DeviceWorkerV3Error {
  self.cause
}

///|
pub fn FailedFp8DeviceWorkerPreparationV3::cleanup_failure(
  self : Self,
) -> Fp8DeviceWorkerV3Error {
  self.cleanup_failure
}

///|
pub fn FailedFp8DeviceWorkerPreparationV3::is_closed(self : Self) -> Bool {
  self.lifecycle == WorkerClosed
}
