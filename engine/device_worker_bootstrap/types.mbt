///|
pub(all) enum DeviceWorkerBootstrapRule {
  SourceDigest
  Recipe
  Locator
  Digest
  ModelIdentity
  OperationCount
  WorkerEnvelope
  InferenceEnvelope
  PageEnvelope
  RecipeLimit
  ReadinessContract
  NumericCapability
  NumericWeightArtifact
  ReleaseIdentity
} derive(Debug, Eq)

///|
pub(all) enum BootstrapRootRole {
  ModelRoot
  KernelRoot
} derive(Debug, Eq)

///|
pub(all) enum BootstrapRootIssue {
  Unavailable
  UnsupportedType
  AuthorityClosed
  Busy
  Failed
} derive(Debug, Eq)

///|
pub(all) suberror DeviceWorkerBootstrapError {
  Invalid(DeviceWorkerBootstrapRule)
  Root(role~ : BootstrapRootRole, issue~ : BootstrapRootIssue)
  Config(@config_file.ModelConfigFileError)
  MistralConfig(@config_reader.ModelConfigReaderError)
  Plan(@plan.PlanValidationError)
  Weights(@device_materialize.DeviceWeightError)
  MistralWeights(@mistral_weights.MistralWeightsError)
  KvLayout(@device_layout.DeviceKvLayoutError)
  DeviceTarget(@catalog.CatalogError)
  DeviceStep(@device_step.DeviceStepError)
  ExecutionManifest(@execution_manifest_file.ExecutionManifestFileError)
  Worker(@device_worker.DeviceWorkerError)
  I8Worker(@i8_device_worker.I8DeviceWorkerError)
  Fp8Worker(@fp8_device_worker_v3.Fp8DeviceWorkerV3Error)
  FusedArtifact(@luna_fused_artifact_admission.FusedArtifactAdmissionError)
  CleanupRoot(role~ : BootstrapRootRole, issue~ : BootstrapRootIssue)
  CleanupWorker(@device_worker.DeviceWorkerError)
  CleanupI8Worker(@i8_device_worker.I8DeviceWorkerError)
  CleanupFp8Worker(@fp8_device_worker_v3.Fp8DeviceWorkerV3Error)
} derive(Debug, Eq)

///|
priv struct BootstrapRoots {
  mut model : @approved_fs.ApprovedRoot?
  mut kernel : @approved_fs.ApprovedRoot?
}

///|
priv enum BootstrapRootAcquisition {
  RootsAcquired(BootstrapRoots)
  RootCleanupRequired(DeviceWorkerBootstrapPreparation)
}

///|
priv enum BootstrapWorkerAuthority {
  LegacyReady(@device_worker.DeviceWorkerOwner)
  LegacyFailed(@device_worker.FailedDeviceWorkerPreparation)
  I8Ready(@i8_device_worker.I8DeviceWorkerOwner)
  I8Failed(@i8_device_worker.FailedI8DeviceWorkerPreparation)
  Fp8Ready(@fp8_device_worker_v3.Fp8DeviceWorkerOwnerV3)
  Fp8Failed(@fp8_device_worker_v3.FailedFp8DeviceWorkerPreparationV3)
}

///|
priv enum BootstrapReadyWorker {
  Legacy(@device_worker.DeviceWorkerOwner)
  I8(@i8_device_worker.I8DeviceWorkerOwner)
  Fp8(@fp8_device_worker_v3.Fp8DeviceWorkerOwnerV3)
}

///|
/// Opaque ready owner. No root, locator, file inspection, execution plan, or
/// native/device resource is exposed.
pub struct DeviceWorkerBootstrapOwner {
  priv worker : BootstrapReadyWorker
}

///|
pub(all) enum DeviceWorkerBootstrapPreparation {
  ChildReady(DeviceWorkerBootstrapOwner)
  CleanupRequired(FailedDeviceWorkerBootstrapPreparation)
}

///|
/// Retryable aggregate authority after preparation and cleanup both fail.
pub struct FailedDeviceWorkerBootstrapPreparation {
  priv roots : BootstrapRoots
  priv mut worker : BootstrapWorkerAuthority?
  priv cause : DeviceWorkerBootstrapError
  priv mut worker_cleanup_failure : DeviceWorkerBootstrapError?
  priv mut model_cleanup_failure : DeviceWorkerBootstrapError?
  priv mut kernel_cleanup_failure : DeviceWorkerBootstrapError?
  priv mut closed : Bool
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::cause(
  self : FailedDeviceWorkerBootstrapPreparation,
) -> DeviceWorkerBootstrapError {
  self.cause
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::cleanup_failure(
  self : FailedDeviceWorkerBootstrapPreparation,
) -> DeviceWorkerBootstrapError? {
  match self.worker_cleanup_failure {
    Some(error) => Some(error)
    None =>
      match self.model_cleanup_failure {
        Some(error) => Some(error)
        None => self.kernel_cleanup_failure
      }
  }
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::worker_cleanup_failure(
  self : FailedDeviceWorkerBootstrapPreparation,
) -> DeviceWorkerBootstrapError? {
  self.worker_cleanup_failure
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::root_cleanup_failure(
  self : FailedDeviceWorkerBootstrapPreparation,
  role : BootstrapRootRole,
) -> DeviceWorkerBootstrapError? {
  match role {
    ModelRoot => self.model_cleanup_failure
    KernelRoot => self.kernel_cleanup_failure
  }
}

///|
pub fn FailedDeviceWorkerBootstrapPreparation::is_closed(
  self : FailedDeviceWorkerBootstrapPreparation,
) -> Bool {
  self.closed
}
