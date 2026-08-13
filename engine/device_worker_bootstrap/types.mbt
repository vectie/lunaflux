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
  Plan(@plan.PlanValidationError)
  Weights(@device_materialize.DeviceWeightError)
  KvLayout(@device_layout.DeviceKvLayoutError)
  DeviceTarget(@catalog.CatalogError)
  DeviceStep(@device_step.DeviceStepError)
  ExecutionManifest(@execution_manifest_file.ExecutionManifestFileError)
  Worker(@device_worker.DeviceWorkerError)
  CleanupRoot(role~ : BootstrapRootRole, issue~ : BootstrapRootIssue)
  CleanupWorker(@device_worker.DeviceWorkerError)
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
  Ready(@device_worker.DeviceWorkerOwner)
  Failed(@device_worker.FailedDeviceWorkerPreparation)
}

///|
/// Opaque ready owner. No root, locator, file inspection, execution plan, or
/// native/device resource is exposed.
pub struct DeviceWorkerBootstrapOwner {
  priv worker : @device_worker.DeviceWorkerOwner
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
