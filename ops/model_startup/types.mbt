///|
pub(all) enum LunaModelPreparationPrerequisite {
  DigestPinnedRuntimeDescriptorAdmission
} derive(Debug, Eq)

///|
pub(all) enum LunaModelPhysicalGate {
  UnsupportedPlatform
  DriverLibraryMissing
  DriverAbiIncomplete
  DriverInitializationFailed
  NoDevices
  InventoryFailed
  AssignedDeviceMissing
  DeviceTargetMismatch
} derive(Debug, Eq)

///|
pub(all) enum LunaModelPreparationOutcome {
  LunaModelPreflightComplete
  LunaModelNotPrepared(LunaModelPreparationPrerequisite)
  LunaModelPhysicalGate(LunaModelPhysicalGate)
} derive(Debug, Eq)

///|
pub(all) enum LunaModelStartupStage {
  ModelRootAdmission
  KernelRootAdmission
  DescriptorLocatorAdmission
  DescriptorDigestAdmission
  DescriptorLimitAdmission
  RuntimeDescriptorAdmission
  ModelRootClose
  KernelRootClose
} derive(Debug, Eq)

///|
pub(all) enum LunaModelStartupCleanup {
  ModelRoot
  KernelRoot
  BothRoots
} derive(Debug, Eq)

///|
pub(all) suberror LunaModelStartupError {
  Failed(LunaModelStartupStage)
  FailedAndClose(
    primary~ : LunaModelStartupStage,
    cleanup~ : LunaModelStartupCleanup
  )
  CloseFailed(LunaModelStartupCleanup)
} derive(Debug, Eq)

///|
/// Result of a fully digest-pinned inert admission and an exact device probe.
/// Once descriptor admission succeeds, its host-side evidence remains present
/// for planning and inspection even when hardware is unavailable, missing, or
/// incompatible. A physical gate still prevents activation and is never
/// service-readiness evidence.
pub struct LunaPinnedModelPreflight {
  priv admission : @descriptor_file.RuntimeDescriptorAdmission?
  priv outcome : LunaModelPreparationOutcome
}

///|
pub fn LunaPinnedModelPreflight::admission(
  self : LunaPinnedModelPreflight,
) -> @descriptor_file.RuntimeDescriptorAdmission? {
  self.admission
}

///|
pub fn LunaPinnedModelPreflight::outcome(
  self : LunaPinnedModelPreflight,
) -> LunaModelPreparationOutcome {
  self.outcome
}
