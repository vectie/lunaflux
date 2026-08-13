///|
/// Startup capacity for reusable CSR/SoA step descriptors.
pub struct DeviceStepLimits {
  priv max_rows : Int
  priv max_tokens : Int
  priv max_pages : Int
  priv max_sequence_tokens : Int
  priv max_token_id : Int
}

///|
/// Worker-confined owner of fixed host staging and matching device storage.
/// The borrowed context must remain open until this owner is closed.
pub struct DeviceStepOwner {
  priv context : @device.Context
  priv identity : @spec.ModelIdentity
  priv model_generation_value : UInt64
  priv layout : @device_layout.DeviceKvLayout
  priv required_capabilities : FixedArray[@plan.KernelCapabilityId]
  priv limits : DeviceStepLimits
  priv host : DeviceStepHostBuffers
  priv resources : DeviceStepResources
  priv mut lifecycle : DeviceStepLifecycle
  priv mut stage_epoch : UInt64
  priv mut has_accepted_predecessor : Bool
  priv mut accepted_predecessor_value : UInt64
  priv mut staged_sequence_value : UInt64
}

///|
/// Exact logical publication proof for one uploaded step.
#valtype
pub struct StagedDeviceStep(DeviceStepOwner, UInt64)

///|
pub(all) enum DeviceStepPreparation {
  Ready(DeviceStepOwner)
  CleanupRequired(FailedDeviceStepPreparation)
}

///|
/// Opaque cleanup authority retained after partial construction cleanup fails.
pub struct FailedDeviceStepPreparation {
  priv resources : PartialDeviceStepResources
  priv cause : DeviceStepError
  priv mut cleanup_failure : DeviceStepError
  priv mut lifecycle : DeviceStepLifecycle
}

///|
#valtype
priv struct DeviceStepSummary {
  sequence_value : UInt64
  prefill_count : Int
  decode_count : Int
  row_count : Int
  token_count : Int
  page_count : Int
}

///|
pub fn DeviceStepLimits::max_rows(self : DeviceStepLimits) -> Int {
  self.max_rows
}

///|
pub fn DeviceStepLimits::max_tokens(self : DeviceStepLimits) -> Int {
  self.max_tokens
}

///|
pub fn DeviceStepLimits::max_pages(self : DeviceStepLimits) -> Int {
  self.max_pages
}

///|
pub fn DeviceStepLimits::max_sequence_tokens(self : DeviceStepLimits) -> Int {
  self.max_sequence_tokens
}

///|
pub fn DeviceStepLimits::max_token_id(self : DeviceStepLimits) -> Int {
  self.max_token_id
}

///|
pub fn DeviceStepOwner::lifecycle(
  self : DeviceStepOwner,
) -> DeviceStepLifecycle {
  self.lifecycle
}

///|
pub fn FailedDeviceStepPreparation::cause(
  self : FailedDeviceStepPreparation,
) -> DeviceStepError {
  self.cause
}

///|
pub fn FailedDeviceStepPreparation::cleanup_failure(
  self : FailedDeviceStepPreparation,
) -> DeviceStepError {
  self.cleanup_failure
}

///|
pub fn FailedDeviceStepPreparation::is_closed(
  self : FailedDeviceStepPreparation,
) -> Bool {
  self.lifecycle == Closed
}
