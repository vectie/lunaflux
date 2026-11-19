///|
/// Stable reasons why the CUDA execution backend cannot be used. Values are
/// deliberately independent of CUDA result numbers and vendor messages.
pub(all) enum UnavailableReason {
  UnsupportedPlatform
  DriverLibraryMissing
  DriverAbiIncomplete
  DriverInitializationFailed
  NoDevices
  InventoryFailed
} derive(Debug, Eq)

///|
pub struct DeviceCapability {
  compute_major : Int32
  compute_minor : Int32
  total_memory_bytes : Int64
  supports_bf16 : Bool
  supports_cublas_lt : Bool
} derive(Debug, Eq)

///|
pub(all) enum DeviceCapabilityObservationRule {
  InvalidComputeCapability
  InvalidTotalMemory
} derive(Debug, Eq)

///|
pub(all) suberror DeviceCapabilityObservationError {
  InvalidDeviceCapabilityObservation(DeviceCapabilityObservationRule)
} derive(Debug, Eq)

///|
/// Constructs a semantic capability snapshot from already-observed inventory.
///
/// This grants no device, context, or kernel authority. Native probing remains
/// the production source; adapters and tests must still pass the resulting
/// value through every later target and numeric admission gate.
pub fn DeviceCapability::from_observed_facts(
  compute_major~ : Int32,
  compute_minor~ : Int32,
  total_memory_bytes~ : Int64,
  supports_bf16~ : Bool,
  supports_cublas_lt~ : Bool,
) -> DeviceCapability raise DeviceCapabilityObservationError {
  if compute_major <= 0 || compute_minor < 0 {
    raise InvalidDeviceCapabilityObservation(InvalidComputeCapability)
  }
  if total_memory_bytes <= 0L {
    raise InvalidDeviceCapabilityObservation(InvalidTotalMemory)
  }
  {
    compute_major,
    compute_minor,
    total_memory_bytes,
    supports_bf16,
    supports_cublas_lt,
  }
}

///|
pub fn DeviceCapability::compute_major(self : DeviceCapability) -> Int32 {
  self.compute_major
}

///|
pub fn DeviceCapability::compute_minor(self : DeviceCapability) -> Int32 {
  self.compute_minor
}

///|
pub fn DeviceCapability::total_memory_bytes(self : DeviceCapability) -> Int64 {
  self.total_memory_bytes
}

///|
pub fn DeviceCapability::supports_bf16(self : DeviceCapability) -> Bool {
  self.supports_bf16
}

///|
pub fn DeviceCapability::supports_cublas_lt(self : DeviceCapability) -> Bool {
  self.supports_cublas_lt
}

///|
pub struct DeviceInfo {
  ordinal : Int32
  name : String
  uuid : String
  pci : String
  capability : DeviceCapability
} derive(Debug, Eq)

///|
pub fn DeviceInfo::ordinal(self : DeviceInfo) -> Int32 {
  self.ordinal
}

///|
pub fn DeviceInfo::name(self : DeviceInfo) -> String {
  self.name
}

///|
pub fn DeviceInfo::uuid(self : DeviceInfo) -> String {
  self.uuid
}

///|
pub fn DeviceInfo::pci(self : DeviceInfo) -> String {
  self.pci
}

///|
pub fn DeviceInfo::capability(self : DeviceInfo) -> DeviceCapability {
  self.capability
}

///|
pub(all) enum ProbeResult {
  Available(driver_version~ : Int32, devices~ : ReadOnlyArray[DeviceInfo])
  Unavailable(UnavailableReason)
} derive(Debug, Eq)

///|
pub extend UnavailableReason with @debug.Debug::{to_repr}

///|
pub extend UnavailableReason with Eq::{not_equal, equal}

///|
pub extend DeviceCapability with @debug.Debug::{to_repr}

///|
pub extend DeviceCapability with Eq::{not_equal, equal}

///|
pub extend DeviceCapabilityObservationRule with @debug.Debug::{to_repr}

///|
pub extend DeviceCapabilityObservationRule with Eq::{not_equal, equal}

///|
pub extend DeviceCapabilityObservationError with @debug.Debug::{to_repr}

///|
pub extend DeviceCapabilityObservationError with Eq::{not_equal, equal}

///|
pub extend DeviceInfo with @debug.Debug::{to_repr}

///|
pub extend DeviceInfo with Eq::{not_equal, equal}

///|
pub extend ProbeResult with @debug.Debug::{to_repr}

///|
pub extend ProbeResult with Eq::{not_equal, equal}
