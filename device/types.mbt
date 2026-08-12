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
  compute_major : Int
  compute_minor : Int
  total_memory_bytes : Int64
  supports_bf16 : Bool
  supports_cublas_lt : Bool
} derive(Debug, Eq)

///|
pub fn DeviceCapability::compute_major(self : DeviceCapability) -> Int {
  self.compute_major
}

///|
pub fn DeviceCapability::compute_minor(self : DeviceCapability) -> Int {
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
  ordinal : Int
  name : String
  capability : DeviceCapability
} derive(Debug, Eq)

///|
pub fn DeviceInfo::ordinal(self : DeviceInfo) -> Int {
  self.ordinal
}

///|
pub fn DeviceInfo::name(self : DeviceInfo) -> String {
  self.name
}

///|
pub fn DeviceInfo::capability(self : DeviceInfo) -> DeviceCapability {
  self.capability
}

///|
pub(all) enum ProbeResult {
  Available(driver_version~ : Int, devices~ : ReadOnlyArray[DeviceInfo])
  Unavailable(UnavailableReason)
} derive(Debug, Eq)
