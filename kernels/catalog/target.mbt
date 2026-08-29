///|
/// Exact CUDA execution target used for startup selection.
///
/// It contains semantic inventory facts only. No ordinal, pointer, context,
/// device name, or raw vendor capability value crosses this boundary.
pub struct DeviceTarget {
  priv compute_major : Int
  priv compute_minor : Int
  priv supports_bf16 : Bool
  priv supports_cublas_lt : Bool
} derive(Debug, Eq)

///|
pub fn DeviceTarget::new(
  compute_major~ : Int,
  compute_minor~ : Int,
  supports_bf16~ : Bool,
  supports_cublas_lt~ : Bool,
) -> DeviceTarget raise CatalogError {
  if compute_major <= 0 || compute_minor < 0 {
    raise InvalidCatalog(InvalidComputeCapability)
  }
  { compute_major, compute_minor, supports_bf16, supports_cublas_lt }
}

///|
pub fn DeviceTarget::from_capability(
  capability : @device.DeviceCapability,
) -> DeviceTarget raise CatalogError {
  DeviceTarget::new(
    compute_major=capability.compute_major(),
    compute_minor=capability.compute_minor(),
    supports_bf16=capability.supports_bf16(),
    supports_cublas_lt=capability.supports_cublas_lt(),
  )
}

///|
/// Returns whether this exact startup target describes the opened context.
pub fn DeviceTarget::matches_capability(
  self : DeviceTarget,
  capability : @device.DeviceCapability,
) -> Bool {
  self.compute_major == capability.compute_major() &&
  self.compute_minor == capability.compute_minor() &&
  self.supports_bf16 == capability.supports_bf16() &&
  self.supports_cublas_lt == capability.supports_cublas_lt()
}

///|
/// Returns whether this exact startup target describes the numeric-capability
/// evidence admitted from the opened context.
pub fn DeviceTarget::matches_numeric_capability(
  self : DeviceTarget,
  capability : @device.DeviceNumericCapability,
) -> Bool {
  self.compute_major == capability.compute_major() &&
  self.compute_minor == capability.compute_minor() &&
  self.supports_bf16 == capability.supports_bf16() &&
  self.supports_cublas_lt == capability.supports_cublas_lt()
}

///|
pub fn DeviceTarget::compute_major(self : DeviceTarget) -> Int {
  self.compute_major
}

///|
pub fn DeviceTarget::compute_minor(self : DeviceTarget) -> Int {
  self.compute_minor
}

///|
pub fn DeviceTarget::supports_bf16(self : DeviceTarget) -> Bool {
  self.supports_bf16
}

///|
pub fn DeviceTarget::supports_cublas_lt(self : DeviceTarget) -> Bool {
  self.supports_cublas_lt
}
