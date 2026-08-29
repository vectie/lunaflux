///|
/// Stable numeric feature admitted from one observed device capability.
///
/// This is a semantic execution feature, not a vendor instruction name and
/// not evidence that a kernel has executed correctly on physical hardware.
pub(all) enum DeviceNumericFeature {
  FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1
  SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1
} derive(Debug, Eq)

///|
/// Stable version of the device numeric-capability record.
pub struct DeviceNumericCapabilityVersion {
  priv value : Int
} derive(Debug, Eq)

///|
pub fn DeviceNumericCapabilityVersion::v1() -> DeviceNumericCapabilityVersion {
  { value: 1 }
}

///|
pub fn DeviceNumericCapabilityVersion::as_int(
  self : DeviceNumericCapabilityVersion,
) -> Int {
  self.value
}

///|
/// SHA-256 identity of one canonical device numeric-capability record.
pub struct DeviceNumericCapabilityDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
pub fn DeviceNumericCapabilityDigest::as_hex(
  self : DeviceNumericCapabilityDigest,
) -> String {
  self.value
}

///|
/// Opaque startup evidence derived from one exact observed device capability.
///
/// Only target-selection facts are retained. Memory capacity remains owned by
/// device planning and cannot silently change numeric support.
pub struct DeviceNumericCapability {
  priv version : DeviceNumericCapabilityVersion
  priv feature : DeviceNumericFeature
  priv compute_major : Int
  priv compute_minor : Int
  priv supports_bf16 : Bool
  priv supports_cublas_lt : Bool
  priv canonical : Bytes
  priv digest : DeviceNumericCapabilityDigest
} derive(Debug, Eq)

///|
pub(all) enum DeviceNumericCapabilityRule {
  InvalidObservedCapability
  UnsupportedArchitecture
  MissingBf16BoundarySupport
} derive(Debug, Eq)

///|
pub(all) suberror DeviceNumericCapabilityError {
  InvalidDeviceNumericCapability(DeviceNumericCapabilityRule)
} derive(Debug, Eq)

///|
fn fp8_e4m3_architecture_allowed_v1(major : Int, minor : Int) -> Bool {
  // Capability policy v1 is intentionally closed over exact target pairs.
  // New architectures require a new reviewed entry plus artifact and physical
  // evidence; numeric ordering must never imply forward compatibility.
  (major == 8 && minor == 9) ||
  (major == 9 && minor == 0) ||
  (major == 12 && minor == 0)
}

///|
fn symmetric_i8_weight_only_architecture_allowed_v1(
  major : Int,
  minor : Int,
) -> Bool {
  // This is a closed software target-admission allowlist, not evidence of
  // physical kernel support or readiness. Ordering never implies support.
  (major == 8 && minor == 9) ||
  (major == 9 && minor == 0) ||
  (major == 12 && minor == 0)
}

///|
fn append_numeric_int(output : Array[Byte], value : Int) -> Unit {
  for byte in Int64::from_int(value).reinterpret_as_uint64().to_le_bytes() {
    output.push(byte)
  }
}

///|
fn encode_device_numeric_capability(
  version : DeviceNumericCapabilityVersion,
  feature : DeviceNumericFeature,
  capability : DeviceCapability,
) -> Bytes {
  let output : Array[Byte] = []
  for byte in b"lunaflux.device-numeric-capability.v1\x00" {
    output.push(byte)
  }
  append_numeric_int(output, version.as_int())
  append_numeric_int(
    output,
    match feature {
      FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1 => 1
      SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1 =>
        2
    },
  )
  append_numeric_int(output, capability.compute_major())
  append_numeric_int(output, capability.compute_minor())
  append_numeric_int(output, if capability.supports_bf16() { 1 } else { 0 })
  append_numeric_int(
    output,
    if capability.supports_cublas_lt() {
      1
    } else {
      0
    },
  )
  Bytes::from_array(output)
}

///|
fn publish_device_numeric_capability(
  feature : DeviceNumericFeature,
  capability : DeviceCapability,
) -> DeviceNumericCapability {
  let version = DeviceNumericCapabilityVersion::v1()
  let canonical = encode_device_numeric_capability(version, feature, capability)
  {
    version,
    feature,
    compute_major: capability.compute_major(),
    compute_minor: capability.compute_minor(),
    supports_bf16: capability.supports_bf16(),
    supports_cublas_lt: capability.supports_cublas_lt(),
    canonical,
    digest: { value: @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) },
  }
}

///|
/// Admits the finite E4M3 W8A8 execution feature from observed inventory.
///
/// This establishes only target support. Exact model numeric contracts,
/// kernels, artifacts, launches, and correctness evidence are separate gates.
pub fn admit_finite_fp8_e4m3_w8a8(
  capability : DeviceCapability,
) -> DeviceNumericCapability raise DeviceNumericCapabilityError {
  if capability.compute_major() <= 0 || capability.compute_minor() < 0 {
    raise InvalidDeviceNumericCapability(InvalidObservedCapability)
  }
  if !fp8_e4m3_architecture_allowed_v1(
      capability.compute_major(),
      capability.compute_minor(),
    ) {
    raise InvalidDeviceNumericCapability(UnsupportedArchitecture)
  }
  if !capability.supports_bf16() {
    raise InvalidDeviceNumericCapability(MissingBf16BoundarySupport)
  }
  let feature = FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1
  publish_device_numeric_capability(feature, capability)
}

///|
/// Admits the symmetric-I8 weight-only v1 software target from observed facts.
///
/// This is inert target-selection evidence only. It proves no artifact,
/// module, launch path, numerical correctness, physical support, or readiness.
pub fn admit_symmetric_i8_weight_only_per_output_channel_v1(
  capability : DeviceCapability,
) -> DeviceNumericCapability raise DeviceNumericCapabilityError {
  if capability.compute_major() <= 0 || capability.compute_minor() < 0 {
    raise InvalidDeviceNumericCapability(InvalidObservedCapability)
  }
  if !symmetric_i8_weight_only_architecture_allowed_v1(
      capability.compute_major(),
      capability.compute_minor(),
    ) {
    raise InvalidDeviceNumericCapability(UnsupportedArchitecture)
  }
  if !capability.supports_bf16() {
    raise InvalidDeviceNumericCapability(MissingBf16BoundarySupport)
  }
  let feature = SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1
  publish_device_numeric_capability(feature, capability)
}

///|
pub fn DeviceNumericCapability::version(
  self : DeviceNumericCapability,
) -> DeviceNumericCapabilityVersion {
  self.version
}

///|
pub fn DeviceNumericCapability::feature(
  self : DeviceNumericCapability,
) -> DeviceNumericFeature {
  self.feature
}

///|
pub fn DeviceNumericCapability::compute_major(
  self : DeviceNumericCapability,
) -> Int {
  self.compute_major
}

///|
pub fn DeviceNumericCapability::compute_minor(
  self : DeviceNumericCapability,
) -> Int {
  self.compute_minor
}

///|
pub fn DeviceNumericCapability::supports_bf16(
  self : DeviceNumericCapability,
) -> Bool {
  self.supports_bf16
}

///|
pub fn DeviceNumericCapability::supports_cublas_lt(
  self : DeviceNumericCapability,
) -> Bool {
  self.supports_cublas_lt
}

///|
pub fn DeviceNumericCapability::canonical_bytes(
  self : DeviceNumericCapability,
) -> Bytes {
  self.canonical
}

///|
pub fn DeviceNumericCapability::digest(
  self : DeviceNumericCapability,
) -> DeviceNumericCapabilityDigest {
  self.digest
}
