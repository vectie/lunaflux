///|
pub struct Fp8StartupAdmissionVersion(Int) derive(Debug, Eq)

///|
pub fn Fp8StartupAdmissionVersion::v1() -> Fp8StartupAdmissionVersion {
  Fp8StartupAdmissionVersion(1)
}

///|
pub fn Fp8StartupAdmissionVersion::as_int(
  self : Fp8StartupAdmissionVersion,
) -> Int {
  self.0
}

///|
pub struct Fp8StartupAdmissionDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8StartupAdmissionDigest::as_hex(
  self : Fp8StartupAdmissionDigest,
) -> String {
  self.0
}

///|
pub struct Fp8StartupAdmissionLimits {
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn Fp8StartupAdmissionLimits::new(
  max_canonical_bytes~ : Int,
) -> Fp8StartupAdmissionLimits raise Fp8StartupAdmissionError {
  if max_canonical_bytes <= 0 || max_canonical_bytes > 1024 * 1024 {
    raise Fp8StartupRejected(Limit)
  }
  { max_canonical_bytes, }
}

///|
pub(all) enum Fp8StartupAdmissionRule {
  Limit
  ModelPlan
  DeviceEvidence
  KernelEvidence
  EntryBinding
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8StartupAdmissionError {
  Fp8StartupRejected(Fp8StartupAdmissionRule)
} derive(Debug, Eq)

///|
/// Pure, inert startup evidence for one exact finite-E4M3 execution path.
///
/// This owns no device context, loaded module, stream, mutable executor, or
/// readiness state. A future physical backend must consume and reauthenticate
/// it before any execution claim exists.
pub struct Fp8StartupAdmission {
  priv version : Fp8StartupAdmissionVersion
  priv plan : @plan.ModelPlan
  priv device_numeric : @device.DeviceNumericCapability
  priv kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts
  priv canonical : Bytes
  priv digest : Fp8StartupAdmissionDigest
}

///|
pub struct Fp8StartupAdmissionV2Version(Int) derive(Debug, Eq)

///|
pub fn Fp8StartupAdmissionV2Version::v2() -> Fp8StartupAdmissionV2Version {
  Fp8StartupAdmissionV2Version(2)
}

///|
pub fn Fp8StartupAdmissionV2Version::as_int(
  self : Fp8StartupAdmissionV2Version,
) -> Int {
  self.0
}

///|
pub struct Fp8StartupAdmissionV2Digest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8StartupAdmissionV2Digest::as_hex(
  self : Fp8StartupAdmissionV2Digest,
) -> String {
  self.0
}

///|
/// Pure startup evidence for staged FP8 plans. The distinct type prevents a
/// legacy v1 startup admission from being replayed into runtime recipe v2.
pub struct Fp8StartupAdmissionV2 {
  priv version : Fp8StartupAdmissionV2Version
  priv base : Fp8StartupAdmission
  priv canonical : Bytes
  priv digest : Fp8StartupAdmissionV2Digest
}

///|
pub fn Fp8StartupAdmissionV2::version(
  self : Fp8StartupAdmissionV2,
) -> Fp8StartupAdmissionV2Version {
  self.version
}

///|
pub fn Fp8StartupAdmissionV2::model_plan(
  self : Fp8StartupAdmissionV2,
) -> @plan.ModelPlan {
  self.base.model_plan()
}

///|
pub fn Fp8StartupAdmissionV2::device_numeric_capability(
  self : Fp8StartupAdmissionV2,
) -> @device.DeviceNumericCapability {
  self.base.device_numeric_capability()
}

///|
pub fn Fp8StartupAdmissionV2::kernel_artifacts(
  self : Fp8StartupAdmissionV2,
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts {
  self.base.kernel_artifacts()
}

///|
pub fn Fp8StartupAdmissionV2::canonical_bytes(
  self : Fp8StartupAdmissionV2,
) -> Bytes {
  self.canonical
}

///|
pub fn Fp8StartupAdmissionV2::digest(
  self : Fp8StartupAdmissionV2,
) -> Fp8StartupAdmissionV2Digest {
  self.digest
}
