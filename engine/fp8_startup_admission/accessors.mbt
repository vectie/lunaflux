///|
pub fn Fp8StartupAdmission::version(
  self : Fp8StartupAdmission,
) -> Fp8StartupAdmissionVersion {
  self.version
}

///|
pub fn Fp8StartupAdmission::model_plan(
  self : Fp8StartupAdmission,
) -> @plan.ModelPlan {
  self.plan
}

///|
pub fn Fp8StartupAdmission::device_numeric_capability(
  self : Fp8StartupAdmission,
) -> @device.DeviceNumericCapability {
  self.device_numeric
}

///|
pub fn Fp8StartupAdmission::kernel_artifacts(
  self : Fp8StartupAdmission,
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts {
  self.kernels
}

///|
pub fn Fp8StartupAdmission::canonical_bytes(
  self : Fp8StartupAdmission,
) -> Bytes {
  self.canonical
}

///|
pub fn Fp8StartupAdmission::digest(
  self : Fp8StartupAdmission,
) -> Fp8StartupAdmissionDigest {
  self.digest
}
