///|
fn validate_staged_plan_v2(
  plan : @plan.ModelPlan,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
) -> Unit raise Fp8StartupAdmissionError {
  let entries = kernels.manifest().entries()
  if entries.is_empty() {
    raise Fp8StartupRejected(EntryBinding)
  }
  for entry in entries {
    let index = entry.operation().as_int()
    guard plan.operations().get(index) is Some(operation) &&
      plan.numeric_binding().operations().get(index) is Some(binding) else {
      raise Fp8StartupRejected(EntryBinding)
    }
    let policy = binding.execution().activation_scale()
    let stages = policy.stage_count()
    if stages < 1 ||
      stages > 2 ||
      operation.workspace_bytes() != stages.to_int64() * 4L {
      raise Fp8StartupRejected(EntryBinding)
    }
    for ordinal in 0..<stages {
      if policy.stage_at(ordinal) is None {
        raise Fp8StartupRejected(EntryBinding)
      }
    }
  }
}

///|
fn encode_fp8_startup_admission_v2(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  maximum : Int,
) -> Bytes raise Fp8StartupAdmissionError {
  let output : Array[Byte] = []
  for byte in b"lunaflux.fp8-startup-admission.v2\x00" {
    output.push(byte)
  }
  append_startup_int(output, 2)
  append_startup_text(output, plan.identity().content().as_hex())
  append_startup_text(output, plan.identity().plan().as_hex())
  append_startup_text(output, plan.numeric_binding().digest().as_hex())
  append_startup_text(output, device_numeric.digest().as_hex())
  append_startup_text(output, kernels.manifest().digest().as_hex())
  if output.length() > maximum {
    raise Fp8StartupRejected(CanonicalBytes)
  }
  Bytes::from_array(output)
}

///|
/// Admits the same authenticated model/device/artifact join as v1 plus exact
/// one- or two-stage activation-scale workspace semantics.
pub fn admit_v2(
  plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  kernels : @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  limits : Fp8StartupAdmissionLimits,
) -> Fp8StartupAdmissionV2 raise Fp8StartupAdmissionError {
  // Reuse v1's complete evidence validation, but never reuse its type or
  // canonical identity as v2 authority.
  let base = admit(plan, device_numeric, kernels, limits)
  validate_staged_plan_v2(plan, kernels)
  let canonical = encode_fp8_startup_admission_v2(
    plan,
    device_numeric,
    kernels,
    limits.max_canonical_bytes,
  )
  {
    version: Fp8StartupAdmissionV2Version::v2(),
    base,
    canonical,
    digest: Fp8StartupAdmissionV2Digest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
