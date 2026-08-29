///|
fn load_i8_execution_v4(
  kernel_root : @approved_fs.ApprovedRoot,
  claims : I8RuntimeDescriptorClaims,
  plan : @plan.ModelPlan,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  kv_layout : @device_layout.DeviceKvLayout,
  numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  step_limits : @device_step.DeviceStepLimits,
  execution_limits : @execution_manifest_file.ExecutionManifestFileLimits,
  bootstrap_limits : @device_step.DeviceBootstrapLimits,
) -> @execution_manifest_file.I8PagedExecutionAdmissionV4 raise RuntimeDescriptorFileError {
  if claims.kernels.policy != I8_RUNTIME_POLICY {
    raise AdmissionRejected(I8ProductionPolicy)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    claims.kernels.manifest_locator,
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  let digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    claims.kernels.manifest_digest,
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  let execution = @execution_manifest_file.load_paged_i8_v4(
    kernel_root,
    locator,
    digest,
    plan,
    weights.layout(),
    kv_layout,
    numeric,
    target,
    step_limits,
    device_ordinal=claims.execution.device_ordinal,
    execution_limits,
    admit_i8_policy_limits(plan),
    bootstrap_limits,
  ) catch {
    _ => raise AdmissionRejected(ExecutionManifest)
  }
  if execution.weight_artifact_digest() != weights.artifact_digest() {
    raise AdmissionRejected(WeightArtifactDigest)
  }
  execution
}

///|
fn require_i8_bootstrap_v3_digest(
  claims : I8RuntimeDescriptorKernelClaims,
  execution : @execution_manifest_file.I8PagedExecutionAdmissionV4,
) -> Unit raise RuntimeDescriptorFileError {
  let expected = @worker_wire.WorkerBootstrapDigest::from_sha256(
    claims.bootstrap_v3_digest,
  ) catch {
    _ => raise AdmissionRejected(BootstrapDigest)
  }
  if execution.bootstrap_manifest().digest() != expected {
    raise AdmissionRejected(BootstrapDigest)
  }
}
