///|
fn fp8_v5_target_supported(target : @catalog.DeviceTarget) -> Bool {
  (target.compute_major() == 8 && target.compute_minor() == 9) ||
  (target.compute_major() == 9 && target.compute_minor() == 0) ||
  (target.compute_major() == 12 && target.compute_minor() == 0)
}

///|
/// Reconstructs a schema-v5 FP8 PagedV4 release recipe from one no-follow,
/// digest-pinned manifest. This remains inert until a separate external v3
/// release approval is joined and the exact module files are re-inspected.
pub fn load_paged_fp8_v5(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : ExecutionManifestDigest,
  plan : @plan.ModelPlan,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  kv_layout : @device_layout.DeviceKvLayout,
  numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  step_limits : @device_step.DeviceStepLimits,
  limits : ExecutionManifestFileLimits,
  fp8_limits : @numeric_capability_manifest.Fp8KernelManifestLimits,
  startup_limits : @fp8_startup_admission.Fp8StartupAdmissionLimits,
  exact_limits : @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits,
  reusable_limits : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
  release_limits : @fp8_release_authority.Fp8ReleaseAuthorityLimits,
  verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
  parent_witness? : @luna_capability_manifest.LunaParentStartupApprovalWitness? = None,
) -> Fp8PagedExecutionAdmissionV5 raise ExecutionManifestFileError {
  if inspection.identity() != plan.identity() ||
    inspection.layout().identity() != plan.identity() ||
    inspection.numeric_schema_digest() != plan.numeric_binding().digest() ||
    kv_layout.identity() != plan.identity() ||
    numeric.feature() != FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1 ||
    target.compute_major() != numeric.compute_major() ||
    target.compute_minor() != numeric.compute_minor() ||
    !fp8_v5_target_supported(target) {
    raise UnsupportedModelEnvelope
  }
  let bytes = read_manifest_snapshot(root, locator, expected_digest, limits)
  let claims = parse_manifest_v5(bytes, plan, numeric, target, limits)
  if claims.numeric.weight_artifact_digest != inspection.artifact_digest() {
    raise IdentityMismatch(WeightArtifactDigest)
  }
  let v4 : ManifestV4Claims = {
    profile_id: claims.profile_id,
    modules: claims.modules,
    operations: claims.operations,
    numeric: {
      i8_admission_digest: "",
      weight_artifact_digest: claims.numeric.weight_artifact_digest,
    },
  }
  let resolved = derive_catalog_v4(v4, plan, kv_layout, target)
  let static_plan = @device_plan.build(plan, inspection.layout(), resolved) catch {
    _ => raise AdmissionRejected(StaticPlan)
  }
  let memory_plan = @device_memory.plan(static_plan, limits.device_memory()) catch {
    _ => raise AdmissionRejected(MemoryPlan)
  }
  let contracts = derive_contracts_v4(
    v4,
    plan,
    static_plan,
    memory_plan,
    kv_layout,
    resolved,
    step_limits,
    limits.launch_contract(),
  )
  let artifacts = @artifact_file.load_paged_v4_admitted(
    root,
    derive_v4_artifact_source(v4, limits.artifact_file()),
    contracts,
    limits.artifact_file(),
  ) catch {
    _ => raise AdmissionRejected(ArtifactFiles)
  }
  let (numeric_artifacts, startup, exact, reusable) = reconstruct_fp8_recipes_v5(
    plan, inspection, kv_layout, numeric, step_limits, claims, resolved, contracts,
    artifacts, fp8_limits, startup_limits, exact_limits, reusable_limits,
  )
  let release_authority = admit_fp8_release_v5(
    root, claims, reusable, artifacts, approval, verifier, parent_witness, release_limits,
  )
  {
    model_plan: plan,
    weight_inspection: inspection,
    kv_layout,
    device_step_limits: step_limits,
    static_plan,
    memory_plan,
    profile: contracts.profiles()[0],
    contracts,
    artifacts,
    numeric_artifacts,
    startup,
    exact_recipe: exact,
    reusable_recipe: reusable,
    release_authority,
  }
}
