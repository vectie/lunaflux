///|
fn v5_mentions_fp8(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  execution.activation_compute() ==
  @numeric_contract.ComputeDType::f8_e4m3_finite() ||
  execution.tensor_input() == @numeric_contract.ComputeDType::f8_e4m3_finite()
}

///|
fn v5_weight_region(
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  tensor : @plan.TensorRef,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion raise ExecutionManifestFileError {
  guard inspection.layout().region(tensor) is Some(region) else {
    raise AdmissionRejected(Fp8NumericLayout)
  }
  @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      inspection.artifact_digest().as_hex(),
    ),
    storage_bytes=inspection.layout().arena_bytes(),
    offset=region.offset_bytes(),
    byte_length=region.byte_length(),
    alignment=inspection.layout().alignment_bytes(),
  ) catch {
    _ => raise AdmissionRejected(Fp8NumericLayout)
  }
}

///|
fn reconstruct_fp8_recipes_v5(
  plan : @plan.ModelPlan,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  kv_layout : @device_layout.DeviceKvLayout,
  numeric : @device.DeviceNumericCapability,
  step_limits : @device_step.DeviceStepLimits,
  claims : ManifestV5Claims,
  resolved : @catalog.ResolvedKernelCatalog,
  contracts : @launch_contract.PagedV4LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  fp8_limits : @numeric_capability_manifest.Fp8KernelManifestLimits,
  startup_limits : @fp8_startup_admission.Fp8StartupAdmissionLimits,
  exact_limits : @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits,
  reusable_limits : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits,
) -> (
  @numeric_capability_manifest.AdmittedFp8KernelArtifacts,
  @fp8_startup_admission.Fp8StartupAdmissionV2,
  @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3,
) raise ExecutionManifestFileError {
  let manifest_claims : Array[
    @numeric_capability_manifest.Fp8KernelManifestClaim,
  ] = []
  for index, operation in plan.operations() {
    let execution = plan.numeric_binding().operations()[index].execution()
    if v5_mentions_fp8(execution) {
      manifest_claims.push(
        @numeric_capability_manifest.Fp8KernelManifestClaim::new(
          profile=claims.profile_id,
          operation=operation.id(),
          execution=execution.digest(),
          entry_point=v4_entry_point(
            {
              profile_id: claims.profile_id,
              modules: claims.modules,
              operations: claims.operations,
              numeric: {
                i8_admission_digest: "",
                weight_artifact_digest: claims.numeric.weight_artifact_digest,
              },
            },
            index,
          ),
        ),
      )
    }
  }
  let numeric_artifacts = @numeric_capability_manifest.admit_paged_v4_fp8_e4m3_w8a8(
    plan, numeric, resolved, contracts, artifacts, manifest_claims, fp8_limits,
  ) catch {
    _ => raise AdmissionRejected(Fp8KernelManifest)
  }
  if numeric_artifacts.manifest().digest().as_hex() !=
    claims.numeric.kernel_manifest_digest {
    raise IdentityMismatch(Fp8KernelManifestDigest)
  }
  let startup = @fp8_startup_admission.admit_v2(
    plan, numeric, numeric_artifacts, startup_limits,
  ) catch {
    _ => raise AdmissionRejected(Fp8StartupV2)
  }
  let profile = contracts.profiles()[0]
  let shape = @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
    profile=claims.profile_id,
    rows=[
      @fp8_launch_abi.Fp8PagedRowShape::new(
        query_tokens=step_limits.max_tokens(),
        context_tokens=0,
        page_table_entries=step_limits.max_pages(),
      ),
    ],
  ) catch {
    _ => raise AdmissionRejected(LaunchContracts)
  }
  let workspace_storage = @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    claims.numeric.workspace_storage_digest,
  ) catch {
    _ => raise invalid_field(NumericAuthorization, InvalidSha256)
  }
  let mut workspace_capacity = 0L
  for entry in numeric_artifacts.manifest().entries() {
    let bytes = plan.operations()[entry.operation().as_int()].workspace_bytes()
    workspace_capacity = workspace_capacity.max(bytes)
  }
  let launches : Array[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi] = []
  let regions : Array[@fp8_runtime_recipe.Fp8LaunchWeightRegionClaimV2] = []
  for entry in numeric_artifacts.manifest().entries() {
    let scale_regions : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    let weight_regions : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    for binding in entry.weights() {
      weight_regions.push(v5_weight_region(inspection, binding.tensor()))
      scale_regions.push(v5_weight_region(inspection, binding.scale().tensor()))
    }
    let operation = plan.operations()[entry.operation().as_int()]
    let workspace = @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage=workspace_storage,
      storage_bytes=workspace_capacity,
      offset=0L,
      byte_length=operation.workspace_bytes(),
      alignment=4L,
    ) catch {
      _ => raise AdmissionRejected(LaunchContracts)
    }
    launches.push(
      @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
        plan,
        numeric_artifacts,
        PagedV4(contracts),
        operation=entry.operation(),
        shape~,
        weight_scale_regions=scale_regions,
        workspace~,
      ) catch {
        _ => raise AdmissionRejected(LaunchContracts)
      },
    )
    regions.push(
      @fp8_runtime_recipe.Fp8LaunchWeightRegionClaimV2::new(
        profile=claims.profile_id,
        operation=entry.operation(),
        regions=weight_regions,
      ) catch {
        _ => raise AdmissionRejected(Fp8NumericLayout)
      },
    )
  }
  let exact = @fp8_runtime_recipe.admit_v2(
    startup, inspection, launches, regions, exact_limits,
  ) catch {
    _ => raise AdmissionRejected(Fp8RuntimeRecipeV2)
  }
  if exact.digest().as_hex() != claims.numeric.runtime_recipe_v2_digest {
    raise IdentityMismatch(Fp8RuntimeRecipeDigest)
  }
  let reusable = @fp8_runtime_recipe.admit_reusable_paged_v3(
    exact, profile, kv_layout, reusable_limits,
  ) catch {
    _ => raise AdmissionRejected(Fp8ReusableRecipeV3)
  }
  if reusable.digest().as_hex() != claims.numeric.reusable_recipe_v3_digest {
    raise IdentityMismatch(Fp8ReusableRecipeDigest)
  }
  (numeric_artifacts, startup, exact, reusable)
}
