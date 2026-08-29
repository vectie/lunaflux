///|
fn authenticate_weight_claim_v2(
  weights : @numeric_weight_file.NumericWeightFileInspection,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  claim : Fp8LaunchWeightRegionClaimV2,
  storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
) -> Fp8LaunchWeightBindingV2 raise Fp8RuntimeRecipeV2Error {
  if claim.profile != entry.profile() ||
    claim.operation != entry.operation() ||
    claim.regions.length() != entry.weights().length() {
    raise InvalidFp8RuntimeRecipeV2(LaunchOrder)
  }
  let authenticated : Array[Fp8WeightOperandStorageRegion] = []
  for index, binding in entry.weights() {
    guard weights.layout().region(binding.tensor()) is Some(layout) else {
      raise InvalidFp8RuntimeRecipeV2(WeightLayout)
    }
    let region = claim.regions[index]
    if region.storage() != storage ||
      region.storage_bytes() != weights.layout().arena_bytes() {
      raise InvalidFp8RuntimeRecipeV2(WeightStorage)
    }
    if region.offset() != layout.offset_bytes() ||
      region.byte_length() != layout.byte_length() ||
      region.alignment() != weights.layout().alignment_bytes() {
      raise InvalidFp8RuntimeRecipeV2(WeightRegion)
    }
    authenticated.push({
      tensor: binding.tensor(),
      storage_digest: binding.storage_digest(),
      region,
    })
  }
  {
    profile: entry.profile(),
    operation: entry.operation(),
    regions: ReadOnlyArray::from_array(authenticated),
  }
}

///|
fn validate_v2_scale_regions(
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
  storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  let bindings = launch.weight_scales()
  let regions = launch.scale_tensor_regions()
  if bindings.length() != regions.length() {
    raise InvalidFp8RuntimeRecipeV2(WeightRegion)
  }
  for index, binding in bindings {
    let bound = regions[index]
    guard weights.layout().region(binding.scale().tensor()) is Some(layout) else {
      raise InvalidFp8RuntimeRecipeV2(WeightLayout)
    }
    let region = bound.region()
    if bound.tensor() != binding.scale() ||
      bound.storage_digest() != binding.scale_storage_digest() ||
      region.storage() != storage ||
      region.storage_bytes() != weights.layout().arena_bytes() {
      raise InvalidFp8RuntimeRecipeV2(WeightStorage)
    }
    if region.offset() != layout.offset_bytes() ||
      region.byte_length() != layout.byte_length() ||
      region.alignment() != weights.layout().alignment_bytes() {
      raise InvalidFp8RuntimeRecipeV2(WeightRegion)
    }
  }
}

///|
fn validate_v2_stages(
  plan : @plan.ModelPlan,
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  let operation_index = launch.operation().as_int()
  guard plan.operations().get(operation_index) is Some(operation) &&
    plan.numeric_binding().operations().get(operation_index) is Some(numeric) else {
    raise InvalidFp8RuntimeRecipeV2(StageBinding)
  }
  let policy = numeric.execution().activation_scale()
  let stages = launch.stages()
  let workspace = launch.workspace()
  if launch.activation_scale_policy() != policy ||
    stages.length() != policy.stage_count() ||
    workspace.byte_length() != stages.length().to_int64() * 4L ||
    workspace.byte_length() != operation.workspace_bytes() ||
    workspace.alignment() != 4L {
    raise InvalidFp8RuntimeRecipeV2(StageBinding)
  }
  for ordinal, stage in stages {
    if stage.ordinal() != ordinal ||
      policy.stage_at(ordinal) != Some(stage.stage()) {
      raise InvalidFp8RuntimeRecipeV2(StageBinding)
    }
    let region = stage.region()
    if region.storage() != workspace.storage() ||
      region.storage_bytes() != workspace.storage_bytes() ||
      region.offset() != workspace.offset() + ordinal.to_int64() * 4L ||
      region.byte_length() != 4L ||
      region.alignment() != 4L {
      raise InvalidFp8RuntimeRecipeV2(StageBinding)
    }
  }
}

///|
fn validate_v2_launch_identity(
  startup : @fp8_startup_admission.Fp8StartupAdmissionV2,
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  let plan = startup.model_plan()
  let manifest = startup.kernel_artifacts().manifest()
  if launch.version() != @fp8_launch_abi.Fp8LaunchAbiVersion::v2() {
    raise InvalidFp8RuntimeRecipeV2(LaunchVersion)
  }
  if launch.plan_identity() != plan.identity() ||
    launch.numeric_schema_digest() != plan.numeric_binding().digest() ||
    launch.manifest_digest() != manifest.digest() ||
    launch.target() != manifest.target() {
    raise InvalidFp8RuntimeRecipeV2(LaunchBinding)
  }
  if launch.profile() != entry.profile() ||
    launch.operation() != entry.operation() ||
    launch.execution_digest() != entry.execution_digest() ||
    launch.module_digest() != entry.module_digest() ||
    launch.entry_point() != entry.entry_point() {
    raise InvalidFp8RuntimeRecipeV2(LaunchOrder)
  }
  guard plan.operations().get(entry.operation().as_int()) is Some(operation) &&
    launch.operation_kind() == operation.kind() else {
    raise InvalidFp8RuntimeRecipeV2(LaunchBinding)
  }
}

///|
fn validate_v2_raw_launch_authority(
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  if launch.launch_source_version() != PagedNumericV4 {
    raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
  }
  let operands = launch.raw_operands()
  let workspace = launch.workspace()
  if operands.is_empty() {
    raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
  }
  let mut weight_index = 0
  let mut pending_scale = false
  let mut workspace_count = 0
  for index, operand in operands {
    match operand.role() {
      WeightInput(weight) => {
        if pending_scale ||
          weight_index >= entry.weights().length() ||
          entry.weights()[weight_index].tensor() != weight {
          raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
        }
        pending_scale = true
      }
      WeightScaleInput(weight, scale) => {
        if !pending_scale ||
          weight_index >= entry.weights().length() ||
          entry.weights()[weight_index].tensor() != weight ||
          entry.weights()[weight_index].scale() != scale {
          raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
        }
        pending_scale = false
        weight_index += 1
      }
      Workspace => {
        if pending_scale ||
          index != operands.length() - 1 ||
          operand.byte_count() != workspace.byte_length() ||
          operand.alignment() != workspace.alignment() {
          raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
        }
        workspace_count += 1
      }
      _ => if pending_scale { raise InvalidFp8RuntimeRecipeV2(LaunchAuthority) }
    }
  }
  if pending_scale ||
    weight_index != entry.weights().length() ||
    workspace_count != 1 {
    raise InvalidFp8RuntimeRecipeV2(LaunchAuthority)
  }
}

///|
fn validate_ordered_workspace_reuse_v2(
  workspace : @fp8_launch_abi.Fp8LaunchStorageRegion,
  expected_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  expected_capacity : Int64,
) -> Unit raise Fp8RuntimeRecipeV2Error {
  if workspace.storage() != expected_storage {
    raise InvalidFp8RuntimeRecipeV2(WorkspaceStorage)
  }
  if workspace.storage_bytes() != expected_capacity {
    raise InvalidFp8RuntimeRecipeV2(WorkspaceCapacity)
  }
}

///|
pub fn admit_v2(
  startup : @fp8_startup_admission.Fp8StartupAdmissionV2,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launches : ArrayView[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi],
  weight_regions : ArrayView[Fp8LaunchWeightRegionClaimV2],
  limits : Fp8RuntimeRecipeV2Limits,
) -> Fp8RuntimeRecipeV2 raise Fp8RuntimeRecipeV2Error {
  if startup.version() !=
    @fp8_startup_admission.Fp8StartupAdmissionV2Version::v2() {
    raise InvalidFp8RuntimeRecipeV2(StartupVersion)
  }
  let plan = startup.model_plan()
  if weights.numeric_schema_digest() != plan.numeric_binding().digest() ||
    weights.layout().numeric_schema_digest() !=
    Some(plan.numeric_binding().digest()) {
    raise InvalidFp8RuntimeRecipeV2(WeightSchema)
  }
  if weights.identity() != plan.identity() {
    raise InvalidFp8RuntimeRecipeV2(WeightIdentity)
  }
  if weights.layout().identity() != plan.identity() ||
    weights.layout().arena_bytes() <= 0L {
    raise InvalidFp8RuntimeRecipeV2(WeightLayout)
  }
  let entries = startup.kernel_artifacts().manifest().entries()
  if launches.is_empty() ||
    launches.length() != entries.length() ||
    launches.length() != weight_regions.length() ||
    launches.length() > limits.max_launches {
    raise InvalidFp8RuntimeRecipeV2(LaunchCount)
  }
  let weight_storage = @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    weights.artifact_digest().as_hex(),
  ) catch {
    _ => raise InvalidFp8RuntimeRecipeV2(WeightStorage)
  }
  let first_workspace = launches[0].workspace()
  let workspace_storage = first_workspace.storage()
  let workspace_capacity_bytes = first_workspace.storage_bytes()
  if workspace_storage == weight_storage {
    raise InvalidFp8RuntimeRecipeV2(WorkspaceStorage)
  }
  if workspace_capacity_bytes <= 0L ||
    workspace_capacity_bytes > limits.max_workspace_arena_bytes {
    raise InvalidFp8RuntimeRecipeV2(WorkspaceCapacity)
  }
  let retained_launches : Array[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi] = []
  let bindings : Array[Fp8LaunchWeightBindingV2] = []
  let workspaces : Array[Fp8WorkspaceBindingV2] = []
  for index, launch in launches {
    let entry = entries[index]
    validate_v2_launch_identity(startup, launch, entry)
    validate_v2_raw_launch_authority(launch, entry)
    validate_v2_scale_regions(weights, launch, weight_storage)
    validate_v2_stages(plan, launch)
    validate_ordered_workspace_reuse_v2(
      launch.workspace(),
      workspace_storage,
      workspace_capacity_bytes,
    )
    bindings.push(
      authenticate_weight_claim_v2(
        weights,
        entry,
        weight_regions[index],
        weight_storage,
      ),
    )
    workspaces.push({
      profile: entry.profile(),
      operation: entry.operation(),
      workspace: launch.workspace(),
    })
    retained_launches.push(launch)
  }
  let immutable_launches = ReadOnlyArray::from_array(retained_launches)
  let immutable_bindings = ReadOnlyArray::from_array(bindings)
  let immutable_workspaces = ReadOnlyArray::from_array(workspaces)
  let canonical = encode_runtime_recipe_v2(
    startup,
    weights,
    immutable_launches,
    immutable_bindings,
    immutable_workspaces,
    weight_storage,
    workspace_storage,
    workspace_capacity_bytes,
    limits.max_canonical_bytes,
  )
  {
    startup,
    weights,
    launches: immutable_launches,
    weight_bindings: immutable_bindings,
    workspaces: immutable_workspaces,
    weight_storage,
    workspace_storage,
    workspace_capacity_bytes,
    canonical,
    digest: Fp8RuntimeRecipeV2Digest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
