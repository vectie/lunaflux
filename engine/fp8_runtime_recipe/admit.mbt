///|
fn require_workspace_region(
  region : @fp8_launch_abi.Fp8LaunchStorageRegion,
  expected_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  expected_bytes : Int64,
) -> Unit raise Fp8RuntimeRecipeError {
  if region.storage() != expected_storage {
    raise InvalidFp8RuntimeRecipe(WorkspaceStorage)
  }
  if region.storage_bytes() != expected_bytes {
    raise InvalidFp8RuntimeRecipe(WorkspaceCapacity)
  }
}

///|
fn validate_weight_regions(
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launch : @fp8_launch_abi.Fp8DynamicLaunchAbi,
  expected_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
) -> Unit raise Fp8RuntimeRecipeError {
  let bindings = launch.weight_scales()
  let regions = launch.scale_tensor_regions()
  if bindings.length() != regions.length() {
    raise InvalidFp8RuntimeRecipe(WeightRegion)
  }
  for index, binding in bindings {
    let bound = regions[index]
    if bound.tensor() != binding.scale() ||
      bound.storage_digest() != binding.scale_storage_digest() {
      raise InvalidFp8RuntimeRecipe(WeightRegion)
    }
    guard weights.layout().region(binding.scale().tensor()) is Some(layout) else {
      raise InvalidFp8RuntimeRecipe(WeightLayout)
    }
    let region = bound.region()
    if region.storage() != expected_storage ||
      region.storage_bytes() != weights.layout().arena_bytes() {
      raise InvalidFp8RuntimeRecipe(WeightStorage)
    }
    if region.alignment() != weights.layout().alignment_bytes() {
      raise InvalidFp8RuntimeRecipe(WeightRegion)
    }
    if region.offset() != layout.offset_bytes() ||
      region.byte_length() != layout.byte_length() {
      raise InvalidFp8RuntimeRecipe(WeightRegion)
    }
  }
}

///|
fn authenticate_weight_operand_claim(
  weights : @numeric_weight_file.NumericWeightFileInspection,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
  claim : Fp8LaunchWeightRegionClaim,
  expected_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
) -> Fp8LaunchWeightBinding raise Fp8RuntimeRecipeError {
  if claim.profile != entry.profile() || claim.operation != entry.operation() {
    raise InvalidFp8RuntimeRecipe(LaunchOrder)
  }
  let manifest_weights = entry.weights()
  if claim.regions.length() != manifest_weights.length() {
    raise InvalidFp8RuntimeRecipe(WeightRegion)
  }
  let authenticated : Array[Fp8WeightOperandStorageRegion] = []
  for index, binding in manifest_weights {
    guard weights.layout().region(binding.tensor()) is Some(layout) else {
      raise InvalidFp8RuntimeRecipe(WeightLayout)
    }
    let region = claim.regions[index]
    if region.storage() != expected_storage ||
      region.storage_bytes() != weights.layout().arena_bytes() {
      raise InvalidFp8RuntimeRecipe(WeightStorage)
    }
    if region.alignment() != weights.layout().alignment_bytes() {
      raise InvalidFp8RuntimeRecipe(WeightRegion)
    }
    if region.offset() != layout.offset_bytes() ||
      region.byte_length() != layout.byte_length() {
      raise InvalidFp8RuntimeRecipe(WeightRegion)
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
fn validate_launch(
  startup : @fp8_startup_admission.Fp8StartupAdmission,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launch : @fp8_launch_abi.Fp8DynamicLaunchAbi,
  expected_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity,
  workspace_bytes : Int64,
  entry : @numeric_capability_manifest.Fp8KernelManifestEntry,
) -> Unit raise Fp8RuntimeRecipeError {
  let manifest = startup.kernel_artifacts().manifest()
  if launch.plan_identity() != startup.model_plan().identity() ||
    launch.numeric_schema_digest() !=
    startup.model_plan().numeric_binding().digest() ||
    launch.manifest_digest() != manifest.digest() ||
    launch.target() != manifest.target() {
    raise InvalidFp8RuntimeRecipe(LaunchBinding)
  }
  if launch.profile() != entry.profile() ||
    launch.operation() != entry.operation() ||
    launch.execution_digest() != entry.execution_digest() ||
    launch.module_digest() != entry.module_digest() ||
    launch.entry_point() != entry.entry_point() {
    raise InvalidFp8RuntimeRecipe(LaunchOrder)
  }
  validate_weight_regions(weights, launch, expected_storage)
  require_workspace_region(
    launch.activation_scale_workspace(),
    workspace_storage,
    workspace_bytes,
  )
  match launch.kernel_workspace() {
    Some(region) =>
      require_workspace_region(region, workspace_storage, workspace_bytes)
    None => ()
  }
}

///|
/// Cross-authenticates the startup admission, inspected numeric artifact, and
/// complete ordered launch-ABI set. Logical weight storage is derived from the
/// pinned numeric artifact digest; callers cannot substitute another storage
/// identity. The workspace identity/capacity remains an exact requirement for
/// a future device owner and grants no allocation authority itself.
pub fn admit(
  startup : @fp8_startup_admission.Fp8StartupAdmission,
  weights : @numeric_weight_file.NumericWeightFileInspection,
  launches : ArrayView[@fp8_launch_abi.Fp8DynamicLaunchAbi],
  weight_regions : ArrayView[Fp8LaunchWeightRegionClaim],
  limits : Fp8RuntimeRecipeLimits,
) -> Fp8RuntimeRecipe raise Fp8RuntimeRecipeError {
  let plan = startup.model_plan()
  if weights.numeric_schema_digest() != plan.numeric_binding().digest() ||
    weights.layout().numeric_schema_digest() !=
    Some(plan.numeric_binding().digest()) {
    raise InvalidFp8RuntimeRecipe(WeightSchema)
  }
  if weights.identity() != plan.identity() {
    raise InvalidFp8RuntimeRecipe(WeightIdentity)
  }
  if weights.layout().identity() != plan.identity() ||
    weights.layout().arena_bytes() <= 0L {
    raise InvalidFp8RuntimeRecipe(WeightLayout)
  }
  let entries = startup.kernel_artifacts().manifest().entries()
  if launches.is_empty() ||
    launches.length() != entries.length() ||
    weight_regions.length() != launches.length() ||
    launches.length() > limits.max_launches {
    raise InvalidFp8RuntimeRecipe(LaunchCount)
  }
  let weight_storage = @fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
    weights.artifact_digest().as_hex(),
  ) catch {
    _ => raise InvalidFp8RuntimeRecipe(WeightStorage)
  }
  let first_workspace = launches[0].activation_scale_workspace()
  let workspace_storage = first_workspace.storage()
  let workspace_bytes = first_workspace.storage_bytes()
  if workspace_storage == weight_storage {
    raise InvalidFp8RuntimeRecipe(WorkspaceStorage)
  }
  if workspace_bytes <= 0L || workspace_bytes > limits.max_workspace_bytes {
    raise InvalidFp8RuntimeRecipe(WorkspaceCapacity)
  }
  let retained : Array[@fp8_launch_abi.Fp8DynamicLaunchAbi] = []
  let authenticated_weights : Array[Fp8LaunchWeightBinding] = []
  for index, launch in launches {
    validate_launch(
      startup,
      weights,
      launch,
      weight_storage,
      workspace_storage,
      workspace_bytes,
      entries[index],
    )
    authenticated_weights.push(
      authenticate_weight_operand_claim(
        weights,
        entries[index],
        weight_regions[index],
        weight_storage,
      ),
    )
    retained.push(launch)
  }
  let immutable = ReadOnlyArray::from_array(retained)
  let immutable_weight_bindings = ReadOnlyArray::from_array(
    authenticated_weights,
  )
  let canonical = encode_runtime_recipe(
    startup,
    weights,
    immutable,
    immutable_weight_bindings,
    weight_storage,
    workspace_storage,
    workspace_bytes,
    limits.max_canonical_bytes,
  )
  {
    startup,
    weights,
    launches: immutable,
    weight_bindings: immutable_weight_bindings,
    weight_storage,
    workspace_storage,
    workspace_bytes,
    canonical,
    digest: Fp8RuntimeRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
