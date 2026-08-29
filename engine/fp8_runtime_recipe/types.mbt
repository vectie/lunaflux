///|
pub(all) enum Fp8RuntimeRecipeRule {
  Limit
  WeightIdentity
  WeightSchema
  WeightLayout
  LaunchCount
  LaunchOrder
  LaunchBinding
  WeightStorage
  WeightRegion
  WorkspaceStorage
  WorkspaceCapacity
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8RuntimeRecipeError {
  InvalidFp8RuntimeRecipe(Fp8RuntimeRecipeRule)
} derive(Debug, Eq)

///|
pub struct Fp8RuntimeRecipeLimits {
  priv max_launches : Int
  priv max_workspace_bytes : Int64
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn Fp8RuntimeRecipeLimits::new(
  max_launches~ : Int,
  max_workspace_bytes~ : Int64,
  max_canonical_bytes~ : Int,
) -> Fp8RuntimeRecipeLimits raise Fp8RuntimeRecipeError {
  if max_launches <= 0 ||
    max_launches > 65536 ||
    max_workspace_bytes <= 0L ||
    max_workspace_bytes > 1099511627776L ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > 16777216 {
    raise InvalidFp8RuntimeRecipe(Limit)
  }
  { max_launches, max_workspace_bytes, max_canonical_bytes }
}

///|
pub struct Fp8RuntimeRecipeDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8RuntimeRecipeDigest::as_hex(self : Fp8RuntimeRecipeDigest) -> String {
  self.0
}

///|
/// Untrusted ordered region claim for the FP8 weight operands of one launch.
/// `admit` derives tensor/storage identity from the manifest and accepts this
/// claim only when every region exactly matches the inspected device layout.
pub struct Fp8LaunchWeightRegionClaim {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv regions : ReadOnlyArray[@fp8_launch_abi.Fp8LaunchStorageRegion]
}

///|
pub fn Fp8LaunchWeightRegionClaim::new(
  profile~ : @launch_contract.KernelProfileId,
  operation~ : @plan.OperationId,
  regions~ : ArrayView[@fp8_launch_abi.Fp8LaunchStorageRegion],
) -> Fp8LaunchWeightRegionClaim raise Fp8RuntimeRecipeError {
  if regions.length() != 1 && regions.length() != 3 {
    raise InvalidFp8RuntimeRecipe(WeightRegion)
  }
  let retained : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
  for region in regions {
    retained.push(region)
  }
  { profile, operation, regions: ReadOnlyArray::from_array(retained) }
}

///|
pub struct Fp8WeightOperandStorageRegion {
  priv tensor : @plan.TensorRef
  priv storage_digest : @numeric_contract.TensorStorageDigest
  priv region : @fp8_launch_abi.Fp8LaunchStorageRegion
} derive(Debug, Eq)

///|
pub fn Fp8WeightOperandStorageRegion::tensor(
  self : Fp8WeightOperandStorageRegion,
) -> @plan.TensorRef {
  self.tensor
}

///|
pub fn Fp8WeightOperandStorageRegion::storage_digest(
  self : Fp8WeightOperandStorageRegion,
) -> @numeric_contract.TensorStorageDigest {
  self.storage_digest
}

///|
pub fn Fp8WeightOperandStorageRegion::region(
  self : Fp8WeightOperandStorageRegion,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  self.region
}

///|
pub struct Fp8LaunchWeightBinding {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv regions : ReadOnlyArray[Fp8WeightOperandStorageRegion]
}

///|
pub fn Fp8LaunchWeightBinding::profile(
  self : Fp8LaunchWeightBinding,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8LaunchWeightBinding::operation(
  self : Fp8LaunchWeightBinding,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8LaunchWeightBinding::regions(
  self : Fp8LaunchWeightBinding,
) -> ReadOnlyArray[Fp8WeightOperandStorageRegion] {
  self.regions
}

///|
/// Inert exact composition evidence. This owns no context, allocation, module,
/// function, stream, executor, worker, route, or readiness authority.
pub struct Fp8RuntimeRecipe {
  priv startup : @fp8_startup_admission.Fp8StartupAdmission
  priv weights : @numeric_weight_file.NumericWeightFileInspection
  priv launches : ReadOnlyArray[@fp8_launch_abi.Fp8DynamicLaunchAbi]
  priv weight_bindings : ReadOnlyArray[Fp8LaunchWeightBinding]
  priv weight_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity
  priv workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity
  priv workspace_bytes : Int64
  priv canonical : Bytes
  priv digest : Fp8RuntimeRecipeDigest
}

///|
pub fn Fp8RuntimeRecipe::startup(
  self : Fp8RuntimeRecipe,
) -> @fp8_startup_admission.Fp8StartupAdmission {
  self.startup
}

///|
pub fn Fp8RuntimeRecipe::weight_inspection(
  self : Fp8RuntimeRecipe,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.weights
}

///|
pub fn Fp8RuntimeRecipe::launches(
  self : Fp8RuntimeRecipe,
) -> ReadOnlyArray[@fp8_launch_abi.Fp8DynamicLaunchAbi] {
  self.launches
}

///|
pub fn Fp8RuntimeRecipe::weight_bindings(
  self : Fp8RuntimeRecipe,
) -> ReadOnlyArray[Fp8LaunchWeightBinding] {
  self.weight_bindings
}

///|
pub fn Fp8RuntimeRecipe::weight_storage(
  self : Fp8RuntimeRecipe,
) -> @fp8_launch_abi.Fp8LaunchStorageIdentity {
  self.weight_storage
}

///|
pub fn Fp8RuntimeRecipe::workspace_storage(
  self : Fp8RuntimeRecipe,
) -> @fp8_launch_abi.Fp8LaunchStorageIdentity {
  self.workspace_storage
}

///|
pub fn Fp8RuntimeRecipe::workspace_bytes(self : Fp8RuntimeRecipe) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Fp8RuntimeRecipe::canonical_bytes(self : Fp8RuntimeRecipe) -> Bytes {
  self.canonical
}

///|
pub fn Fp8RuntimeRecipe::digest(
  self : Fp8RuntimeRecipe,
) -> Fp8RuntimeRecipeDigest {
  self.digest
}
