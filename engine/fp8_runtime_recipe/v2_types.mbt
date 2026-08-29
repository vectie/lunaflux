///|
pub(all) enum Fp8RuntimeRecipeV2Rule {
  Limit
  StartupVersion
  WeightIdentity
  WeightSchema
  WeightLayout
  LaunchVersion
  LaunchCount
  LaunchOrder
  LaunchBinding
  LaunchAuthority
  WeightStorage
  WeightRegion
  WorkspaceStorage
  WorkspaceCapacity
  StageBinding
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8RuntimeRecipeV2Error {
  InvalidFp8RuntimeRecipeV2(Fp8RuntimeRecipeV2Rule)
} derive(Debug, Eq)

///|
pub struct Fp8RuntimeRecipeV2Limits {
  priv max_launches : Int
  priv max_workspace_arena_bytes : Int64
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn Fp8RuntimeRecipeV2Limits::new(
  max_launches~ : Int,
  max_workspace_arena_bytes~ : Int64,
  max_canonical_bytes~ : Int,
) -> Fp8RuntimeRecipeV2Limits raise Fp8RuntimeRecipeV2Error {
  if max_launches <= 0 ||
    max_launches > 65536 ||
    max_workspace_arena_bytes <= 0L ||
    max_workspace_arena_bytes > 1099511627776L ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > 16777216 {
    raise InvalidFp8RuntimeRecipeV2(Limit)
  }
  { max_launches, max_workspace_arena_bytes, max_canonical_bytes }
}

///|
pub struct Fp8RuntimeRecipeV2Digest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8RuntimeRecipeV2Digest::as_hex(
  self : Fp8RuntimeRecipeV2Digest,
) -> String {
  self.0
}

///|
pub struct Fp8LaunchWeightRegionClaimV2 {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv regions : ReadOnlyArray[@fp8_launch_abi.Fp8LaunchStorageRegion]
}

///|
pub fn Fp8LaunchWeightRegionClaimV2::new(
  profile~ : @launch_contract.KernelProfileId,
  operation~ : @plan.OperationId,
  regions~ : ArrayView[@fp8_launch_abi.Fp8LaunchStorageRegion],
) -> Fp8LaunchWeightRegionClaimV2 raise Fp8RuntimeRecipeV2Error {
  if regions.length() != 1 && regions.length() != 3 {
    raise InvalidFp8RuntimeRecipeV2(WeightRegion)
  }
  let retained : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
  for region in regions {
    retained.push(region)
  }
  { profile, operation, regions: ReadOnlyArray::from_array(retained) }
}

///|
pub struct Fp8LaunchWeightBindingV2 {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv regions : ReadOnlyArray[Fp8WeightOperandStorageRegion]
}

///|
pub fn Fp8LaunchWeightBindingV2::profile(
  self : Fp8LaunchWeightBindingV2,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8LaunchWeightBindingV2::operation(
  self : Fp8LaunchWeightBindingV2,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8LaunchWeightBindingV2::regions(
  self : Fp8LaunchWeightBindingV2,
) -> ReadOnlyArray[Fp8WeightOperandStorageRegion] {
  self.regions
}

///|
pub struct Fp8WorkspaceBindingV2 {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv workspace : @fp8_launch_abi.Fp8LaunchStorageRegion
}

///|
pub fn Fp8WorkspaceBindingV2::profile(
  self : Fp8WorkspaceBindingV2,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8WorkspaceBindingV2::operation(
  self : Fp8WorkspaceBindingV2,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8WorkspaceBindingV2::workspace(
  self : Fp8WorkspaceBindingV2,
) -> @fp8_launch_abi.Fp8LaunchStorageRegion {
  self.workspace
}

///|
/// Separate inert staged-runtime domain. It cannot contain v1 startup or v1
/// launches and owns no allocation, module, executor, route, or readiness.
pub struct Fp8RuntimeRecipeV2 {
  priv startup : @fp8_startup_admission.Fp8StartupAdmissionV2
  priv weights : @numeric_weight_file.NumericWeightFileInspection
  priv launches : ReadOnlyArray[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi]
  priv weight_bindings : ReadOnlyArray[Fp8LaunchWeightBindingV2]
  priv workspaces : ReadOnlyArray[Fp8WorkspaceBindingV2]
  priv weight_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity
  priv workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity
  priv workspace_capacity_bytes : Int64
  priv canonical : Bytes
  priv digest : Fp8RuntimeRecipeV2Digest
}

///|
pub fn Fp8RuntimeRecipeV2::startup(
  self : Fp8RuntimeRecipeV2,
) -> @fp8_startup_admission.Fp8StartupAdmissionV2 {
  self.startup
}

///|
pub fn Fp8RuntimeRecipeV2::weight_inspection(
  self : Fp8RuntimeRecipeV2,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.weights
}

///|
pub fn Fp8RuntimeRecipeV2::launches(
  self : Fp8RuntimeRecipeV2,
) -> ReadOnlyArray[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi] {
  self.launches
}

///|
pub fn Fp8RuntimeRecipeV2::weight_bindings(
  self : Fp8RuntimeRecipeV2,
) -> ReadOnlyArray[Fp8LaunchWeightBindingV2] {
  self.weight_bindings
}

///|
pub fn Fp8RuntimeRecipeV2::workspaces(
  self : Fp8RuntimeRecipeV2,
) -> ReadOnlyArray[Fp8WorkspaceBindingV2] {
  self.workspaces
}

///|
pub fn Fp8RuntimeRecipeV2::weight_storage(
  self : Fp8RuntimeRecipeV2,
) -> @fp8_launch_abi.Fp8LaunchStorageIdentity {
  self.weight_storage
}

///|
pub fn Fp8RuntimeRecipeV2::workspace_storage(
  self : Fp8RuntimeRecipeV2,
) -> @fp8_launch_abi.Fp8LaunchStorageIdentity {
  self.workspace_storage
}

///|
pub fn Fp8RuntimeRecipeV2::workspace_capacity_bytes(
  self : Fp8RuntimeRecipeV2,
) -> Int64 {
  self.workspace_capacity_bytes
}

///|
pub fn Fp8RuntimeRecipeV2::canonical_bytes(self : Fp8RuntimeRecipeV2) -> Bytes {
  self.canonical
}

///|
pub fn Fp8RuntimeRecipeV2::digest(
  self : Fp8RuntimeRecipeV2,
) -> Fp8RuntimeRecipeV2Digest {
  self.digest
}
