///|
pub(all) enum Fp8ReusablePagedRecipeV3Rule {
  Limit
  Profile
  ModelEnvelope
  KvLayout
  LaunchOrder
  LaunchAuthority
  Workspace
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8ReusablePagedRecipeV3Error {
  InvalidFp8ReusablePagedRecipeV3(Fp8ReusablePagedRecipeV3Rule)
} derive(Debug, Eq)

///|
pub struct Fp8ReusablePagedRecipeV3Limits {
  priv max_launches : Int
  priv max_workspace_arena_bytes : Int64
  priv max_canonical_bytes : Int
}

///|
pub fn Fp8ReusablePagedRecipeV3Limits::new(
  max_launches~ : Int,
  max_workspace_arena_bytes~ : Int64,
  max_canonical_bytes~ : Int,
) -> Fp8ReusablePagedRecipeV3Limits raise Fp8ReusablePagedRecipeV3Error {
  if max_launches <= 0 ||
    max_launches > 65536 ||
    max_workspace_arena_bytes <= 0L ||
    max_workspace_arena_bytes > 1099511627776L ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > 16777216 {
    raise InvalidFp8ReusablePagedRecipeV3(Limit)
  }
  { max_launches, max_workspace_arena_bytes, max_canonical_bytes }
}

///|
pub struct Fp8ReusablePagedRecipeV3Digest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8ReusablePagedRecipeV3Digest::as_hex(
  self : Fp8ReusablePagedRecipeV3Digest,
) -> String {
  self.0
}

///|
/// Reusable launch evidence deliberately omits the v2 concrete row shape.
/// Its raw PagedV4 ABI remains profile-bounded and every live frame needs a
/// separate device-step admission token before enqueue.
pub struct Fp8ReusablePagedLaunchV3 {
  priv source : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi
  priv workspace_offset : Int64
}

///|
pub fn Fp8ReusablePagedLaunchV3::operation(
  self : Fp8ReusablePagedLaunchV3,
) -> @plan.OperationId {
  self.source.operation()
}

///|
pub fn Fp8ReusablePagedLaunchV3::operation_kind(
  self : Fp8ReusablePagedLaunchV3,
) -> @plan.OperationKind {
  self.source.operation_kind()
}

///|
pub fn Fp8ReusablePagedLaunchV3::execution_digest(
  self : Fp8ReusablePagedLaunchV3,
) -> @numeric_contract.OperationExecutionDigest {
  self.source.execution_digest()
}

///|
pub fn Fp8ReusablePagedLaunchV3::activation_scale_policy(
  self : Fp8ReusablePagedLaunchV3,
) -> @numeric_contract.ActivationScalePolicy {
  self.source.activation_scale_policy()
}

///|
pub fn Fp8ReusablePagedLaunchV3::module_digest(
  self : Fp8ReusablePagedLaunchV3,
) -> @catalog.AotArtifactDigest {
  self.source.module_digest()
}

///|
pub fn Fp8ReusablePagedLaunchV3::entry_point(
  self : Fp8ReusablePagedLaunchV3,
) -> @catalog.AotKernelEntryPoint {
  self.source.entry_point()
}

///|
pub fn Fp8ReusablePagedLaunchV3::dimensions(
  self : Fp8ReusablePagedLaunchV3,
) -> @launch_contract.AotLaunchDimensions {
  self.source.launch_dimensions()
}

///|
pub fn Fp8ReusablePagedLaunchV3::operands(
  self : Fp8ReusablePagedLaunchV3,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.source.raw_operands()
}

///|
pub fn Fp8ReusablePagedLaunchV3::workspace_offset(
  self : Fp8ReusablePagedLaunchV3,
) -> Int64 {
  self.workspace_offset
}

///|
pub fn Fp8ReusablePagedLaunchV3::workspace_bytes(
  self : Fp8ReusablePagedLaunchV3,
) -> Int64 {
  self.source.workspace().byte_length()
}

///|
pub fn Fp8ReusablePagedLaunchV3::workspace_alignment(
  self : Fp8ReusablePagedLaunchV3,
) -> Int64 {
  self.source.workspace().alignment()
}

///|
pub fn Fp8ReusablePagedLaunchV3::stage_count(
  self : Fp8ReusablePagedLaunchV3,
) -> Int {
  self.source.stages().length()
}

///|
pub fn Fp8ReusablePagedLaunchV3::stage_at(
  self : Fp8ReusablePagedLaunchV3,
  ordinal : Int,
) -> @numeric_contract.ActivationScaleStage? {
  self.source.stages().get(ordinal).map(stage => stage.stage())
}

///|
pub fn Fp8ReusablePagedLaunchV3::scale_tensor_regions(
  self : Fp8ReusablePagedLaunchV3,
) -> ReadOnlyArray[@fp8_launch_abi.Fp8ScaleTensorStorageRegion] {
  self.source.scale_tensor_regions()
}

///|
pub struct Fp8ReusablePagedRecipeV3 {
  priv source : Fp8RuntimeRecipeV2
  priv profile : @launch_contract.PagedKernelProfile
  priv kv_layout : @device_layout.DeviceKvLayout
  priv launches : ReadOnlyArray[Fp8ReusablePagedLaunchV3]
  priv workspace_capacity_bytes : Int64
  priv canonical : Bytes
  priv digest : Fp8ReusablePagedRecipeV3Digest
}

///|
pub fn Fp8ReusablePagedRecipeV3::startup(
  self : Fp8ReusablePagedRecipeV3,
) -> @fp8_startup_admission.Fp8StartupAdmissionV2 {
  self.source.startup()
}

///|
pub fn Fp8ReusablePagedRecipeV3::weight_inspection(
  self : Fp8ReusablePagedRecipeV3,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.source.weight_inspection()
}

///|
pub fn Fp8ReusablePagedRecipeV3::profile(
  self : Fp8ReusablePagedRecipeV3,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn Fp8ReusablePagedRecipeV3::kv_layout(
  self : Fp8ReusablePagedRecipeV3,
) -> @device_layout.DeviceKvLayout {
  self.kv_layout
}

///|
pub fn Fp8ReusablePagedRecipeV3::launches(
  self : Fp8ReusablePagedRecipeV3,
) -> ReadOnlyArray[Fp8ReusablePagedLaunchV3] {
  self.launches
}

///|
pub fn Fp8ReusablePagedRecipeV3::weight_bindings(
  self : Fp8ReusablePagedRecipeV3,
) -> ReadOnlyArray[Fp8LaunchWeightBindingV2] {
  self.source.weight_bindings()
}

///|
pub fn Fp8ReusablePagedRecipeV3::workspace_capacity_bytes(
  self : Fp8ReusablePagedRecipeV3,
) -> Int64 {
  self.workspace_capacity_bytes
}

///|
pub fn Fp8ReusablePagedRecipeV3::canonical_bytes(
  self : Fp8ReusablePagedRecipeV3,
) -> Bytes {
  self.canonical
}

///|
pub fn Fp8ReusablePagedRecipeV3::digest(
  self : Fp8ReusablePagedRecipeV3,
) -> Fp8ReusablePagedRecipeV3Digest {
  self.digest
}
