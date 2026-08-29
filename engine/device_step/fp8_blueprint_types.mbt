///|
pub struct Fp8PagedExecutionStepV3 {
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_execution_digest : @numeric_contract.OperationExecutionDigest
  priv entry_point : @catalog.AotKernelEntryPoint
  priv artifact_module_index : Int
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[PagedKvOperandBinding]
  priv fp8_release_operation : Bool
}

///|
/// Inert complete mixed graph: BF16 paged steps retain their existing AOT
/// bindings while exactly the schema-marked projection families are replaced
/// by externally approved reusable FP8-v3 release operations.
pub struct Fp8PagedExecutionBlueprintV3 {
  priv authority : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3
  priv profile : @launch_contract.PagedKernelProfile
  priv limits : DeviceStepLimits
  priv layout : @device_layout.DeviceKvLayout
  priv weight_layout : @device_materialize.DeviceWeightLayout
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv artifacts : @artifact.KernelArtifactBundle
  priv fp8_workspace_base : Int64
  priv fp8_workspace_bytes : Int64
  priv activation_workspace_arena_bytes : Int64
  priv steps : ReadOnlyArray[Fp8PagedExecutionStepV3]
}

///|
pub fn Fp8PagedExecutionBlueprintV3::authority(
  self : Fp8PagedExecutionBlueprintV3,
) -> @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3 {
  self.authority
}

///|
pub fn Fp8PagedExecutionBlueprintV3::profile(
  self : Fp8PagedExecutionBlueprintV3,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn Fp8PagedExecutionBlueprintV3::limits(
  self : Fp8PagedExecutionBlueprintV3,
) -> DeviceStepLimits {
  self.limits
}

///|
pub fn Fp8PagedExecutionBlueprintV3::layout(
  self : Fp8PagedExecutionBlueprintV3,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn Fp8PagedExecutionBlueprintV3::weight_layout(
  self : Fp8PagedExecutionBlueprintV3,
) -> @device_materialize.DeviceWeightLayout {
  self.weight_layout
}

///|
pub fn Fp8PagedExecutionBlueprintV3::memory_plan(
  self : Fp8PagedExecutionBlueprintV3,
) -> @device_memory.DeviceMemoryPlan {
  self.memory_plan
}

///|
pub fn Fp8PagedExecutionBlueprintV3::artifacts(
  self : Fp8PagedExecutionBlueprintV3,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}

///|
pub fn Fp8PagedExecutionBlueprintV3::activation_workspace_arena_bytes(
  self : Fp8PagedExecutionBlueprintV3,
) -> Int64 {
  self.activation_workspace_arena_bytes
}

///|
pub fn Fp8PagedExecutionBlueprintV3::fp8_workspace_base(
  self : Fp8PagedExecutionBlueprintV3,
) -> Int64 {
  self.fp8_workspace_base
}

///|
pub fn Fp8PagedExecutionBlueprintV3::fp8_workspace_bytes(
  self : Fp8PagedExecutionBlueprintV3,
) -> Int64 {
  self.fp8_workspace_bytes
}

///|
pub fn Fp8PagedExecutionBlueprintV3::steps(
  self : Fp8PagedExecutionBlueprintV3,
) -> ReadOnlyArray[Fp8PagedExecutionStepV3] {
  self.steps
}

///|
pub fn Fp8PagedExecutionStepV3::operation_id(
  self : Fp8PagedExecutionStepV3,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn Fp8PagedExecutionStepV3::operation_kind(
  self : Fp8PagedExecutionStepV3,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn Fp8PagedExecutionStepV3::operation_execution_digest(
  self : Fp8PagedExecutionStepV3,
) -> @numeric_contract.OperationExecutionDigest {
  self.operation_execution_digest
}

///|
pub fn Fp8PagedExecutionStepV3::entry_point(
  self : Fp8PagedExecutionStepV3,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8PagedExecutionStepV3::artifact_module_index(
  self : Fp8PagedExecutionStepV3,
) -> Int {
  self.artifact_module_index
}

///|
pub fn Fp8PagedExecutionStepV3::function_symbol(
  self : Fp8PagedExecutionStepV3,
) -> String {
  self.function_symbol
}

///|
pub fn Fp8PagedExecutionStepV3::dimensions(
  self : Fp8PagedExecutionStepV3,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn Fp8PagedExecutionStepV3::operands(
  self : Fp8PagedExecutionStepV3,
) -> ReadOnlyArray[PagedKvOperandBinding] {
  self.operands
}

///|
pub fn Fp8PagedExecutionStepV3::is_fp8_release_operation(
  self : Fp8PagedExecutionStepV3,
) -> Bool {
  self.fp8_release_operation
}
