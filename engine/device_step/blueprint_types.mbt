///|
/// Physical allocation class for one inert paged-KV operand binding.
///
/// Values contain scalar offsets only. They carry no allocation alias, device
/// address, kernel argument, or lifecycle authority.
pub(all) enum PagedKvOperandSpace {
  StepDescriptor(DeviceStepStage)
  ActivationWorkspaceArena
  PersistentKvArena
} derive(Debug, Eq)

///|
/// Immutable scalar placement for one exact semantic kernel operand.
pub struct PagedKvOperandBinding {
  priv role : @launch_contract.AotOperandRole
  priv space : PagedKvOperandSpace
  priv offset_bytes : Int64
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
/// One admitted positioned-rotary or paged-attention launch step.
pub struct PagedKvExecutionStep {
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv kv_layer : @plan.KvLayerId
  priv capability : @plan.KernelCapabilityId
  priv semantic_version : @catalog.KernelSemanticVersion
  priv entry_point : @catalog.AotKernelEntryPoint
  priv artifact_module_index : Int
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[PagedKvOperandBinding]
} derive(Debug, Eq)

///|
/// Opaque immutable startup blueprint for the paged-KV operation subsequence.
///
/// This value is deliberately inert. It opens no module or device resource,
/// constructs no kernel argument, and cannot launch work.
pub struct PagedKvExecutionBlueprint {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv profile : @launch_contract.PagedKernelProfile
  priv layout : @device_layout.DeviceKvLayout
  priv steps : ReadOnlyArray[PagedKvExecutionStep]
} derive(Debug, Eq)

///|
pub fn PagedKvOperandBinding::role(
  self : PagedKvOperandBinding,
) -> @launch_contract.AotOperandRole {
  self.role
}

///|
pub fn PagedKvOperandBinding::space(
  self : PagedKvOperandBinding,
) -> PagedKvOperandSpace {
  self.space
}

///|
pub fn PagedKvOperandBinding::offset_bytes(
  self : PagedKvOperandBinding,
) -> Int64 {
  self.offset_bytes
}

///|
pub fn PagedKvOperandBinding::byte_count(self : PagedKvOperandBinding) -> Int64 {
  self.byte_count
}

///|
pub fn PagedKvOperandBinding::alignment(self : PagedKvOperandBinding) -> Int64 {
  self.alignment
}

///|
pub fn PagedKvExecutionStep::operation_id(
  self : PagedKvExecutionStep,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PagedKvExecutionStep::operation_kind(
  self : PagedKvExecutionStep,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn PagedKvExecutionStep::kv_layer(
  self : PagedKvExecutionStep,
) -> @plan.KvLayerId {
  self.kv_layer
}

///|
pub fn PagedKvExecutionStep::capability(
  self : PagedKvExecutionStep,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn PagedKvExecutionStep::semantic_version(
  self : PagedKvExecutionStep,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn PagedKvExecutionStep::entry_point(
  self : PagedKvExecutionStep,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PagedKvExecutionStep::artifact_module_index(
  self : PagedKvExecutionStep,
) -> Int {
  self.artifact_module_index
}

///|
pub fn PagedKvExecutionStep::function_symbol(
  self : PagedKvExecutionStep,
) -> String {
  self.function_symbol
}

///|
pub fn PagedKvExecutionStep::dimensions(
  self : PagedKvExecutionStep,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedKvExecutionStep::operands(
  self : PagedKvExecutionStep,
) -> ReadOnlyArray[PagedKvOperandBinding] {
  self.operands
}

///|
pub fn PagedKvExecutionBlueprint::identity(
  self : PagedKvExecutionBlueprint,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn PagedKvExecutionBlueprint::target(
  self : PagedKvExecutionBlueprint,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PagedKvExecutionBlueprint::catalog_version(
  self : PagedKvExecutionBlueprint,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn PagedKvExecutionBlueprint::profile(
  self : PagedKvExecutionBlueprint,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn PagedKvExecutionBlueprint::layout(
  self : PagedKvExecutionBlueprint,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn PagedKvExecutionBlueprint::steps(
  self : PagedKvExecutionBlueprint,
) -> ReadOnlyArray[PagedKvExecutionStep] {
  self.steps
}
