///|
/// Physical allocation class for one inert paged-KV operand binding.
///
/// Values contain scalar offsets only. They carry no allocation alias, device
/// address, kernel argument, or lifecycle authority.
pub(all) enum PagedKvOperandSpace {
  StepDescriptor(DeviceStepStage)
  WeightArena
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
  priv kv_layer : @plan.KvLayerId?
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
  priv scope : @launch_contract.PagedContractScope
  priv profile : @launch_contract.PagedKernelProfile
  priv limits : DeviceStepLimits
  priv layout : @device_layout.DeviceKvLayout
  priv weight_arena_bytes : Int64
  priv weight_alignment : Int64
  priv activation_workspace_arena_bytes : Int64
  priv graph_memory : PagedGraphMemoryEvidence
  priv graph_shape : @luna_capability_manifest.LunaGraphShapeClass?
  priv steps : ReadOnlyArray[PagedKvExecutionStep]
} derive(Debug, Eq)

///|
/// One catalog-v4 physical step whose numeric operation identity and complete
/// positioned operand sequence have been reauthenticated for production I8.
pub struct I8PagedExecutionStep {
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv kv_layer : @plan.KvLayerId?
  priv capability : @plan.KernelCapabilityId
  priv semantic_version : @catalog.KernelSemanticVersion
  priv operation_execution_digest : @numeric_contract.OperationExecutionDigest
  priv entry_point : @catalog.AotKernelEntryPoint
  priv artifact_module_index : Int
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[PagedKvOperandBinding]
} derive(Debug, Eq)

///|
/// Inert pre-hardware blueprint for the complete production symmetric-I8
/// paged graph.
///
/// The full numeric weight layout is retained so every parameter and metadata
/// tensor remains authenticated. Scale operands preserve their exact
/// `WeightScaleInput(weight, scale)` role and region. This value contains no
/// loaded-weight authority, allocation, module, function, kernel argument,
/// executor, or readiness state.
pub struct I8PagedExecutionBlueprint {
  priv identity : @spec.ModelIdentity
  priv numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest
  priv i8_admission_digest : @i8_admission.I8InertKernelCapabilityAdmissionDigest
  priv device_numeric_digest : @device.DeviceNumericCapabilityDigest
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv profile : @launch_contract.PagedKernelProfile
  priv limits : DeviceStepLimits
  priv layout : @device_layout.DeviceKvLayout
  priv weight_layout : @device_materialize.DeviceWeightLayout
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv artifacts : @artifact.KernelArtifactBundle
  priv activation_workspace_arena_bytes : Int64
  priv steps : ReadOnlyArray[I8PagedExecutionStep]
}

///|
pub fn PagedKvOperandBinding::role(
  self : PagedKvOperandBinding,
) -> @launch_contract.AotOperandRole {
  self.role
}

///|
/// Inert post-load prerequisite for production symmetric-I8 execution.
///
/// This value retains the only authenticated numeric-weight allocation owner,
/// but deliberately exposes neither that allocation nor a module/function
/// primitive. The I8 executor preparation API consumes this exact prerequisite
/// and still makes no physical CUDA-correctness or release-readiness claim.
pub struct I8LoadedExecutionBlueprint {
  priv blueprint : I8PagedExecutionBlueprint
  priv bootstrap : I8DeviceWorkerBootstrapManifestV3
  priv weight_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest
  priv weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority
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
) -> @plan.KvLayerId? {
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
pub fn PagedKvExecutionBlueprint::scope(
  self : PagedKvExecutionBlueprint,
) -> @launch_contract.PagedContractScope {
  self.scope
}

///|
/// Returns the exact reusable descriptor and scalar envelope admitted for this
/// blueprint. Profile dimensions cover rows, tokens, and page-table cells;
/// this value additionally preserves sequence and token-ID bounds.
pub fn PagedKvExecutionBlueprint::limits(
  self : PagedKvExecutionBlueprint,
) -> DeviceStepLimits {
  self.limits
}

///|
pub fn PagedKvExecutionBlueprint::layout(
  self : PagedKvExecutionBlueprint,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn PagedKvExecutionBlueprint::weight_arena_bytes(
  self : PagedKvExecutionBlueprint,
) -> Int64 {
  self.weight_arena_bytes
}

///|
pub fn PagedKvExecutionBlueprint::weight_alignment(
  self : PagedKvExecutionBlueprint,
) -> Int64 {
  self.weight_alignment
}

///|
pub fn PagedKvExecutionBlueprint::activation_workspace_arena_bytes(
  self : PagedKvExecutionBlueprint,
) -> Int64 {
  self.activation_workspace_arena_bytes
}

///|
pub fn PagedKvExecutionBlueprint::steps(
  self : PagedKvExecutionBlueprint,
) -> ReadOnlyArray[PagedKvExecutionStep] {
  self.steps
}

///|
pub fn I8PagedExecutionStep::operation_id(
  self : I8PagedExecutionStep,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn I8PagedExecutionStep::operation_kind(
  self : I8PagedExecutionStep,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn I8PagedExecutionStep::kv_layer(
  self : I8PagedExecutionStep,
) -> @plan.KvLayerId? {
  self.kv_layer
}

///|
pub fn I8PagedExecutionStep::capability(
  self : I8PagedExecutionStep,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn I8PagedExecutionStep::semantic_version(
  self : I8PagedExecutionStep,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn I8PagedExecutionStep::operation_execution_digest(
  self : I8PagedExecutionStep,
) -> @numeric_contract.OperationExecutionDigest {
  self.operation_execution_digest
}

///|
pub fn I8PagedExecutionStep::entry_point(
  self : I8PagedExecutionStep,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn I8PagedExecutionStep::artifact_module_index(
  self : I8PagedExecutionStep,
) -> Int {
  self.artifact_module_index
}

///|
pub fn I8PagedExecutionStep::function_symbol(
  self : I8PagedExecutionStep,
) -> String {
  self.function_symbol
}

///|
pub fn I8PagedExecutionStep::dimensions(
  self : I8PagedExecutionStep,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn I8PagedExecutionStep::operands(
  self : I8PagedExecutionStep,
) -> ReadOnlyArray[PagedKvOperandBinding] {
  self.operands
}

///|
pub fn I8PagedExecutionBlueprint::identity(
  self : I8PagedExecutionBlueprint,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn I8PagedExecutionBlueprint::numeric_schema_digest(
  self : I8PagedExecutionBlueprint,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema_digest
}

///|
pub fn I8PagedExecutionBlueprint::i8_admission_digest(
  self : I8PagedExecutionBlueprint,
) -> @i8_admission.I8InertKernelCapabilityAdmissionDigest {
  self.i8_admission_digest
}

///|
pub fn I8PagedExecutionBlueprint::device_numeric_digest(
  self : I8PagedExecutionBlueprint,
) -> @device.DeviceNumericCapabilityDigest {
  self.device_numeric_digest
}

///|
pub fn I8PagedExecutionBlueprint::target(
  self : I8PagedExecutionBlueprint,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn I8PagedExecutionBlueprint::catalog_version(
  self : I8PagedExecutionBlueprint,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn I8PagedExecutionBlueprint::profile(
  self : I8PagedExecutionBlueprint,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn I8PagedExecutionBlueprint::limits(
  self : I8PagedExecutionBlueprint,
) -> DeviceStepLimits {
  self.limits
}

///|
pub fn I8PagedExecutionBlueprint::layout(
  self : I8PagedExecutionBlueprint,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn I8PagedExecutionBlueprint::weight_layout(
  self : I8PagedExecutionBlueprint,
) -> @device_materialize.DeviceWeightLayout {
  self.weight_layout
}

///|
pub fn I8PagedExecutionBlueprint::activation_workspace_arena_bytes(
  self : I8PagedExecutionBlueprint,
) -> Int64 {
  self.activation_workspace_arena_bytes
}

///|
pub fn I8PagedExecutionBlueprint::steps(
  self : I8PagedExecutionBlueprint,
) -> ReadOnlyArray[I8PagedExecutionStep] {
  self.steps
}

///|
pub fn I8LoadedExecutionBlueprint::blueprint(
  self : I8LoadedExecutionBlueprint,
) -> I8PagedExecutionBlueprint {
  self.blueprint
}

///|
pub fn I8LoadedExecutionBlueprint::bootstrap_manifest(
  self : I8LoadedExecutionBlueprint,
) -> I8DeviceWorkerBootstrapManifestV3 {
  self.bootstrap
}

///|
pub fn I8LoadedExecutionBlueprint::weight_artifact_digest(
  self : I8LoadedExecutionBlueprint,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  self.weight_artifact_digest
}

///|
/// Deterministically releases the retained numeric-weight arena.
pub fn I8LoadedExecutionBlueprint::close(
  self : I8LoadedExecutionBlueprint,
) -> Unit raise DeviceStepError {
  self.weights.close() catch {
    _ => raise NumericWeightClose
  }
}
