///|
/// Untrusted positioned input for one catalog-v4 full-graph launch contract.
/// Construction defensively owns the operand table; only v4 admission can
/// mint the corresponding contract and set authority.
pub struct PagedV4AotLaunchContractInput {
  priv profile_id : KernelProfileId
  priv operation_id : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : AotLaunchDimensions
  priv operands : ReadOnlyArray[AotOperand]
} derive(Debug, Eq)

///|
/// Fully admitted inert catalog-v4 full-graph launch metadata.
pub struct PagedV4AotLaunchContract {
  priv profile : PagedKernelProfile
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_shape : @plan.OperationShape
  priv capability : @plan.KernelCapabilityId
  priv semantic_version : @catalog.KernelSemanticVersion
  priv operation_execution_digest : @numeric_contract.OperationExecutionDigest
  priv kv_layer : @plan.KvLayerId?
  priv runtime_inputs : ReadOnlyArray[@plan.OperationRuntimeInput]
  priv persistent_state : @plan.PersistentStateAccess
  priv kv_layout_version : Int
  priv tokens_per_page : Int
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : AotLaunchDimensions
  priv operands : ReadOnlyArray[AotOperand]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
} derive(Debug, Eq)

///|
/// Exact complete-graph catalog-v4 launch coverage for one numeric model ABI.
pub struct PagedV4LaunchContractSet {
  priv model_identity : @spec.ModelIdentity
  priv numeric_schema_digest : @numeric_contract.ModelNumericSchemaDigest
  priv device_target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv device_kv_layout : @device_layout.DeviceKvLayout
  priv profiles : ReadOnlyArray[PagedKernelProfile]
  priv contracts : ReadOnlyArray[PagedV4AotLaunchContract]
} derive(Debug, Eq)

///|
pub fn PagedV4AotLaunchContractInput::new(
  profile_id~ : KernelProfileId,
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : AotLaunchDimensions,
  operands~ : ArrayView[AotOperand],
) -> PagedV4AotLaunchContractInput raise LaunchContractError {
  {
    profile_id,
    operation_id,
    entry_point,
    dimensions,
    operands: checked_owned_operands(operands, operation_id),
  }
}

///|
pub fn PagedV4AotLaunchContract::profile(
  self : PagedV4AotLaunchContract,
) -> PagedKernelProfile {
  self.profile
}

///|
pub fn PagedV4AotLaunchContract::operation_id(
  self : PagedV4AotLaunchContract,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PagedV4AotLaunchContract::operation_kind(
  self : PagedV4AotLaunchContract,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn PagedV4AotLaunchContract::operation_shape(
  self : PagedV4AotLaunchContract,
) -> @plan.OperationShape {
  self.operation_shape
}

///|
pub fn PagedV4AotLaunchContract::capability(
  self : PagedV4AotLaunchContract,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn PagedV4AotLaunchContract::semantic_version(
  self : PagedV4AotLaunchContract,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn PagedV4AotLaunchContract::operation_execution_digest(
  self : PagedV4AotLaunchContract,
) -> @numeric_contract.OperationExecutionDigest {
  self.operation_execution_digest
}

///|
pub fn PagedV4AotLaunchContract::kv_layer(
  self : PagedV4AotLaunchContract,
) -> @plan.KvLayerId? {
  self.kv_layer
}

///|
pub fn PagedV4AotLaunchContract::runtime_inputs(
  self : PagedV4AotLaunchContract,
) -> ReadOnlyArray[@plan.OperationRuntimeInput] {
  self.runtime_inputs
}

///|
pub fn PagedV4AotLaunchContract::persistent_state(
  self : PagedV4AotLaunchContract,
) -> @plan.PersistentStateAccess {
  self.persistent_state
}

///|
pub fn PagedV4AotLaunchContract::kv_layout_version(
  self : PagedV4AotLaunchContract,
) -> Int {
  self.kv_layout_version
}

///|
pub fn PagedV4AotLaunchContract::tokens_per_page(
  self : PagedV4AotLaunchContract,
) -> Int {
  self.tokens_per_page
}

///|
pub fn PagedV4AotLaunchContract::entry_point(
  self : PagedV4AotLaunchContract,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PagedV4AotLaunchContract::dimensions(
  self : PagedV4AotLaunchContract,
) -> AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedV4AotLaunchContract::operands(
  self : PagedV4AotLaunchContract,
) -> ReadOnlyArray[AotOperand] {
  self.operands
}

///|
pub fn PagedV4AotLaunchContract::workspace_bytes(
  self : PagedV4AotLaunchContract,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn PagedV4AotLaunchContract::workspace_alignment(
  self : PagedV4AotLaunchContract,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn PagedV4LaunchContractSet::model_identity(
  self : PagedV4LaunchContractSet,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn PagedV4LaunchContractSet::numeric_schema_digest(
  self : PagedV4LaunchContractSet,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema_digest
}

///|
pub fn PagedV4LaunchContractSet::device_target(
  self : PagedV4LaunchContractSet,
) -> @catalog.DeviceTarget {
  self.device_target
}

///|
pub fn PagedV4LaunchContractSet::catalog_version(
  self : PagedV4LaunchContractSet,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn PagedV4LaunchContractSet::device_kv_layout(
  self : PagedV4LaunchContractSet,
) -> @device_layout.DeviceKvLayout {
  self.device_kv_layout
}

///|
pub fn PagedV4LaunchContractSet::profiles(
  self : PagedV4LaunchContractSet,
) -> ReadOnlyArray[PagedKernelProfile] {
  self.profiles
}

///|
pub fn PagedV4LaunchContractSet::contracts(
  self : PagedV4LaunchContractSet,
) -> ReadOnlyArray[PagedV4AotLaunchContract] {
  self.contracts
}

///|
pub fn PagedV4LaunchContractSet::contract(
  self : PagedV4LaunchContractSet,
  profile_id : KernelProfileId,
  operation_id : @plan.OperationId,
) -> PagedV4AotLaunchContract? {
  for contract in self.contracts {
    if contract.profile.id == profile_id &&
      contract.operation_id == operation_id {
      return Some(contract)
    }
  }
  None
}
