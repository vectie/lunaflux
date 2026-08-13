///|
/// Untrusted metadata for one positioned-rotary or paged-attention operation.
pub struct PagedKvAotLaunchContractInput {
  profile_id : KernelProfileId
  operation_id : @plan.OperationId
  entry_point : @catalog.AotKernelEntryPoint
  dimensions : AotLaunchDimensions
  operands : ReadOnlyArray[AotOperand]
} derive(Debug, Eq)

///|
/// Fully admitted inert catalog-v2 paged-KV launch metadata.
///
/// This value is not executable. It contains no device address, CUDA symbol,
/// compiler path, fallback, or JIT channel.
pub struct PagedKvAotLaunchContract {
  profile : PagedKernelProfile
  operation_id : @plan.OperationId
  operation_kind : @plan.OperationKind
  capability : @plan.KernelCapabilityId
  semantic_version : @catalog.KernelSemanticVersion
  kv_layer : @plan.KvLayerId?
  runtime_inputs : ReadOnlyArray[@plan.OperationRuntimeInput]
  persistent_state : @plan.PersistentStateAccess
  kv_layout_version : Int
  tokens_per_page : Int
  entry_point : @catalog.AotKernelEntryPoint
  dimensions : AotLaunchDimensions
  operands : ReadOnlyArray[AotOperand]
  workspace_bytes : Int64
  workspace_alignment : Int64
} derive(Debug, Eq)

///|
pub struct PagedKvLaunchContractSet {
  model_identity : @spec.ModelIdentity
  device_target : @catalog.DeviceTarget
  catalog_version : @catalog.CatalogVersion
  device_kv_layout : @device_layout.DeviceKvLayout
  profiles : ReadOnlyArray[PagedKernelProfile]
  contracts : ReadOnlyArray[PagedKvAotLaunchContract]
} derive(Debug, Eq)

///|
pub fn PagedKvAotLaunchContractInput::new(
  profile_id~ : KernelProfileId,
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : AotLaunchDimensions,
  operands~ : ArrayView[AotOperand],
) -> PagedKvAotLaunchContractInput {
  {
    profile_id,
    operation_id,
    entry_point,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
  }
}

///|
pub fn PagedKvAotLaunchContract::profile(
  self : PagedKvAotLaunchContract,
) -> PagedKernelProfile {
  self.profile
}

///|
pub fn PagedKvAotLaunchContract::operation_id(
  self : PagedKvAotLaunchContract,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PagedKvAotLaunchContract::operation_kind(
  self : PagedKvAotLaunchContract,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn PagedKvAotLaunchContract::capability(
  self : PagedKvAotLaunchContract,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn PagedKvAotLaunchContract::semantic_version(
  self : PagedKvAotLaunchContract,
) -> @catalog.KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn PagedKvAotLaunchContract::kv_layer(
  self : PagedKvAotLaunchContract,
) -> @plan.KvLayerId? {
  self.kv_layer
}

///|
pub fn PagedKvAotLaunchContract::runtime_inputs(
  self : PagedKvAotLaunchContract,
) -> ReadOnlyArray[@plan.OperationRuntimeInput] {
  self.runtime_inputs
}

///|
pub fn PagedKvAotLaunchContract::persistent_state(
  self : PagedKvAotLaunchContract,
) -> @plan.PersistentStateAccess {
  self.persistent_state
}

///|
pub fn PagedKvAotLaunchContract::kv_layout_version(
  self : PagedKvAotLaunchContract,
) -> Int {
  self.kv_layout_version
}

///|
pub fn PagedKvAotLaunchContract::tokens_per_page(
  self : PagedKvAotLaunchContract,
) -> Int {
  self.tokens_per_page
}

///|
pub fn PagedKvAotLaunchContract::entry_point(
  self : PagedKvAotLaunchContract,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn PagedKvAotLaunchContract::dimensions(
  self : PagedKvAotLaunchContract,
) -> AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedKvAotLaunchContract::operands(
  self : PagedKvAotLaunchContract,
) -> ReadOnlyArray[AotOperand] {
  self.operands
}

///|
pub fn PagedKvAotLaunchContract::workspace_bytes(
  self : PagedKvAotLaunchContract,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn PagedKvAotLaunchContract::workspace_alignment(
  self : PagedKvAotLaunchContract,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn PagedKvLaunchContractSet::model_identity(
  self : PagedKvLaunchContractSet,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn PagedKvLaunchContractSet::catalog_version(
  self : PagedKvLaunchContractSet,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
pub fn PagedKvLaunchContractSet::device_target(
  self : PagedKvLaunchContractSet,
) -> @catalog.DeviceTarget {
  self.device_target
}

///|
pub fn PagedKvLaunchContractSet::device_kv_layout(
  self : PagedKvLaunchContractSet,
) -> @device_layout.DeviceKvLayout {
  self.device_kv_layout
}

///|
pub fn PagedKvLaunchContractSet::profiles(
  self : PagedKvLaunchContractSet,
) -> ReadOnlyArray[PagedKernelProfile] {
  self.profiles
}

///|
pub fn PagedKvLaunchContractSet::contracts(
  self : PagedKvLaunchContractSet,
) -> ReadOnlyArray[PagedKvAotLaunchContract] {
  self.contracts
}
