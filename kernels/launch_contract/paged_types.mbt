///|
/// Untrusted metadata for one positioned-rotary or paged-attention operation.
pub struct PagedKvAotLaunchContractInput {
  priv profile_id : KernelProfileId
  priv operation_id : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv dimensions : AotLaunchDimensions
  priv operands : ReadOnlyArray[AotOperand]
} derive(Debug, Eq)

///|
/// Fully admitted inert catalog-v3 paged-KV launch metadata.
///
/// This value is not executable. It contains no device address, CUDA symbol,
/// compiler path, fallback, or JIT channel.
pub struct PagedKvAotLaunchContract {
  priv profile : PagedKernelProfile
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv operation_shape : @plan.OperationShape
  priv capability : @plan.KernelCapabilityId
  priv semantic_version : @catalog.KernelSemanticVersion
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
/// Exact semantic coverage of one paged launch-contract set.
pub(all) enum PagedContractScope {
  KvSubsequence
  FullGraph
} derive(Debug, Eq)

///|
pub struct PagedKvLaunchContractSet {
  priv model_identity : @spec.ModelIdentity
  priv device_target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv scope : PagedContractScope
  priv device_kv_layout : @device_layout.DeviceKvLayout
  priv profiles : ReadOnlyArray[PagedKernelProfile]
  priv contracts : ReadOnlyArray[PagedKvAotLaunchContract]
} derive(Debug, Eq)

///|
pub fn PagedKvAotLaunchContractInput::new(
  profile_id~ : KernelProfileId,
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : AotLaunchDimensions,
  operands~ : ArrayView[AotOperand],
) -> PagedKvAotLaunchContractInput raise LaunchContractError {
  {
    profile_id,
    operation_id,
    entry_point,
    dimensions,
    operands: checked_owned_operands(operands, operation_id),
  }
}

///|
pub fn PagedKvLaunchContractSet::scope(
  self : PagedKvLaunchContractSet,
) -> PagedContractScope {
  self.scope
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
pub fn PagedKvAotLaunchContract::operation_shape(
  self : PagedKvAotLaunchContract,
) -> @plan.OperationShape {
  self.operation_shape
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
