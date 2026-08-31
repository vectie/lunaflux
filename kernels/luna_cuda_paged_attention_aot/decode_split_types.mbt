///|
/// Deterministic two-stage decode split-K CUDA candidate.
///
/// The candidate alone is not manifest bindable. Production binding is a
/// separate startup join against an exact ordered multi-launch contract.
pub struct PagedAttentionDecodeSplitCudaBundle {
  priv dispatch_plan : @luna_attention_strategy.AttentionDispatchPlan
  priv workspace_plan : @luna_attention_strategy.AttentionDecodeWorkspacePlan
  priv target : @catalog.DeviceTarget
  priv operation_id : @plan.OperationId
  priv profile : @launch_contract.PagedKernelProfile
  priv layout : @device_layout.DeviceKvLayout
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv tokens_per_page : Int
  priv page_stride_bytes : Int64
  priv physical_page_capacity : Int64
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv partial_entry_point_id : @catalog.AotKernelEntryPointId
  priv merge_entry_point_id : @catalog.AotKernelEntryPointId
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv partial_function_symbol : String
  priv merge_function_symbol : String
  priv partial_dimensions : @launch_contract.AotLaunchDimensions
  priv merge_dimensions : @launch_contract.AotLaunchDimensions
  priv source : Bytes
  priv source_digest : PagedAttentionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : PagedAttentionCudaRecipeDigest
} derive(Debug, Eq)

///|
/// Exact post-compile startup join for the two-stage decode path. The legacy
/// single-launch manifest binding remains an independent ABI.
pub struct PagedAttentionDecodeSplitManifestCudaBinding {
  priv candidate : PagedAttentionDecodeSplitCudaBundle
  priv contract : @luna_attention_multilaunch_contract.AttentionDecodeMultiLaunchContract
  priv binding_record : Bytes
  priv binding_digest : PagedAttentionManifestBindingDigest
} derive(Debug, Eq)

///|
pub fn PagedAttentionDecodeSplitCudaBundle::dispatch_plan(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @luna_attention_strategy.AttentionDispatchPlan {
  self.dispatch_plan
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::workspace_plan(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @luna_attention_strategy.AttentionDecodeWorkspacePlan {
  self.workspace_plan
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::target(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::operation_id(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::profile(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::device_kv_layout(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::query_heads(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int {
  self.query_heads
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::kv_heads(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int {
  self.kv_heads
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::head_dimension(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int {
  self.head_dimension
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::tokens_per_page(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int {
  self.tokens_per_page
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::page_stride_bytes(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::physical_page_capacity(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Int64 {
  self.physical_page_capacity
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::operands(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::entry_point_ids(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> (@catalog.AotKernelEntryPointId, @catalog.AotKernelEntryPointId) {
  (self.partial_entry_point_id, self.merge_entry_point_id)
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::compiler_policy(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::function_symbols(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> (String, String) {
  (self.partial_function_symbol, self.merge_function_symbol)
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::launch_dimensions(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> (
  @launch_contract.AotLaunchDimensions,
  @launch_contract.AotLaunchDimensions,
) {
  (self.partial_dimensions, self.merge_dimensions)
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::source_bytes(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Bytes {
  self.source
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::source_digest(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> PagedAttentionCudaSourceDigest {
  self.source_digest
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::recipe_bytes(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::recipe_digest(
  self : PagedAttentionDecodeSplitCudaBundle,
) -> PagedAttentionCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionDecodeSplitCudaBundle::manifest_bindable(
  _self : PagedAttentionDecodeSplitCudaBundle,
) -> Bool {
  false
}

///|
pub fn PagedAttentionDecodeSplitManifestCudaBinding::candidate(
  self : PagedAttentionDecodeSplitManifestCudaBinding,
) -> PagedAttentionDecodeSplitCudaBundle {
  self.candidate
}

///|
pub fn PagedAttentionDecodeSplitManifestCudaBinding::contract(
  self : PagedAttentionDecodeSplitManifestCudaBinding,
) -> @luna_attention_multilaunch_contract.AttentionDecodeMultiLaunchContract {
  self.contract
}

///|
pub fn PagedAttentionDecodeSplitManifestCudaBinding::binding_record_bytes(
  self : PagedAttentionDecodeSplitManifestCudaBinding,
) -> Bytes {
  self.binding_record
}

///|
pub fn PagedAttentionDecodeSplitManifestCudaBinding::binding_digest(
  self : PagedAttentionDecodeSplitManifestCudaBinding,
) -> PagedAttentionManifestBindingDigest {
  self.binding_digest
}

///|
pub fn PagedAttentionDecodeSplitManifestCudaBinding::manifest_bindable(
  _self : PagedAttentionDecodeSplitManifestCudaBinding,
) -> Bool {
  true
}
