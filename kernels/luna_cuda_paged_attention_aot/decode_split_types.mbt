///|
/// Deterministic two-stage decode split-K CUDA candidate.
///
/// This candidate is intentionally not manifest bindable: the current paged
/// launch contract has one attention launch and no explicit F32 partial
/// workspace operand. Production binding requires a later versioned contract
/// that names both entry points and this exact workspace geometry.
pub struct PagedAttentionDecodeSplitCudaBundle {
  priv dispatch_plan : @luna_attention_strategy.AttentionDispatchPlan
  priv workspace_plan : @luna_attention_strategy.AttentionDecodeWorkspacePlan
  priv target : @catalog.DeviceTarget
  priv operation_id : @plan.OperationId
  priv profile : @launch_contract.PagedKernelProfile
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
