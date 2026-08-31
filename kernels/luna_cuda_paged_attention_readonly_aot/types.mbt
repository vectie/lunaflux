///|
pub(all) enum PagedAttentionReadOnlyKernelAbi {
  RotatedQueryPagedKvReadOnlyRawPointerV1
  RotatedQueryPagedKvReadOnlyProductionRawPointerV2
} derive(Debug, Eq)

///|
pub fn PagedAttentionReadOnlyKernelAbi::canonical_name(
  self : PagedAttentionReadOnlyKernelAbi,
) -> String {
  match self {
    RotatedQueryPagedKvReadOnlyRawPointerV1 =>
      "step_counts:i32*,query_positions:i32*,query_row_offsets:i32*," +
      "sequence_lengths:i32*,page_table_offsets:i32*," +
      "page_table_page_indices:i32*,rotated_qkv_rows:const_bf16*," +
      "output:bf16*,key_cache:const_bf16*,value_cache:const_bf16*," +
      "dispatch_canary:u32*"
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2 =>
      "step_counts:i32*,query_positions:i32*,query_row_offsets:i32*," +
      "sequence_lengths:i32*,page_table_offsets:i32*," +
      "page_table_page_indices:i32*,rotated_qkv_rows:const_bf16*," +
      "output:bf16*,key_cache:const_bf16*,value_cache:const_bf16*"
  }
}

///|
pub struct PagedAttentionReadOnlyAotLimits {
  priv maximum_source_bytes : Int
  priv maximum_recipe_bytes : Int
  priv maximum_canonical_bytes : Int
}

///|
pub fn PagedAttentionReadOnlyAotLimits::new(
  maximum_source_bytes~ : Int,
  maximum_recipe_bytes~ : Int,
  maximum_canonical_bytes~ : Int,
) -> PagedAttentionReadOnlyAotLimits raise PagedAttentionReadOnlyAotError {
  if maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 ||
    maximum_recipe_bytes <= 0 ||
    maximum_recipe_bytes > 1048576 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 1048576 {
    raise PagedAttentionReadOnlyInvalid(Limits)
  }
  { maximum_source_bytes, maximum_recipe_bytes, maximum_canonical_bytes }
}

///|
/// Qualification-only candidate for the post-write, cache-read-only seam.
pub struct PagedAttentionReadOnlyCudaCandidate {
  priv reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv function_symbol : String
  priv kernel_abi : PagedAttentionReadOnlyKernelAbi
  priv input_row_width : Int
  priv input_activation : @plan.ActivationRef
  priv dispatch_canary_per_token : Int
  priv launch : @launch_contract.AotLaunchDimensions
  priv source : Bytes
  priv source_digest : PagedAttentionReadOnlyDigest
  priv recipe : Bytes
  priv recipe_digest : PagedAttentionReadOnlyDigest
  priv canonical : Bytes
  priv candidate_digest : PagedAttentionReadOnlyDigest
}

///|
/// Canary-free production companion for one exact readonly qualification
/// candidate. It has no deployment or dispatch authority.
pub struct PagedAttentionReadOnlyProductionCudaCandidate {
  priv qualification : PagedAttentionReadOnlyCudaCandidate
  priv function_symbol : String
  priv kernel_abi : PagedAttentionReadOnlyKernelAbi
  priv launch : @launch_contract.AotLaunchDimensions
  priv source : Bytes
  priv source_digest : PagedAttentionReadOnlyDigest
  priv recipe : Bytes
  priv recipe_digest : PagedAttentionReadOnlyDigest
  priv canonical : Bytes
  priv candidate_digest : PagedAttentionReadOnlyDigest
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::reference_fallback(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering {
  self.reference
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::target(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @catalog.DeviceTarget {
  self.reference.target()
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::profile(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.reference.profile()
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::operation_id(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @plan.OperationId {
  self.reference.operation_id()
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::entry_point_id(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::function_symbol(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::kernel_abi(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> PagedAttentionReadOnlyKernelAbi {
  self.kernel_abi
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::input_row_width(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> Int {
  self.input_row_width
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::input_activation(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @plan.ActivationRef {
  self.input_activation
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::dispatch_canary_per_token(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> Int {
  self.dispatch_canary_per_token
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::launch_dimensions(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::source_bytes(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::source_digest(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.source_digest
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::recipe_bytes(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::recipe_digest(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::canonical_bytes(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> Bytes {
  self.canonical
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::candidate_digest(
  self : PagedAttentionReadOnlyCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.candidate_digest
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::manifest_bindable(
  _self : PagedAttentionReadOnlyCudaCandidate,
) -> Bool {
  false
}

///|
pub fn PagedAttentionReadOnlyCudaCandidate::promotion_authorized(
  _self : PagedAttentionReadOnlyCudaCandidate,
) -> Bool {
  false
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::qualification_candidate_digest(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.qualification.candidate_digest
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::function_symbol(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::kernel_abi(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> PagedAttentionReadOnlyKernelAbi {
  self.kernel_abi
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::launch_dimensions(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::source_bytes(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::source_digest(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.source_digest
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::recipe_bytes(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::recipe_digest(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::canonical_bytes(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> Bytes {
  self.canonical
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::candidate_digest(
  self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> PagedAttentionReadOnlyDigest {
  self.candidate_digest
}

///|
pub fn PagedAttentionReadOnlyProductionCudaCandidate::manifest_bindable(
  _self : PagedAttentionReadOnlyProductionCudaCandidate,
) -> Bool {
  false
}
