///|
const PAGED_ATTENTION_PARALLEL_BLOCK_THREADS : Int = 128

///|
const PAGED_ATTENTION_PARALLEL_MAX_CONTEXT : Int = 8192

///|
const PAGED_ATTENTION_PARALLEL_MAX_SHARED_BYTES : Int = 48 * 1024

///|
pub struct PagedAttentionParallelAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_canonical_bytes : Int
}

///|
pub fn PagedAttentionParallelAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_canonical_bytes~ : Int,
) -> PagedAttentionParallelAotLimits raise PagedAttentionParallelAotError {
  if max_source_bytes <= 0 ||
    max_source_bytes > 16777216 ||
    max_recipe_bytes <= 0 ||
    max_recipe_bytes > 1048576 ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > 1048576 {
    raise PagedAttentionParallelInvalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes, max_canonical_bytes }
}

///|
pub struct PagedAttentionParallelDigest(String) derive(Debug, Eq, Hash)

///|
pub fn PagedAttentionParallelDigest::from_sha256(
  value : String,
) -> PagedAttentionParallelDigest raise PagedAttentionParallelAotError {
  if value.length() != 64 {
    raise PagedAttentionParallelInvalid(Digest)
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise PagedAttentionParallelInvalid(Digest)
    }
  }
  PagedAttentionParallelDigest(value)
}

///|
pub fn PagedAttentionParallelDigest::as_hex(
  self : PagedAttentionParallelDigest,
) -> String {
  self.0
}

///|
/// Separately named numerical contract for the tree-reduced candidate. It is
/// intentionally not the serial reference kernel's operation-order contract.
pub struct PagedAttentionParallelNumericalPolicy {
  priv name : String
  priv maximum_absolute_error_ppb : Int64
  priv maximum_relative_error_ppb : Int64
  priv requires_token_agreement : Bool
}

///|
fn paged_attention_parallel_numerical_policy() -> PagedAttentionParallelNumericalPolicy {
  {
    name: "bf16-input-fp32-block128-tree-softmax-tolerance-v1",
    maximum_absolute_error_ppb: 4000000L,
    maximum_relative_error_ppb: 4000000L,
    requires_token_agreement: true,
  }
}

///|
pub fn PagedAttentionParallelNumericalPolicy::name(
  self : PagedAttentionParallelNumericalPolicy,
) -> String {
  self.name
}

///|
pub fn PagedAttentionParallelNumericalPolicy::maximum_absolute_error_ppb(
  self : PagedAttentionParallelNumericalPolicy,
) -> Int64 {
  self.maximum_absolute_error_ppb
}

///|
pub fn PagedAttentionParallelNumericalPolicy::maximum_relative_error_ppb(
  self : PagedAttentionParallelNumericalPolicy,
) -> Int64 {
  self.maximum_relative_error_ppb
}

///|
pub fn PagedAttentionParallelNumericalPolicy::requires_token_agreement(
  self : PagedAttentionParallelNumericalPolicy,
) -> Bool {
  self.requires_token_agreement
}

///|
/// Non-bindable offline candidate. It retains the independently lowered serial
/// reference as explicit fallback identity, never as an alias of this source.
pub struct PagedAttentionParallelCudaCandidate {
  priv reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv maximum_context_tokens : Int
  priv shared_memory_bytes : Int
  priv tile_program_digest : @luna_tile_ir.LunaTileProgramDigest
  priv tile_aot_digest : @luna_tile_ir.LunaTileCudaAotDigest
  priv numerical_policy : PagedAttentionParallelNumericalPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : PagedAttentionParallelDigest
  priv recipe : Bytes
  priv recipe_digest : PagedAttentionParallelDigest
  priv candidate_digest : PagedAttentionParallelDigest
}

///|
pub fn PagedAttentionParallelCudaCandidate::reference_fallback(
  self : PagedAttentionParallelCudaCandidate,
) -> @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering {
  self.reference
}

///|
pub fn PagedAttentionParallelCudaCandidate::target(
  self : PagedAttentionParallelCudaCandidate,
) -> @catalog.DeviceTarget {
  self.reference.target()
}

///|
pub fn PagedAttentionParallelCudaCandidate::profile(
  self : PagedAttentionParallelCudaCandidate,
) -> @launch_contract.PagedKernelProfile {
  self.reference.profile()
}

///|
pub fn PagedAttentionParallelCudaCandidate::query_heads(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.reference.query_heads()
}

///|
pub fn PagedAttentionParallelCudaCandidate::kv_heads(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.reference.kv_heads()
}

///|
pub fn PagedAttentionParallelCudaCandidate::head_dimension(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.reference.head_dimension()
}

///|
pub fn PagedAttentionParallelCudaCandidate::tokens_per_page(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.reference.tokens_per_page()
}

///|
pub fn PagedAttentionParallelCudaCandidate::entry_point_id(
  self : PagedAttentionParallelCudaCandidate,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn PagedAttentionParallelCudaCandidate::launch_dimensions(
  self : PagedAttentionParallelCudaCandidate,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn PagedAttentionParallelCudaCandidate::maximum_context_tokens(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.maximum_context_tokens
}

///|
pub fn PagedAttentionParallelCudaCandidate::shared_memory_bytes(
  self : PagedAttentionParallelCudaCandidate,
) -> Int {
  self.shared_memory_bytes
}

///|
pub fn PagedAttentionParallelCudaCandidate::tile_program_digest(
  self : PagedAttentionParallelCudaCandidate,
) -> @luna_tile_ir.LunaTileProgramDigest {
  self.tile_program_digest
}

///|
pub fn PagedAttentionParallelCudaCandidate::tile_aot_digest(
  self : PagedAttentionParallelCudaCandidate,
) -> @luna_tile_ir.LunaTileCudaAotDigest {
  self.tile_aot_digest
}

///|
pub fn PagedAttentionParallelCudaCandidate::numerical_policy(
  self : PagedAttentionParallelCudaCandidate,
) -> PagedAttentionParallelNumericalPolicy {
  self.numerical_policy
}

///|
pub fn PagedAttentionParallelCudaCandidate::function_symbol(
  self : PagedAttentionParallelCudaCandidate,
) -> String {
  self.function_symbol
}

///|
pub fn PagedAttentionParallelCudaCandidate::source_bytes(
  self : PagedAttentionParallelCudaCandidate,
) -> Bytes {
  self.source
}

///|
pub fn PagedAttentionParallelCudaCandidate::source_digest(
  self : PagedAttentionParallelCudaCandidate,
) -> PagedAttentionParallelDigest {
  self.source_digest
}

///|
pub fn PagedAttentionParallelCudaCandidate::recipe_bytes(
  self : PagedAttentionParallelCudaCandidate,
) -> Bytes {
  self.recipe
}

///|
pub fn PagedAttentionParallelCudaCandidate::recipe_digest(
  self : PagedAttentionParallelCudaCandidate,
) -> PagedAttentionParallelDigest {
  self.recipe_digest
}

///|
pub fn PagedAttentionParallelCudaCandidate::candidate_digest(
  self : PagedAttentionParallelCudaCandidate,
) -> PagedAttentionParallelDigest {
  self.candidate_digest
}

///|
pub fn PagedAttentionParallelCudaCandidate::manifest_bindable(
  _self : PagedAttentionParallelCudaCandidate,
) -> Bool {
  false
}
