///|
pub fn GlmHybridSparseAttentionCudaCandidate::family(
  self : Self,
) -> GlmHybridSparseAttentionAotFamily {
  self.family
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::numerical_contract(
  self : Self,
) -> @glm_hybrid_sparse_attention_cuda_source.GlmHybridSparseAttentionNumericalContract {
  self.numerical_contract
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::requirements_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::query_head_count(
  self : Self,
) -> Int {
  self.query_head_count
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::key_value_head_count(
  self : Self,
) -> Int {
  self.key_value_head_count
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::query_key_dimension(
  self : Self,
) -> Int {
  self.query_key_dimension
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::value_dimension(
  self : Self,
) -> Int {
  self.value_dimension
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::top_k(self : Self) -> Int {
  self.top_k
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::selected_width(
  self : Self,
) -> Int {
  self.selected_width
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::maximum_tokens(
  self : Self,
) -> Int {
  self.maximum_tokens
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::operands(
  self : Self,
) -> ReadOnlyArray[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::source_digest(
  self : Self,
) -> GlmHybridCudaSourceDigest {
  self.source_digest
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::recipe_bytes(
  self : Self,
) -> Bytes {
  self.recipe
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::recipe_digest(
  self : Self,
) -> GlmHybridCudaRecipeDigest {
  self.recipe_digest
}

///|
pub fn GlmHybridSparseAttentionCudaCandidate::manifest_bindable(
  _self : Self,
) -> Bool {
  false
}
