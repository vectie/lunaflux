///|
fn official_query_a_shape(input_width : Int, output_width : Int) -> Bool {
  (input_width == 6144 && output_width == 2048) ||
  (input_width == 4096 && output_width == 1536)
}

///|
fn official_key_value_a_shape(
  input_width : Int,
  compressed_width : Int,
  rope_width : Int,
) -> Bool {
  (input_width == 6144 && compressed_width == 512 && rope_width == 64) ||
  (input_width == 4096 && compressed_width == 512 && rope_width == 0)
}

///|
/// Exact first query low-rank projection: hidden BF16 multiplied by the
/// row-major [query-rank, hidden] q_a weight, with no bias.
pub fn lower_dsa_query_a_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (DsaQueryAProjection(input_width, output_width), DsaQueryAProjectionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if input_width != requirements.geometry().hidden_size() ||
    !official_query_a_shape(input_width, output_width) {
    raise Invalid(Shape)
  }
  lower_dense_projection_cuda_aot_candidate(
    DsaQueryAProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width~,
    output_role=DsaQueryCompressedOutput,
    function_symbol="lunaflux_glm_hybrid_dsa_query_a_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}

///|
/// Exact first KV projection. The output row is the official contiguous
/// [compressed-kv, rope-key] layout; splitting, RMSNorm, and kv_b are excluded.
pub fn lower_dsa_key_value_a_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DsaKeyValueAProjection(input_width, compressed_width, rope_width),
      DsaKeyValueAProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if input_width != requirements.geometry().hidden_size() ||
    !official_key_value_a_shape(input_width, compressed_width, rope_width) {
    raise Invalid(Shape)
  }
  let output_width = compressed_width + rope_width
  lower_dense_projection_cuda_aot_candidate(
    DsaKeyValueAProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width~,
    output_role=DsaKeyValueCompressedRopeOutput,
    function_symbol="lunaflux_glm_hybrid_dsa_key_value_a_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}

///|
/// Exact q_b projection over normalized q-A rows. The flat output row has the
/// zero-copy view `[query_head, qk_component]`; Full RoPE is downstream.
pub fn lower_dsa_query_b_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DsaQueryBProjection(input_width, head_count, head_dimension),
      DsaQueryBProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if head_count != 64 ||
    head_dimension != 256 ||
    !((requirements.geometry().hidden_size() == 6144 && input_width == 2048) ||
    (requirements.geometry().hidden_size() == 4096 && input_width == 1536)) {
    raise Invalid(Shape)
  }
  let output_width = checked_mul(
    head_count.to_int64(),
    head_dimension.to_int64(),
  )
  if output_width > 2147483647L {
    raise SizeOverflow
  }
  lower_dense_projection_cuda_aot_candidate(
    DsaQueryBProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width=output_width.to_int(),
    input_role=DsaQueryNormalizedInput,
    output_role=DsaQueryHeadsOutput,
    function_symbol="lunaflux_glm_hybrid_dsa_query_b_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}

///|
/// Exact kv_b projection over normalized compressed KV. Each output row has
/// the zero-copy view `[kv_head, key_nope_component then value_component]`.
/// The separate KV-A RoPE-key suffix is deliberately not consumed here.
pub fn lower_dsa_key_value_b_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      KeyValueLowRankProjection(
        input_width,
        head_count,
        key_nope_width,
        value_width
      ),
      Advanced(LowRankAttentionProjectionAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if input_width != 512 ||
    head_count != 64 ||
    value_width != 256 ||
    !((requirements.geometry().hidden_size() == 6144 && key_nope_width == 192) ||
    (requirements.geometry().hidden_size() == 4096 && key_nope_width == 256)) {
    raise Invalid(Shape)
  }
  let per_head_width = key_nope_width + value_width
  let output_width = checked_mul(
    head_count.to_int64(),
    per_head_width.to_int64(),
  )
  if output_width > 2147483647L {
    raise SizeOverflow
  }
  lower_dense_projection_cuda_aot_candidate(
    DsaKeyValueBProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width=output_width.to_int(),
    input_role=DsaKeyValueNormalizedInput,
    output_role=DsaKeyValueHeadsOutput,
    function_symbol="lunaflux_glm_hybrid_dsa_key_value_b_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}
