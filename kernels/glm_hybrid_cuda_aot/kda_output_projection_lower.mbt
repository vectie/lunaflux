///|
/// Lowers the exact Flash KDA output projection. Input is the flattened
/// zero-copy `[token, head, dimension]` gated-norm output. Gated normalization,
/// residual addition, and cache ownership are deliberately excluded.
pub fn lower_kda_output_projection_cuda_aot_candidate(
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
    is (KdaOutputProjection(input_width, output_width), KdaOutputProjectionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if input_width != 8192 ||
    output_width != 4096 ||
    output_width != requirements.geometry().hidden_size() {
    raise Invalid(Shape)
  }
  lower_dense_projection_cuda_aot_candidate(
    KdaOutputProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width~,
    input_role=KdaOutputProjectionInput,
    weight_role=KdaOutputProjectionWeights,
    output_role=KdaOutputProjectionOutput,
    function_symbol="lunaflux_glm_hybrid_kda_output_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}
