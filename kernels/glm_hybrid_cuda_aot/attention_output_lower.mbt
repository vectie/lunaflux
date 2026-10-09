///|
/// Lowers the terminal projection of GLM dynamic sparse attention through the
/// shared family-neutral BF16 dense-projection renderer. The counts operand is
/// part of the exact renderer ABI and bounds the active row prefix.
pub fn lower_attention_output_projection_cuda_aot_candidate(
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
      AttentionOutputProjection(output_width, input_width),
      AttentionOutputProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if output_width != requirements.geometry().hidden_size() {
    raise Invalid(Shape)
  }
  lower_dense_projection_cuda_aot_candidate(
    AttentionOutputProjectionBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width~,
    output_role=HiddenOutput,
    function_symbol="lunaflux_glm_hybrid_attention_output_projection_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}
