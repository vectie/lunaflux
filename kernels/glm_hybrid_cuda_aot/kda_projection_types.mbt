///|
pub(all) enum GlmHybridKdaProjectionAotFamily {
  SerialQkvForgetAndInputGateBf16
} derive(Debug, Eq)

///|
/// Non-bindable serial correctness candidate for the BF16-side KDA input and
/// control projection stage. F32 decay control is a separate requirement.
pub struct GlmHybridKdaProjectionCudaCandidate {
  priv family : GlmHybridKdaProjectionAotFamily
  priv numerical_contract : @recurrent_attention_projection_cuda_source.RecurrentAttentionProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv hidden_width : Int
  priv head_count : Int
  priv head_dimension : Int
  priv control_rank : Int
  priv maximum_tokens : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}
