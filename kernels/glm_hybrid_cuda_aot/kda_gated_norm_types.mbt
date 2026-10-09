///|
pub(all) enum GlmHybridKdaGatedNormAotFamily {
  SerialPerHeadSigmoidGatedRmsNormBf16
} derive(Debug, Eq)

///|
pub struct GlmHybridKdaGatedNormCudaCandidate {
  priv family : GlmHybridKdaGatedNormAotFamily
  priv numerical_contract : @gated_rms_norm_cuda_source.GatedRmsNormNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv head_count : Int
  priv head_dimension : Int
  priv epsilon : Double
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
