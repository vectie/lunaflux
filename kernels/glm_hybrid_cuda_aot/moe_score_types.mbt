///|
pub(all) enum GlmHybridMoeScoreAotFamily {
  FullSigmoidCorrection256
  FlashSigmoidCorrection288
} derive(Debug, Eq)

///|
pub struct GlmHybridMoeScoreCudaCandidate {
  priv family : GlmHybridMoeScoreAotFamily
  priv numerical_contract : @sigmoid_correction_cuda_source.SigmoidCorrectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv expert_count : Int
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
