///|
pub(all) enum GlmHybridKdaDecayControlAotFamily {
  PointwiseSafeLowerBoundSigmoidF32
} derive(Debug, Eq)

///|
pub struct GlmHybridKdaDecayControlCudaCandidate {
  priv family : GlmHybridKdaDecayControlAotFamily
  priv numerical_contract : @decay_control_cuda_source.DecayControlNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv head_count : Int
  priv head_dimension : Int
  priv lower_bound : Double
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
