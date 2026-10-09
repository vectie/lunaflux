///|
pub(all) enum GlmHybridMoeFeedForwardAotFamily {
  FullRoutedExperts256
  FlashRoutedExperts288
  FullSharedExpert
  FlashSharedExpert
  FullRoutedSharedAdd
  FlashRoutedSharedAdd
  FullDecoderResidualAdd
  FlashHyperConnectionMix4
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridMoeFeedForwardNumericalContract {
  SelectedExpertBf16WeightedAccumulate
  SharedExpertBf16SwiGlu
  RoutedSharedF32AddBf16Round
  DecoderResidualF32AddBf16Round
  HyperControlBf16OrderedResidualMix
} derive(Debug, Eq)

///|
pub struct GlmHybridMoeFeedForwardCudaCandidate {
  priv family : GlmHybridMoeFeedForwardAotFamily
  priv numerical_contract : GlmHybridMoeFeedForwardNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
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
