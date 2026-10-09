///|
pub(all) enum GlmHybridMoeGroupedRoutingAotFamily {
  FullGroupedRouting256
  FlashGroupedRouting288
} derive(Debug, Eq)

///|
pub struct GlmHybridMoeGroupedRoutingCudaCandidate {
  priv family : GlmHybridMoeGroupedRoutingAotFamily
  priv numerical_contract : @grouped_routing_cuda_source.GroupedRoutingNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv routed_expert_count : Int
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
