///|
pub(all) enum GlmHybridMoeRouterAotFamily {
  FullFloat32Router256
  FlashFloat32Router288
} derive(Debug, Eq)

///|
pub struct GlmHybridMoeRouterCudaCandidate {
  priv family : GlmHybridMoeRouterAotFamily
  priv numerical_contract : @router_projection_cuda_source.RouterProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int32
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv hidden_width : Int32
  priv expert_count : Int32
  priv maximum_tokens : Int32
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
