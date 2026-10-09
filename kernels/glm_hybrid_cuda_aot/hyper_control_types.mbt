///|
pub(all) enum GlmHybridHyperControlAotFamily {
  FlashFunctionAndCollapse4x4096
  FlashPositiveSinkhorn4x20
} derive(Debug, Eq)

///|
pub struct GlmHybridHyperControlCudaCandidate {
  priv family : GlmHybridHyperControlAotFamily
  priv numerical_contract : @decoder_hyper_connection_cuda_source.DecoderHyperConnectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv stream_count : Int
  priv hidden_width : Int
  priv sinkhorn_iterations : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
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
