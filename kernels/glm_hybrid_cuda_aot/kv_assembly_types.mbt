///|
pub(all) enum GlmHybridKvAssemblyAotFamily {
  FullSharedRopeSuffixBroadcast
  FlashNoRopeSplit
} derive(Debug, Eq)

///|
pub struct GlmHybridKvAssemblyCudaCandidate {
  priv family : GlmHybridKvAssemblyAotFamily
  priv numerical_contract : @decoder_kv_assembly_cuda_source.DecoderKvAssemblyNumericalContract
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
