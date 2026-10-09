///|
pub(all) enum AdvancedDecoderInverseRotaryAotFamily {
  AttentionOutputInverseScaledYarnBf16V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderInverseRotaryNumericalContract {
  PrefixBitExactAdjacentPairInverseF32YarnBf16RoundV1
} derive(Debug, Eq)

///|
/// Offline-only source candidate for full-head attention-output inverse RoPE.
/// It carries no cache, compiled module, loader, launch, or physical authority.
pub struct AdvancedDecoderInverseRotaryCudaCandidate {
  priv family : AdvancedDecoderInverseRotaryAotFamily
  priv numerical_contract : AdvancedDecoderInverseRotaryNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv head_count : Int
  priv head_dimension : Int
  priv rope_dimension : Int
  priv maximum_positions : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}
