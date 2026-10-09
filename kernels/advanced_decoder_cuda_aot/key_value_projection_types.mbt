///|
pub(all) enum AdvancedDecoderKeyValueProjectionFamily {
  ReplicatedKeyValueBlockFp8E4M3Ue8M0ProjectionV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderKeyValueProjectionNumericalContract {
  KeyValuePerRowBlock128Ue8M0OrderedF32Bf16RoundV1
} derive(Debug, Eq)

///|
/// Inert pre-normalization shared key/value projection source candidate.
pub struct AdvancedDecoderKeyValueProjectionCudaCandidate {
  priv family : AdvancedDecoderKeyValueProjectionFamily
  priv numerical_contract : AdvancedDecoderKeyValueProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv input_width : Int
  priv output_width : Int
  priv rope_width : Int
  priv block_width : Int
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
