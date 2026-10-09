///|
pub(all) enum AdvancedDecoderCompressedKvPrepareAotFamily {
  NonOverlappingSharedKeyValuePrepareBf16V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderCompressedKvPrepareNumericalContract {
  BitExactBf16SegmentCopyV1
} derive(Debug, Eq)

///|
/// Offline-only source candidate that materializes borrowed ordinary and
/// compressed BF16 K/V segments into the exact shared-attention layout. It
/// owns no compressor projection/state, live cache, compiled module, or launch.
pub struct AdvancedDecoderCompressedKvPrepareCudaCandidate {
  priv family : AdvancedDecoderCompressedKvPrepareAotFamily
  priv numerical_contract : AdvancedDecoderCompressedKvPrepareNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_batches : Int
  priv window_size : Int
  priv compression_ratio : Int
  priv head_dimension : Int
  priv maximum_position_count : Int
  priv maximum_compressed_slots : Int
  priv maximum_key_slots : Int
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
