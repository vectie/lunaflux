///|
pub(all) enum AdvancedDecoderOutputAProjectionFamily {
  GroupedOutputAProjectionBf16V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderOutputAProjectionNumericalContract {
  ConvertedGroupedWeightOrderedF32Bf16RoundV1
} derive(Debug, Eq)

///|
/// Inert source candidate for the grouped first half of an output low-rank
/// projection. It proves no raw-checkpoint conversion, compilation, or launch.
pub struct AdvancedDecoderOutputAProjectionCudaCandidate {
  priv family : AdvancedDecoderOutputAProjectionFamily
  priv numerical_contract : AdvancedDecoderOutputAProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv group_count : Int
  priv input_width_per_group : Int
  priv output_rank : Int
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
