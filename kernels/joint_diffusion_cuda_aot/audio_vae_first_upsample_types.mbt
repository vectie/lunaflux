///|
pub(all) enum JointDiffusionAudioVaeFirstUpsampleStageKind {
  NormalizeWeight
  ConvolveTranspose
} derive(Debug, Eq)

///|
pub struct JointDiffusionAudioVaeFirstUpsampleCudaStage {
  priv kind : JointDiffusionAudioVaeFirstUpsampleStageKind
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
}

///|
/// Offline-only and non-bindable. NormalizeWeight must execute before
/// ConvolveTranspose. Artifact admission authenticates metadata, not the
/// finite/nonzero runtime value precondition for each input-channel norm.
pub struct JointDiffusionAudioVaeFirstUpsampleCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstUpsampleDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv input_time : Int
  priv output_time : Int
  priv kernel_size : Int
  priv stride : Int
  priv padding : Int
  priv input_element_count : Int64
  priv weight_g_element_count : Int64
  priv weight_v_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv stages : ReadOnlyArray[JointDiffusionAudioVaeFirstUpsampleCudaStage]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}
