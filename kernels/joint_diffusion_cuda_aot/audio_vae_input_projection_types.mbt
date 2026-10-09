///|
/// Offline-only request-bound F32 AudioVAE `dec_in_proj` candidate. It ends
/// before the weight-normalized BigVGAN `decoder.conv_pre` operation.
pub struct JointDiffusionAudioVaeDecoderInputProjectionCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeAudioDecoderInputProjectionDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv time_count : Int
  priv input_element_count : Int64
  priv weight_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}
