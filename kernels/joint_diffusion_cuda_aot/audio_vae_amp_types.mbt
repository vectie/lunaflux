///|
pub struct JointDiffusionAudioVaeAmpActivationCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeAudioDecoderAmpActivationDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv stage_index : Int
  priv batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv upsample_time : Int
  priv element_count : Int64
  priv intermediate_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv stages : ReadOnlyArray[JointDiffusionAudioVaeFirstAmpCudaStage]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}

///|
pub struct JointDiffusionAudioVaeAmpStageCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv stage_index : Int
  priv parameter_layout_digest : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayoutDigest
  priv workspace_layout_digest : @joint_audio_amp_workspace_layout.JointAudioAmpWorkspaceLayoutDigest
  priv batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv element_count : Int64
  priv target : @catalog.DeviceTarget
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv stages : ReadOnlyArray[JointDiffusionAudioVaeFirstAmpExecutionStage]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}
