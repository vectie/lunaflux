///|
pub(all) enum JointDiffusionAudioVaeFirstAmpStageKind {
  Upsample
  SnakeBeta
  Downsample
} derive(Debug, Eq)

///|
pub struct JointDiffusionAudioVaeFirstAmpCudaStage {
  priv kind : JointDiffusionAudioVaeFirstAmpStageKind
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
}

///|
pub struct JointDiffusionAudioVaeFirstAmpCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstAmpActivationDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
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
