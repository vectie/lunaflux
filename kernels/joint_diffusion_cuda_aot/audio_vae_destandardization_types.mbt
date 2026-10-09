///|
/// Offline-only request-bound inverse audio-latent normalization. The batch
/// axis is the decoder batch (stereo is represented as batch 2), not a model
/// request-batch claim. This prefix ends before the decoder dtype cast.
pub struct JointDiffusionAudioVaeLatentDestandardizationCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeLatentDestandardizationDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv channel_count : Int
  priv elements_per_channel : Int64
  priv element_count : Int64
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
