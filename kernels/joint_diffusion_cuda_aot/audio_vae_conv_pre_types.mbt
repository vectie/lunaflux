///|
pub(all) enum JointDiffusionAudioVaeConvPreStageKind {
  NormalizeWeight
  Convolve
} derive(Debug, Eq)

///|
pub struct JointDiffusionAudioVaeConvPreCudaStage {
  priv kind : JointDiffusionAudioVaeConvPreStageKind
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
}

///|
/// Offline-only two-stage candidate. `NormalizeWeight` must execute before
/// `Convolve` and materializes the declared F32 normalized-weight workspace.
/// Artifact admission authenticates ABI/geometry, not tensor values; finite
/// `weight_g` and finite, nonzero per-output `weight_v` norms are preconditions.
pub struct JointDiffusionAudioVaeConvPreCudaCandidate {
  priv numerical_contract : JointDiffusionCudaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation
  priv capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability
  priv parameter_digest : @joint_diffusion_vae_reference.JointVaeAudioDecoderConvPreDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv time_count : Int
  priv kernel_size : Int
  priv padding : Int
  priv stride : Int
  priv input_element_count : Int64
  priv weight_g_element_count : Int64
  priv weight_v_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv target : @catalog.DeviceTarget
  priv operands : ReadOnlyArray[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ]
  priv stages : ReadOnlyArray[JointDiffusionAudioVaeConvPreCudaStage]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv source : Bytes
  priv source_digest : JointDiffusionCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : JointDiffusionCudaRecipeDigest
}
