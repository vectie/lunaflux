///|
pub(all) enum JointVaeDecodeGap {
  VideoOutputChannels
  VideoDecoderDTypeCast
  VideoPostQuantConvolution
  VideoSpatialTiling
  VideoTemporalChunking
  VideoTransformerDecode
  VideoOverlapBlending
  VideoPixelPostprocessing
  VideoNormalization
  VideoActivation
  VideoConvolutionKernel
  VideoPadding
  VideoChannelSchedule
  VideoTemporalUpsample
  VideoSpatialUpsample
  VideoResidualPath
  VideoFinalProjection
  AudioNormalization
  AudioActivation
  AudioConvolutionKernel
  AudioPadding
  AudioChannelSchedule
  AudioTemporalUpsample
  AudioResidualPath
  AudioFinalProjection
} derive(Debug, Eq)

///|
pub(all) suberror JointVaeDecodeReferenceError {
  InvalidLimit
  InvalidPlan
  InvalidRequest
  GeometryOverflow
  GeometryBudgetExceeded(@joint_diffusion_plan.JointDiffusionComponent)
  InvalidLatentElementCount(@joint_diffusion_plan.JointDiffusionComponent)
  NonFiniteLatent(@joint_diffusion_plan.JointDiffusionComponent, Int)
  LatentIdentityMismatch(@joint_diffusion_plan.JointDiffusionComponent)
  InvalidDestandardizationCoefficients
  InvalidDestandardizationInput
  InvalidProjectionTensorIdentity
  InvalidProjectionInput
  DecodeSemanticUnavailable(
    @joint_diffusion_plan.JointDiffusionComponent,
    JointVaeDecodeGap
  )
} derive(Debug, Eq)

///|
pub struct JointVaeDecodeReferenceLimits {
  priv maximum_video_latent_elements : Int64
  priv maximum_video_output_sites : Int64
  priv maximum_audio_latent_elements : Int64
  priv maximum_audio_output_elements : Int64
}

///|
pub struct JointVaeDecodeGeometry {
  priv plan_digest : @spec.PlanDigest
  priv video_latent_channels : Int
  priv video_latent_frames : Int
  priv video_latent_height : Int
  priv video_latent_width : Int
  priv video_latent_elements : Int64
  priv video_output_frames : Int
  priv video_output_height : Int
  priv video_output_width : Int
  priv video_output_sites : Int64
  priv video_spatial_compression : Int
  priv audio_latent_channels : Int
  priv audio_latent_frames : Int
  priv audio_latent_elements : Int64
  priv audio_output_samples : Int64
  priv audio_output_channels : Int
  priv audio_output_elements : Int64
}

///|
pub struct JointVaeDecodeGapDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeDecodeAssessment {
  priv geometry : JointVaeDecodeGeometry
  priv gap_digest : JointVaeDecodeGapDigest
  priv video_gaps : ReadOnlyArray[JointVaeDecodeGap]
  priv audio_gaps : ReadOnlyArray[JointVaeDecodeGap]
}

///|
pub struct JointVaeLatentBuffer {
  priv component : @joint_diffusion_plan.JointDiffusionComponent
  priv plan_digest : @spec.PlanDigest
  priv values : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
pub struct JointVaeLatentDestandardizationDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Request-bound F32 inverse-normalization contract. Coefficients are ordered
/// by channel and memory is contiguous batch/channel/frame/height/width.
pub struct JointVaeLatentDestandardization {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionVaeLatentDestandardizationStorage
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionVideoVaeLatentLayout
  priv batch_count : Int
  priv channel_count : Int
  priv elements_per_channel : Int64
  priv element_count : Int64
  priv means : ReadOnlyArray[Float]
  priv standard_deviations : ReadOnlyArray[Float]
  priv digest : JointVaeLatentDestandardizationDigest
}

///|
/// Request-bound F32 audio inverse-normalization contract. Storage is
/// contiguous batch/channel/time and coefficient ordering is by channel.
pub struct JointVaeAudioLatentDestandardization {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionVaeLatentDestandardizationStorage
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv batch_count : Int
  priv channel_count : Int
  priv elements_per_channel : Int64
  priv element_count : Int64
  priv means : ReadOnlyArray[Float]
  priv standard_deviations : ReadOnlyArray[Float]
  priv digest : JointVaeLatentDestandardizationDigest
}

///|
pub struct JointVaeAudioDecoderInputProjectionDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Request-bound F32 pointwise decoder-input projection contract. Tensor
/// names are retained because the family adapter authenticates their exact
/// checkpoint roles before this family-neutral contract is constructed.
pub struct JointVaeAudioDecoderInputProjection {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv kernel_size : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeInputProjectionStorage
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv weight_name : String
  priv bias_name : String
  priv input_element_count : Int64
  priv weight_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv digest : JointVaeAudioDecoderInputProjectionDigest
}

///|
pub struct JointVaeAudioDecoderConvPreDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeAudioDecoderConvPre {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv kernel_size : Int
  priv padding : Int
  priv stride : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvPreStorage
  priv axes : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv weight_g_name : String
  priv weight_v_name : String
  priv bias_name : String
  priv input_element_count : Int64
  priv weight_g_element_count : Int64
  priv weight_v_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv digest : JointVaeAudioDecoderConvPreDigest
}

///|
pub struct JointVaeAudioDecoderFirstUpsampleDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeAudioDecoderFirstUpsample {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv input_time : Int
  priv output_time : Int
  priv kernel_size : Int
  priv stride : Int
  priv padding : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvTransposeStorage
  priv axes : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv weight_g_name : String
  priv weight_v_name : String
  priv bias_name : String
  priv input_element_count : Int64
  priv weight_g_element_count : Int64
  priv weight_v_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv digest : JointVaeAudioDecoderFirstUpsampleDigest
}

///|
pub struct JointVaeAudioDecoderUpsampleTransitionDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeAudioDecoderUpsampleTransition {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv stage_index : Int
  priv decoder_batch_count : Int
  priv input_channels : Int
  priv output_channels : Int
  priv input_time : Int
  priv output_time : Int
  priv kernel_size : Int
  priv stride : Int
  priv padding : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeConvTransposeStorage
  priv axes : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeWeightNormAxes
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv weight_g_name : String
  priv weight_v_name : String
  priv bias_name : String
  priv input_element_count : Int64
  priv weight_g_element_count : Int64
  priv weight_v_element_count : Int64
  priv bias_element_count : Int64
  priv output_element_count : Int64
  priv digest : JointVaeAudioDecoderUpsampleTransitionDigest
}

///|
pub struct JointVaeAudioDecoderFirstAmpActivationDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeAudioDecoderFirstAmpActivation {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv decoder_batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv upsample_time : Int
  priv ratio : Int
  priv kernel_size : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeAliasFreeActivationStorage
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv alpha_name : String
  priv beta_name : String
  priv upsample_filter_name : String
  priv downsample_filter_name : String
  priv element_count : Int64
  priv intermediate_element_count : Int64
  priv digest : JointVaeAudioDecoderFirstAmpActivationDigest
}

///|
pub struct JointVaeAudioDecoderAmpActivationDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct JointVaeAudioDecoderAmpActivation {
  priv plan_digest : @spec.PlanDigest
  priv request_shape : @joint_diffusion_plan.JointDiffusionRequestShape
  priv stage_index : Int
  priv decoder_batch_count : Int
  priv channel_count : Int
  priv time_count : Int
  priv upsample_time : Int
  priv ratio : Int
  priv kernel_size : Int
  priv storage : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeAliasFreeActivationStorage
  priv layout : @joint_diffusion_kernel_requirements.JointDiffusionAudioVaeLatentLayout
  priv alpha_name : String
  priv beta_name : String
  priv upsample_filter_name : String
  priv downsample_filter_name : String
  priv element_count : Int64
  priv intermediate_element_count : Int64
  priv digest : JointVaeAudioDecoderAmpActivationDigest
}
