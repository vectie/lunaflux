///|
fn checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise JointDiffusionCudaAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow
  }
  left * right
}

///|
fn validate_request_shape(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Unit raise JointDiffusionCudaAotError {
  let resolved = requirements
    .shape_contract()
    .resolve_request(
      height=request.height(),
      width=request.width(),
      requested_frames=request.frames(),
      inference_steps=request.inference_steps(),
    ) catch {
      error => raise Request(error)
    }
  if resolved != request ||
    request.model_evaluations() != request.inference_steps() {
    raise Invalid(RequestShape)
  }
}

///|
fn element_count(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int64 raise JointDiffusionCudaAotError {
  let shapes = requirements.shape_contract()
  match latent {
    VideoLatent =>
      checked_mul(
        checked_mul(
          checked_mul(
            shapes.video_latent_channels().to_int64(),
            request.video_latent_frames().to_int64(),
          ),
          request.video_latent_height().to_int64(),
        ),
        request.video_latent_width().to_int64(),
      )
    AudioLatent =>
      checked_mul(
        shapes.audio_latent_channels().to_int64(),
        request.audio_token_rows().to_int64(),
      )
  }
}

///|
fn launch_operand(
  role : @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand raise JointDiffusionCudaAotError {
  @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand::new(
    role~,
    byte_count~,
    alignment~,
  ) catch {
    error => raise Operand(error)
  }
}

///|
fn role_tag(
  role : @joint_diffusion_artifact_admission.JointDiffusionLaunchOperandRole,
) -> String {
  match role {
    TokenIdsInput => "token-ids"
    TextConditioningOutput => "text-conditioning-output"
    KeyframeMediaInput => "keyframe-media-input"
    OrderedReferenceMediaInput => "ordered-reference-media-input"
    OrderedConditioningOutput => "ordered-conditioning-output"
    VideoLatentInput => "video-latent-input"
    AudioLatentInput => "audio-latent-input"
    VideoPredictionInput => "video-prediction-input"
    AudioPredictionInput => "audio-prediction-input"
    FlowCoefficientsInput => "flow-f32-coefficients-input"
    JointHiddenInput => "joint-hidden-input"
    JointHiddenOutput => "joint-hidden-output"
    ModelWeights => "model-weights"
    VideoLatentOutput => "video-latent-output"
    AudioLatentOutput => "audio-latent-output"
    VideoMediaOutput => "video-media-output"
    AudioMediaOutput => "audio-media-output"
    Workspace => "workspace"
    ProjectionWeight => "projection-weight"
    ProjectionBias => "projection-bias"
    VideoHiddenInput => "video-hidden-input"
    AudioHiddenInput => "audio-hidden-input"
    VideoPredictionOutput => "video-prediction-output"
    AudioPredictionOutput => "audio-prediction-output"
    VideoFinalNormInput => "video-final-norm-input"
    AudioFinalNormInput => "audio-final-norm-input"
    FinalNormWeight => "final-norm-weight"
    VideoFinalNormOutput => "video-final-norm-output"
    AudioFinalNormOutput => "audio-final-norm-output"
    FinalAdaLnShiftInput => "final-adaln-shift-input"
    FinalAdaLnScaleInput => "final-adaln-scale-input"
    VideoF32HeadInput => "video-f32-head-input"
    AudioF32HeadInput => "audio-f32-head-input"
    FinalAdaLnConditioningInput => "final-adaln-conditioning-input"
    FinalAdaLnWeight => "final-adaln-weight"
    FinalAdaLnBias => "final-adaln-bias"
    FinalAdaLnShiftOutput => "final-adaln-shift-output"
    FinalAdaLnScaleOutput => "final-adaln-scale-output"
    PackedAdaLnShiftInput => "packed-adaln-shift-input"
    PackedAdaLnScaleInput => "packed-adaln-scale-input"
    VideoPackedRowIndices => "video-packed-row-indices"
    AudioPackedRowIndices => "audio-packed-row-indices"
    VideoAdaLnShiftOutput => "video-adaln-shift-output"
    VideoAdaLnScaleOutput => "video-adaln-scale-output"
    AudioAdaLnShiftOutput => "audio-adaln-shift-output"
    AudioAdaLnScaleOutput => "audio-adaln-scale-output"
    DistinctTimestepValuesInput => "distinct-timestep-values-input"
    TimestepFrequencyRowsOutput => "timestep-frequency-rows-output"
    TimestepMlpWeight1 => "timestep-mlp-weight-1"
    TimestepMlpBias1 => "timestep-mlp-bias-1"
    TimestepMlpWeight2 => "timestep-mlp-weight-2"
    TimestepMlpBias2 => "timestep-mlp-bias-2"
    TimestepEmbeddingRowsOutput => "timestep-embedding-rows-output"
    TransformerMlpPackedGateUpWeight => "transformer-mlp-packed-gate-up-weight"
    TransformerMlpDownWeight => "transformer-mlp-down-weight"
    AttentionQueryWeight => "attention-query-weight"
    AttentionKeyWeight => "attention-key-weight"
    AttentionValueWeight => "attention-value-weight"
    AttentionOutputWeight => "attention-output-weight"
    AttentionQueryNormWeight => "attention-query-norm-weight"
    AttentionKeyNormWeight => "attention-key-norm-weight"
    AttentionRotaryCosInput => "attention-rotary-cos-input"
    AttentionRotarySinInput => "attention-rotary-sin-input"
    AttentionQueryBuffer => "attention-query-buffer"
    AttentionKeyBuffer => "attention-key-buffer"
    AttentionValueBuffer => "attention-value-buffer"
    AttentionNormalizedQueryBuffer => "attention-normalized-query-buffer"
    AttentionNormalizedKeyBuffer => "attention-normalized-key-buffer"
    AttentionOutputBuffer => "attention-output-buffer"
    VideoNormalizedLatentInput => "video-normalized-latent-input"
    VideoLatentStandardDeviationInput => "video-latent-std-input"
    VideoLatentMeanInput => "video-latent-mean-input"
    VideoDestandardizedLatentOutput => "video-destandardized-latent-output"
    AudioNormalizedLatentInput => "audio-normalized-latent-input"
    AudioLatentStandardDeviationInput => "audio-latent-standard-deviation-input"
    AudioLatentMeanInput => "audio-latent-mean-input"
    AudioDestandardizedLatentOutput => "audio-destandardized-latent-output"
    AudioDecoderInputProjectionWeight => "audio-decoder-input-projection-weight"
    AudioDecoderInputProjectionBias => "audio-decoder-input-projection-bias"
    AudioDecoderInputProjectionOutput => "audio-decoder-input-projection-output"
    AudioDecoderConvPreWeightG => "audio-decoder-conv-pre-weight-g"
    AudioDecoderConvPreWeightV => "audio-decoder-conv-pre-weight-v"
    AudioDecoderConvPreBias => "audio-decoder-conv-pre-bias"
    AudioDecoderConvPreNormalizedWeightWorkspace =>
      "audio-decoder-conv-pre-normalized-weight-workspace"
    AudioDecoderConvPreOutput => "audio-decoder-conv-pre-output"
    AudioDecoderFirstUpsampleWeightG => "audio-decoder-first-upsample-weight-g"
    AudioDecoderFirstUpsampleWeightV => "audio-decoder-first-upsample-weight-v"
    AudioDecoderFirstUpsampleBias => "audio-decoder-first-upsample-bias"
    AudioDecoderFirstUpsampleNormalizedWeightWorkspace =>
      "audio-decoder-first-upsample-normalized-weight-workspace"
    AudioDecoderFirstUpsampleOutput => "audio-decoder-first-upsample-output"
    AudioDecoderFirstAmpAlpha => "audio-decoder-first-amp-alpha"
    AudioDecoderFirstAmpBeta => "audio-decoder-first-amp-beta"
    AudioDecoderFirstAmpUpsampleFilter =>
      "audio-decoder-first-amp-upsample-filter"
    AudioDecoderFirstAmpDownsampleFilter =>
      "audio-decoder-first-amp-downsample-filter"
    AudioDecoderFirstAmpUpsampleWorkspace =>
      "audio-decoder-first-amp-upsample-workspace"
    AudioDecoderFirstAmpActivatedWorkspace =>
      "audio-decoder-first-amp-activated-workspace"
    AudioDecoderFirstAmpActivationOutput =>
      "audio-decoder-first-amp-activation-output"
    AudioDecoderFirstAmpStageOutput => "audio-decoder-first-amp-stage-output"
    AudioDecoderUpsampleTransitionWeightG =>
      "audio-decoder-upsample-transition-weight-g"
    AudioDecoderUpsampleTransitionWeightV =>
      "audio-decoder-upsample-transition-weight-v"
    AudioDecoderUpsampleTransitionBias =>
      "audio-decoder-upsample-transition-bias"
    AudioDecoderUpsampleTransitionNormalizedWeightWorkspace =>
      "audio-decoder-upsample-transition-normalized-weight-workspace"
    AudioDecoderUpsampleTransitionOutput =>
      "audio-decoder-upsample-transition-output"
    AudioDecoderAmpAlpha => "audio-decoder-amp-alpha"
    AudioDecoderAmpBeta => "audio-decoder-amp-beta"
    AudioDecoderAmpUpsampleFilter => "audio-decoder-amp-upsample-filter"
    AudioDecoderAmpDownsampleFilter => "audio-decoder-amp-downsample-filter"
    AudioDecoderAmpUpsampleWorkspace => "audio-decoder-amp-upsample-workspace"
    AudioDecoderAmpActivatedWorkspace => "audio-decoder-amp-activated-workspace"
    AudioDecoderAmpActivationOutput => "audio-decoder-amp-activation-output"
    AudioDecoderAmpStageOutput => "audio-decoder-amp-stage-output"
    AudioDecoderUpsampleTransition2WeightG =>
      "audio-decoder-upsample-transition-2-weight-g"
    AudioDecoderUpsampleTransition2WeightV =>
      "audio-decoder-upsample-transition-2-weight-v"
    AudioDecoderUpsampleTransition2Bias =>
      "audio-decoder-upsample-transition-2-bias"
    AudioDecoderUpsampleTransition2NormalizedWeightWorkspace =>
      "audio-decoder-upsample-transition-2-normalized-weight-workspace"
    AudioDecoderUpsampleTransition2Output =>
      "audio-decoder-upsample-transition-2-output"
    AudioDecoderAmpActivation2Output => "audio-decoder-amp-activation-2-output"
  }
}
