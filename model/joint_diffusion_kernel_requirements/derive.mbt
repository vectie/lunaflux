///|
fn add_unique_requirement(
  entries : Array[JointDiffusionKernelRequirement],
  operation : JointDiffusionSemanticOperation,
  capability : JointDiffusionAotCapability,
  limits : JointDiffusionKernelRequirementLimits,
) -> Unit raise JointDiffusionKernelRequirementsError {
  for index, entry in entries {
    if entry.operation == operation {
      if entry.capability == capability {
        return
      }
      raise DuplicateSemanticOperation(index)
    }
  }
  if entries.length() >= limits.maximum_requirements {
    raise PlanExceedsLimits
  }
  entries.push({ operation, capability, })
}

///|
fn ordered_capabilities(
  entries : ArrayView[JointDiffusionKernelRequirement],
) -> ReadOnlyArray[JointDiffusionAotCapability] {
  let capabilities : Array[JointDiffusionAotCapability] = []
  for entry in entries {
    if !capabilities.contains(entry.capability) {
      capabilities.push(entry.capability)
    }
  }
  ReadOnlyArray::from_array(capabilities)
}

///|
fn add_conditioning_requirements(
  entries : Array[JointDiffusionKernelRequirement],
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  limits : JointDiffusionKernelRequirementLimits,
) -> Unit raise JointDiffusionKernelRequirementsError {
  let denoiser = plan.denoiser()
  let conditioning = plan.conditioning()
  add_unique_requirement(
    entries,
    TokenTextConditioning(denoiser.text_conditioning_dimension()),
    TokenTextConditioningAot,
    limits,
  )
  if plan.workflows().contains(FirstLastFrameToAudioVideo) {
    add_unique_requirement(
      entries,
      OrderedKeyframeConditioning(
        conditioning.max_keyframes(),
        plan.shapes().video_latent_channels(),
      ),
      OrderedKeyframeConditioningAot,
      limits,
    )
  }
  if plan.workflows().contains(ReferenceToAudioVideo) {
    add_unique_requirement(
      entries,
      OrderedReferenceConditioning(
        conditioning.max_image_references(),
        conditioning.max_video_references(),
        conditioning.max_audio_references(),
        conditioning.max_total_references(),
        conditioning.allow_audio_only_references(),
      ),
      OrderedReferenceConditioningAot,
      limits,
    )
  }
}

///|
fn add_transformer_requirements(
  entries : Array[JointDiffusionKernelRequirement],
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  limits : JointDiffusionKernelRequirementLimits,
) -> Unit raise JointDiffusionKernelRequirementsError {
  let shape = plan.shapes()
  let denoiser = plan.denoiser()
  add_unique_requirement(
    entries,
    VideoLatentProjection(
      shape.video_latent_channels(),
      shape.video_patch_temporal(),
      shape.video_patch_height(),
      shape.video_patch_width(),
      denoiser.hidden_size(),
    ),
    VideoLatentProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioLatentProjection(shape.audio_latent_channels(), denoiser.hidden_size()),
    AudioLatentProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    JointTransformerAttention(
      denoiser.layer_count(),
      denoiser.refiner_layer_count(),
      denoiser.attention_head_count(),
      denoiser.attention_head_dimension(),
      denoiser.hidden_size(),
    ),
    JointTransformerAttentionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    JointTransformerMlp(
      denoiser.layer_count(),
      denoiser.refiner_layer_count(),
      denoiser.hidden_size(),
      denoiser.feed_forward_dimension(),
    ),
    JointTransformerMlpAot,
    limits,
  )
  add_unique_requirement(
    entries,
    VideoFinalRmsNorm(
      denoiser.hidden_size(),
      denoiser.final_norm_epsilon_numerator(),
      denoiser.final_norm_epsilon_denominator(),
    ),
    VideoFinalRmsNormAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioFinalRmsNorm(
      denoiser.hidden_size(),
      denoiser.final_norm_epsilon_numerator(),
      denoiser.final_norm_epsilon_denominator(),
    ),
    AudioFinalRmsNormAot,
    limits,
  )
  add_unique_requirement(
    entries,
    VideoFinalAdaLnF32Boundary(denoiser.hidden_size()),
    VideoFinalAdaLnF32BoundaryAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioFinalAdaLnF32Boundary(denoiser.hidden_size()),
    AudioFinalAdaLnF32BoundaryAot,
    limits,
  )
  add_unique_requirement(
    entries,
    FinalAdaLnParameterRow(
      denoiser.final_adaln_conditioning_dimension(),
      denoiser.hidden_size(),
    ),
    FinalAdaLnParameterRowAot,
    limits,
  )
  add_unique_requirement(
    entries,
    PackedModalityAdaLnRows(denoiser.hidden_size()),
    PackedModalityAdaLnRowsAot,
    limits,
  )
  add_unique_requirement(
    entries,
    TimestepFrequencyRows(denoiser.timestep_frequency_dimension()),
    TimestepFrequencyRowsAot,
    limits,
  )
  add_unique_requirement(
    entries,
    TimestepEmbeddingMlpRows(
      denoiser.timestep_frequency_dimension(),
      denoiser.timestep_embedding_hidden_dimension(),
      denoiser.final_adaln_conditioning_dimension(),
    ),
    TimestepEmbeddingMlpRowsAot,
    limits,
  )
  add_unique_requirement(
    entries,
    DistinctTimestepAdaLnParameterRows(
      denoiser.final_adaln_conditioning_dimension(),
      denoiser.hidden_size(),
    ),
    DistinctTimestepAdaLnParameterRowsAot,
    limits,
  )
  add_unique_requirement(
    entries,
    VideoOutputProjection(
      shape.video_latent_channels(),
      shape.video_patch_temporal(),
      shape.video_patch_height(),
      shape.video_patch_width(),
      denoiser.hidden_size(),
    ),
    VideoOutputProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioOutputProjection(shape.audio_latent_channels(), denoiser.hidden_size()),
    AudioOutputProjectionAot,
    limits,
  )
}

///|
fn add_scheduler_and_decode_requirements(
  entries : Array[JointDiffusionKernelRequirement],
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  limits : JointDiffusionKernelRequirementLimits,
) -> Unit raise JointDiffusionKernelRequirementsError {
  for scheduler in [plan.video_scheduler(), plan.audio_scheduler()] {
    add_unique_requirement(
      entries,
      ShiftedFlowStep(
        scheduler.latent(),
        scheduler.shift_numerator(),
        scheduler.shift_denominator(),
        scheduler.training_timestep_count(),
      ),
      ShiftedFlowStepAot,
      limits,
    )
  }
  let shape = plan.shapes()
  add_unique_requirement(
    entries,
    VideoVaeLatentDestandardization(
      shape.video_latent_channels(),
      Float32InputScaleMeanOutput,
      BatchChannelFrameHeightWidth,
    ),
    VideoVaeLatentDestandardizationAot,
    limits,
  )
  add_unique_requirement(
    entries,
    VideoVaeDecoderBody(
      shape.video_latent_channels(),
      shape.video_spatial_compression(),
      shape.frames_per_chunk(),
      shape.latents_per_chunk(),
    ),
    VideoVaeDecoderBodyAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeLatentDestandardization(
      shape.audio_latent_channels(),
      // The authenticated joint-audio layout flattens stereo into decoder batch;
      // this axis is not the request batch. The shape contract's exact stereo
      // cardinality is retained here as decoder-batch geometry.
      shape.audio_output_channels(),
      Float32InputScaleMeanOutput,
      BatchChannelTime,
    ),
    AudioVaeLatentDestandardizationAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderInputProjection(
      shape.audio_output_channels(),
      shape.audio_latent_channels(),
      shape.audio_vae_latent_dimension(),
      1,
      Float32InputWeightBiasOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderInputProjectionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderConvPre(
      shape.audio_output_channels(),
      shape.audio_vae_latent_dimension(),
      shape.audio_vae_decoder_dimension(),
      7,
      3,
      1,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerOutputChannelInputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderConvPreAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderFirstUpsample(
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension(),
      shape.audio_vae_decoder_dimension() / 2,
      9,
      5,
      2,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstUpsampleAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderFirstAmpActivation(
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 2,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstAmpActivationAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderFirstAmpStage(
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 2,
      3,
      3,
      Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderFirstAmpStageAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderUpsampleTransition(
      1,
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 2,
      shape.audio_vae_decoder_dimension() / 4,
      9,
      5,
      2,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderUpsampleTransitionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderAmpActivation(
      1,
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 4,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpActivationAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderAmpStage(
      1,
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 4,
      3,
      3,
      Kernels3_7_11Dilations1_3_5,
      Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpStageAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderUpsampleTransition(
      2,
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 4,
      shape.audio_vae_decoder_dimension() / 8,
      4,
      2,
      1,
      Float32InputWeightGWeightVBiasWorkspaceOutput,
      PerInputChannelOutputAndKernel,
      BatchChannelTime,
    ),
    AudioVaeDecoderUpsampleTransitionAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderAmpActivation(
      2,
      shape.audio_output_channels(),
      shape.audio_vae_decoder_dimension() / 8,
      2,
      12,
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
      BatchChannelTime,
    ),
    AudioVaeDecoderAmpActivationAot,
    limits,
  )
  add_unique_requirement(
    entries,
    AudioVaeDecoderBody(
      shape.audio_vae_decoder_dimension() / 8,
      shape.audio_latents_per_second(),
      shape.audio_sample_rate(),
      shape.audio_output_channels(),
    ),
    AudioVaeDecoderBodyAot,
    limits,
  )
}

///|
/// Derives a complete first-use-ordered semantic/AOT requirement set. This
/// performs bounded startup allocation and no artifact selection or execution.
pub fn derive_requirements(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  limits~ : JointDiffusionKernelRequirementLimits,
) -> JointDiffusionKernelRequirements raise JointDiffusionKernelRequirementsError {
  if plan.schema_version() != 1 {
    raise UnsupportedSchema
  }
  let denoiser = plan.denoiser()
  if denoiser.layer_count() >
    limits.maximum_transformer_layers - denoiser.refiner_layer_count() {
    raise PlanExceedsLimits
  }
  let source_identity = @spec.ModelIdentity::new(
    plan.content_digest(),
    plan.plan_digest(),
  ) catch {
    _ => raise InvalidSourceIdentity
  }
  let entries : Array[JointDiffusionKernelRequirement] = []
  add_conditioning_requirements(entries, plan, limits)
  add_transformer_requirements(entries, plan, limits)
  add_scheduler_and_decode_requirements(entries, plan, limits)
  if entries.is_empty() {
    abort("joint diffusion requirements unexpectedly empty")
  }
  let digest = canonical_requirements_digest(
    source_identity,
    plan.projection(),
    entries,
  )
  {
    source_identity,
    source_schema_version: plan.schema_version(),
    shape_contract: plan.shapes(),
    conditioning_limits: plan.conditioning(),
    denoiser_shape: plan.denoiser(),
    projection_contract: plan.projection(),
    digest,
    requirements: ReadOnlyArray::from_array(entries),
    capabilities: ordered_capabilities(entries),
  }
}
