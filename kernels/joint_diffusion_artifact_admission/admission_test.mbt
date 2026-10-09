///|
fn admission_test_content(fill : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, fill))
}

///|
fn admission_test_plan(
  fill? : Char = 'a',
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=1,
    content_digest=admission_test_content(fill),
    workflows=[
      TextToAudioVideo,
      FirstLastFrameToAudioVideo,
      ReferenceToAudioVideo,
    ],
    stages=[
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=ResolveRequest,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeTextConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeKeyframeConditioning,
        presence=Optional,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeReferenceConditioning,
        presence=Optional,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=InitializeJointLatents,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=JointAudioVideoDenoise,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeVideo,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeAudio,
        presence=Required,
      ),
    ],
    video_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=VideoLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=2,
      shift_denominator=1,
    ),
    audio_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=AudioLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=1,
      shift_denominator=1,
    ),
    shapes=@joint_diffusion_plan.JointDiffusionShapeContract::new(
      frames_per_second=4,
      min_duration_seconds=1,
      max_duration_seconds=2,
      frames_per_chunk=3,
      latents_per_chunk=1,
      canvas_multiple=4,
      max_canvas_pixels=16L,
      min_aspect_numerator=1,
      min_aspect_denominator=1,
      max_aspect_numerator=1,
      max_aspect_denominator=1,
      video_latent_channels=2,
      video_spatial_compression=2,
      audio_latent_channels=2,
      audio_latents_per_second=2,
      audio_sample_rate=8,
      audio_output_channels=1,
      audio_vae_latent_dimension=4,
      audio_vae_decoder_dimension=2,
      min_inference_steps=2,
      max_inference_steps=8,
      video_patch_temporal=1,
      video_patch_height=2,
      video_patch_width=2,
    ),
    conditioning=@joint_diffusion_plan.JointDiffusionConditioningLimits::new(
      max_keyframes=2,
      max_image_references=3,
      max_video_references=2,
      max_audio_references=1,
      max_total_references=6,
      allow_audio_only_references=false,
    ),
    projection=@joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=8,
      layer_count=2,
      refiner_layer_count=1,
      attention_head_count=2,
      attention_head_dimension=4,
      feed_forward_dimension=16,
      text_conditioning_dimension=8,
      timestep_frequency_dimension=2,
    ),
  )
}

///|
fn admission_test_requirements(
  fill? : Char = 'a',
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements {
  try! @joint_diffusion_kernel_requirements.derive_requirements(
    admission_test_plan(fill~),
    limits=try! @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementLimits::new(
      maximum_transformer_layers=8,
      maximum_requirements=34,
    ),
  )
}

///|
fn admission_test_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn admission_test_dimensions() -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn admission_test_digest(bytes : Bytes) -> @catalog.AotArtifactDigest {
  try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn admission_test_module_digest() -> @catalog.AotArtifactDigest {
  admission_test_digest(b"abc")
}

///|
fn admission_test_entry(
  ordinal : Int,
  digest? : @catalog.AotArtifactDigest = admission_test_module_digest(),
) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=digest,
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(ordinal + 1),
    ),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
  )
}

///|
fn admission_test_roles(
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
) -> Array[JointDiffusionLaunchOperandRole] {
  match operation {
    TokenTextConditioning(_) => [TokenIdsInput, TextConditioningOutput]
    OrderedKeyframeConditioning(_, _) =>
      [KeyframeMediaInput, OrderedConditioningOutput]
    OrderedReferenceConditioning(_, _, _, _, _) =>
      [OrderedReferenceMediaInput, OrderedConditioningOutput]
    VideoLatentProjection(_, _, _, _, _) =>
      [VideoLatentInput, ProjectionWeight, ProjectionBias, JointHiddenOutput]
    AudioLatentProjection(_, _) =>
      [AudioLatentInput, ProjectionWeight, ProjectionBias, JointHiddenOutput]
    JointTransformerAttention(_, _, _, _, _) =>
      [JointHiddenInput, ModelWeights, JointHiddenOutput, Workspace]
    JointTransformerMlp(_, _, _, _) =>
      [
        JointHiddenInput,
        TransformerMlpPackedGateUpWeight,
        TransformerMlpDownWeight,
        JointHiddenOutput,
      ]
    VideoFinalRmsNorm(_, _, _) =>
      [VideoFinalNormInput, FinalNormWeight, VideoFinalNormOutput]
    AudioFinalRmsNorm(_, _, _) =>
      [AudioFinalNormInput, FinalNormWeight, AudioFinalNormOutput]
    VideoFinalAdaLnF32Boundary(_) =>
      [
        VideoFinalNormOutput,
        FinalAdaLnShiftInput,
        FinalAdaLnScaleInput,
        VideoF32HeadInput,
      ]
    AudioFinalAdaLnF32Boundary(_) =>
      [
        AudioFinalNormOutput,
        FinalAdaLnShiftInput,
        FinalAdaLnScaleInput,
        AudioF32HeadInput,
      ]
    FinalAdaLnParameterRow(_, _) =>
      [
        FinalAdaLnConditioningInput,
        FinalAdaLnWeight,
        FinalAdaLnBias,
        FinalAdaLnShiftOutput,
        FinalAdaLnScaleOutput,
      ]
    PackedModalityAdaLnRows(_) =>
      [
        PackedAdaLnShiftInput,
        PackedAdaLnScaleInput,
        VideoPackedRowIndices,
        AudioPackedRowIndices,
        VideoAdaLnShiftOutput,
        VideoAdaLnScaleOutput,
        AudioAdaLnShiftOutput,
        AudioAdaLnScaleOutput,
      ]
    TimestepFrequencyRows(_) =>
      [DistinctTimestepValuesInput, TimestepFrequencyRowsOutput]
    TimestepEmbeddingMlpRows(_, _, _) =>
      [
        TimestepFrequencyRowsOutput,
        TimestepMlpWeight1,
        TimestepMlpBias1,
        TimestepMlpWeight2,
        TimestepMlpBias2,
        TimestepEmbeddingRowsOutput,
      ]
    DistinctTimestepAdaLnParameterRows(_, _) =>
      [
        TimestepEmbeddingRowsOutput,
        FinalAdaLnWeight,
        FinalAdaLnBias,
        FinalAdaLnShiftOutput,
        FinalAdaLnScaleOutput,
      ]
    VideoOutputProjection(_, _, _, _, _) =>
      [
        VideoF32HeadInput,
        ProjectionWeight,
        ProjectionBias,
        VideoPredictionOutput,
      ]
    AudioOutputProjection(_, _) =>
      [
        AudioF32HeadInput,
        ProjectionWeight,
        ProjectionBias,
        AudioPredictionOutput,
      ]
    ShiftedFlowStep(VideoLatent, _, _, _) =>
      [
        VideoLatentInput,
        VideoPredictionInput,
        FlowCoefficientsInput,
        VideoLatentOutput,
      ]
    ShiftedFlowStep(AudioLatent, _, _, _) =>
      [
        AudioLatentInput,
        AudioPredictionInput,
        FlowCoefficientsInput,
        AudioLatentOutput,
      ]
    VideoVaeLatentDestandardization(_, _, _) =>
      [
        VideoNormalizedLatentInput,
        VideoLatentStandardDeviationInput,
        VideoLatentMeanInput,
        VideoDestandardizedLatentOutput,
      ]
    VideoVaeDecoderBody(_, _, _, _) =>
      [VideoDestandardizedLatentOutput, ModelWeights, VideoMediaOutput]
    AudioVaeLatentDestandardization(_, _, _, _) =>
      [
        AudioNormalizedLatentInput,
        AudioLatentStandardDeviationInput,
        AudioLatentMeanInput,
        AudioDestandardizedLatentOutput,
      ]
    AudioVaeDecoderInputProjection(_, _, _, _, _, _) =>
      [
        AudioDestandardizedLatentOutput,
        AudioDecoderInputProjectionWeight,
        AudioDecoderInputProjectionBias,
        AudioDecoderInputProjectionOutput,
      ]
    AudioVaeDecoderConvPre(_, _, _, _, _, _, _, _, _) =>
      [
        AudioDecoderInputProjectionOutput,
        AudioDecoderConvPreWeightG,
        AudioDecoderConvPreWeightV,
        AudioDecoderConvPreBias,
        AudioDecoderConvPreNormalizedWeightWorkspace,
        AudioDecoderConvPreOutput,
      ]
    AudioVaeDecoderFirstUpsample(_, _, _, _, _, _, _, _, _) =>
      [
        AudioDecoderConvPreOutput,
        AudioDecoderFirstUpsampleWeightG,
        AudioDecoderFirstUpsampleWeightV,
        AudioDecoderFirstUpsampleBias,
        AudioDecoderFirstUpsampleNormalizedWeightWorkspace,
        AudioDecoderFirstUpsampleOutput,
      ]
    AudioVaeDecoderFirstAmpActivation(_, _, _, _, _, _) =>
      [
        AudioDecoderFirstUpsampleOutput,
        AudioDecoderFirstAmpAlpha,
        AudioDecoderFirstAmpBeta,
        AudioDecoderFirstAmpUpsampleFilter,
        AudioDecoderFirstAmpDownsampleFilter,
        AudioDecoderFirstAmpUpsampleWorkspace,
        AudioDecoderFirstAmpActivatedWorkspace,
        AudioDecoderFirstAmpActivationOutput,
      ]
    AudioVaeDecoderFirstAmpStage(_, _, _, _, _, _) =>
      [
        AudioDecoderFirstUpsampleOutput,
        AudioDecoderFirstAmpActivationOutput,
        ModelWeights,
        Workspace,
        AudioDecoderFirstAmpStageOutput,
      ]
    AudioVaeDecoderUpsampleTransition(stage, _, _, _, _, _, _, _, _, _) =>
      admission_test_upsample_transition_roles(stage)
    AudioVaeDecoderAmpActivation(stage, _, _, _, _, _, _) =>
      admission_test_amp_activation_roles(stage)
    AudioVaeDecoderAmpStage(_, _, _, _, _, _, _, _) =>
      [
        AudioDecoderUpsampleTransitionOutput,
        AudioDecoderAmpActivationOutput,
        ModelWeights,
        Workspace,
        AudioDecoderAmpStageOutput,
      ]
    AudioVaeDecoderBody(_, _, _, _) =>
      [AudioDecoderAmpActivation2Output, ModelWeights, AudioMediaOutput]
  }
}

///|
fn admission_test_launch(
  target : @catalog.DeviceTarget,
  ordinal : Int,
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
  capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability,
  digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest,
  roles : ArrayView[JointDiffusionLaunchOperandRole],
) -> JointDiffusionLaunchEvidenceInput {
  try! JointDiffusionLaunchEvidenceInput::new(
    requirements_digest=digest,
    requirement_ordinal=ordinal,
    operation~,
    capability~,
    target~,
    entry_point=admission_test_entry(ordinal),
    dimensions=admission_test_dimensions(),
    operands=admission_test_operands(roles),
  )
}

///|
fn admission_test_launches(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  target : @catalog.DeviceTarget,
) -> Array[JointDiffusionLaunchEvidenceInput] {
  let launches : Array[JointDiffusionLaunchEvidenceInput] = []
  for ordinal, requirement in requirements.requirements() {
    launches.push(
      admission_test_launch(
        target,
        ordinal,
        requirement.operation(),
        requirement.capability(),
        requirements.digest(),
        admission_test_roles(requirement.operation()),
      ),
    )
  }
  launches
}

///|
fn admission_test_entries(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Array[@artifact.KernelEntryPointInput] {
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for ordinal, _ in requirements.requirements() {
    entries.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=admission_test_entry(ordinal),
        function_symbol="joint_diffusion_\{ordinal}",
      ),
    )
  }
  entries
}

///|
fn admission_test_modules() -> Array[JointDiffusionModuleInput] {
  [
    JointDiffusionModuleInput::new(
      digest=admission_test_module_digest(),
      module_bytes=b"abc",
    ),
  ]
}

///|
test "exact evidence is canonical inert and requirement ordered" {
  let requirements = admission_test_requirements()
  let target = admission_test_target()
  let modules = admission_test_modules()
  let entries = admission_test_entries(requirements)
  let launches = admission_test_launches(requirements, target)
  let first = admit(
    requirements,
    target~,
    module_inputs=modules,
    entry_point_inputs=entries,
    launch_inputs=launches,
    limits=admission_test_limits(),
  )
  let reordered_launches = launches.copy()
  let last = reordered_launches.length() - 1
  let saved_launch = reordered_launches[0]
  reordered_launches[0] = reordered_launches[last]
  reordered_launches[last] = saved_launch
  let reordered_entries = entries.copy()
  let saved_entry = reordered_entries[0]
  reordered_entries[0] = reordered_entries[last]
  reordered_entries[last] = saved_entry
  let repeated = admit(
    requirements,
    target~,
    module_inputs=modules,
    entry_point_inputs=reordered_entries,
    launch_inputs=reordered_launches,
    limits=admission_test_limits(),
  )
  assert_eq(first.schema_version(), 1)
  assert_eq(first.source_identity(), requirements.source_identity())
  assert_eq(first.requirements_digest(), requirements.digest())
  assert_eq(first.status(), InertUnqualified)
  assert_eq(first.modules().length(), 1)
  assert_eq(first.entry_points().length(), 34)
  assert_eq(first.launches().length(), 34)
  assert_eq(first.total_module_bytes(), 3L)
  assert_eq(first.digest(), repeated.digest())
  for ordinal, launch in first.launches() {
    assert_eq(launch.requirement_ordinal(), ordinal)
    assert_eq(
      launch.operation(),
      requirements.requirements()[ordinal].operation(),
    )
  }
  try first.require_execution_authority() catch {
    NoExecutionAuthority => ()
    _ => fail("unexpected execution-authority rejection")
  } noraise {
    _ => fail("inert admission granted execution authority")
  }
}
