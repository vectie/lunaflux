///|
fn diffusion_aot_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, 'e'))
}

///|
fn diffusion_aot_plan(
  float32_output_heads? : Bool = true,
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=1,
    content_digest=diffusion_aot_content(),
    workflows=[TextToAudioVideo],
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
      shift_numerator=12,
      shift_denominator=1,
    ),
    audio_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=AudioLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=3,
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
      max_keyframes=0,
      max_image_references=0,
      max_video_references=0,
      max_audio_references=0,
      max_total_references=0,
      allow_audio_only_references=false,
    ),
    projection=if float32_output_heads {
      @joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads()
    } else {
      @joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_bfloat16_heads()
    },
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=4,
      layer_count=1,
      refiner_layer_count=0,
      attention_head_count=1,
      attention_head_dimension=4,
      feed_forward_dimension=8,
      text_conditioning_dimension=4,
      timestep_frequency_dimension=2,
    ),
  )
}

///|
fn diffusion_aot_requirements(
  float32_output_heads? : Bool = true,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements {
  try! @joint_diffusion_kernel_requirements.derive_requirements(
    diffusion_aot_plan(float32_output_heads~),
    limits=try! @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementLimits::new(
      maximum_transformer_layers=4,
      maximum_requirements=32,
    ),
  )
}

///|
fn diffusion_aot_request() -> @joint_diffusion_plan.JointDiffusionRequestShape {
  try! diffusion_aot_plan()
  .shapes()
  .resolve_request(height=4, width=4, requested_frames=4, inference_steps=5)
}

///|
fn diffusion_aot_foreign_request() -> @joint_diffusion_plan.JointDiffusionRequestShape {
  let foreign_shapes = try! @joint_diffusion_plan.JointDiffusionShapeContract::new(
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
    audio_latents_per_second=4,
    audio_sample_rate=8,
    audio_output_channels=1,
    audio_vae_latent_dimension=4,
    audio_vae_decoder_dimension=2,
    min_inference_steps=2,
    max_inference_steps=8,
    video_patch_temporal=1,
    video_patch_height=2,
    video_patch_width=2,
  )
  try! foreign_shapes.resolve_request(
    height=4,
    width=4,
    requested_frames=4,
    inference_steps=5,
  )
}

///|
fn diffusion_aot_target(bf16? : Bool = true) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=bf16,
    supports_cublas_lt=true,
  )
}

///|
fn diffusion_aot_compiler(
  fused_multiply_add? : Bool = false,
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, 'f'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add~,
    reassociate~,
    max_registers=128,
  )
}

///|
fn diffusion_aot_limits(
  maximum_elements? : Int64 = 4096L,
  maximum_source_bytes? : Int = 65536,
  maximum_recipe_bytes? : Int = 65536,
) -> JointDiffusionCudaAotLimits {
  try! JointDiffusionCudaAotLimits::new(
    maximum_elements~,
    maximum_source_bytes~,
    maximum_recipe_bytes~,
  )
}

///|
test "video and audio shifted-flow candidates bind exact request geometry" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video = lower_shifted_flow_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=16,
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.latent(), VideoLatent)
  assert_eq(video.element_count(), 24L)
  assert_eq(video.operation(), ShiftedFlowStep(VideoLatent, 12, 1, 1000))
  assert_eq(video.capability(), ShiftedFlowStepAot)
  assert_eq(video.operands().length(), 4)
  assert_eq(video.operands()[0].role(), VideoLatentInput)
  assert_eq(video.operands()[0].byte_count(), 96L)
  assert_eq(video.operands()[2].role(), FlowCoefficientsInput)
  assert_eq(video.operands()[2].byte_count(), 12L)
  assert_eq(video.operands()[3].role(), VideoLatentOutput)
  assert_false(video.manifest_bindable())
  let video_source = try! @utf8.decode(video.source_bytes())
  assert_false(video_source.contains("bfloat16"))
  assert_true(video_source.contains("__fadd_rn(x, scaled)"))
  assert_eq(video.numerical_contract(), OrderedFloat32RfVelocityEtaZeroV1)
  assert_true(video_source.contains("__fmul_rn"))
  assert_true(video_source.contains("__fadd_rn"))
  let audio = lower_shifted_flow_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=17,
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(audio.latent(), AudioLatent)
  assert_eq(audio.element_count(), 4L)
  assert_eq(audio.operation(), ShiftedFlowStep(AudioLatent, 3, 1, 1000))
  assert_eq(audio.operands()[0].role(), AudioLatentInput)
  assert_eq(audio.operands()[0].byte_count(), 16L)
  assert_eq(audio.operands()[1].role(), AudioPredictionInput)
  assert_eq(audio.operands()[2].role(), FlowCoefficientsInput)
  assert_eq(audio.operands()[3].role(), AudioLatentOutput)
  assert_true(video.source_digest() != audio.source_digest())
  inspect(
    video.source_digest().as_hex(),
    content=(
      #|65cae764ce5c74e6b1b8832513b86f7e0be87f92918a2786b574eb3042a993d2
    ),
  )
  inspect(
    video.recipe_digest().as_hex(),
    content=(
      #|3d99a7bf018387f455bec44f0028fc7e955acc753a6e93f6dae2728cc7c7e724
    ),
  )
  inspect(
    audio.source_digest().as_hex(),
    content=(
      #|bc132ee7f35159909d94b817ce86548739f70fb2b2cabdb804bbc03f7860e95c
    ),
  )
  inspect(
    audio.recipe_digest().as_hex(),
    content=(
      #|0445848fb5505d9bb506c46f17353417691b1d6f80cddd28f094126c1ab857b9
    ),
  )
  let repeated = lower_shifted_flow_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=16,
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.source_digest(), repeated.source_digest())
  assert_eq(video.recipe_digest(), repeated.recipe_digest())
}

///|
test "shifted-flow lowering rejects unsupported and unsafe profiles" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  try
    lower_shifted_flow_cuda_aot_candidate(
      requirements,
      request_shape=diffusion_aot_foreign_request(),
      requirement_ordinal=16,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign-request rejection")
  } noraise {
    _ => fail("request geometry from another shape contract was accepted")
  }
  try
    lower_shifted_flow_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=0,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected unsupported-operation rejection")
  } noraise {
    _ => fail("text conditioning was lowered as shifted flow")
  }
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_shifted_flow_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=16,
        target=diffusion_aot_target(),
        compiler~,
        limits=diffusion_aot_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected compiler-policy rejection")
    } noraise {
      _ => fail("unsafe compiler policy was accepted")
    }
  }
  let f32_only = lower_shifted_flow_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=16,
    target=diffusion_aot_target(bf16=false),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(f32_only.operands()[0].byte_count(), 96L)
  try
    lower_shifted_flow_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=16,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=23L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected element-bound rejection")
  } noraise {
    _ => fail("oversized request profile was accepted")
  }
  for
    limits in [
      diffusion_aot_limits(maximum_source_bytes=1),
      diffusion_aot_limits(maximum_recipe_bytes=1),
    ] {
    try
      lower_shifted_flow_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=16,
        target=diffusion_aot_target(),
        compiler=diffusion_aot_compiler(),
        limits~,
      )
    catch {
      Invalid(Encoding) => ()
      _ => fail("unexpected encoding-bound rejection")
    } noraise {
      _ => fail("undersized source/recipe limit was accepted")
    }
  }
}
