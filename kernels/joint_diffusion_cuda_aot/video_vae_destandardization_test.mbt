///|
fn video_vae_destandardization_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoVaeLatentDestandardization(_, _, _) => return ordinal
      _ => ()
    }
  }
  abort("video latent destandardization requirement missing")
}

///|
fn video_vae_destandardization_parameters(
  plan? : @joint_diffusion_plan.JointDiffusionPlan,
  means? : Array[Float] = [0.5F, -1.0F],
) -> @joint_diffusion_vae_reference.JointVaeLatentDestandardization {
  let exact_plan = match plan {
    Some(value) => value
    None => diffusion_aot_plan()
  }
  try! @joint_diffusion_vae_reference.bind_video_latent_destandardization(
    exact_plan,
    try! exact_plan
    .shapes()
    .resolve_request(height=4, width=4, requested_frames=4, inference_steps=5),
    means~,
    standard_deviations=[2.0F, 0.25F],
  )
}

///|
test "video VAE latent destandardization binds exact F32 request ABI" {
  let requirements = diffusion_aot_requirements()
  let parameters = video_vae_destandardization_parameters()
  let candidate = lower_video_vae_latent_destandardization_cuda_aot_candidate(
    requirements,
    parameters~,
    requirement_ordinal=video_vae_destandardization_ordinal(requirements),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(
    candidate.numerical_contract(),
    OrderedF32ChannelMajorDestandardizationV1,
  )
  assert_eq(
    candidate.operation(),
    VideoVaeLatentDestandardization(
      2,
      Float32InputScaleMeanOutput,
      BatchChannelFrameHeightWidth,
    ),
  )
  assert_eq(candidate.capability(), VideoVaeLatentDestandardizationAot)
  assert_eq(candidate.parameter_digest(), parameters.digest())
  assert_eq(candidate.channel_count(), 2)
  assert_eq(candidate.batch_count(), 1)
  assert_eq(candidate.elements_per_channel(), 12L)
  assert_eq(candidate.element_count(), 24L)
  assert_eq(candidate.dimensions().grid_x(), 1)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), VideoNormalizedLatentInput)
  assert_eq(candidate.operands()[0].byte_count(), 96L)
  assert_eq(candidate.operands()[1].role(), VideoLatentStandardDeviationInput)
  assert_eq(candidate.operands()[1].byte_count(), 8L)
  assert_eq(candidate.operands()[2].role(), VideoLatentMeanInput)
  assert_eq(candidate.operands()[3].role(), VideoDestandardizedLatentOutput)
  assert_false(candidate.manifest_bindable())
  let source = try! @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const float *normalized"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("__nv_bfloat16"))
  assert_true(
    try! @utf8.decode(candidate.recipe_bytes()).contains(
      "|parameters=\{parameters.digest().as_hex()}|",
    ),
  )
  let shifted = lower_video_vae_latent_destandardization_cuda_aot_candidate(
    requirements,
    parameters=video_vae_destandardization_parameters(means=[0.25F, -1.0F]),
    requirement_ordinal=video_vae_destandardization_ordinal(requirements),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(candidate.source_digest(), shifted.source_digest())
  assert_true(candidate.recipe_digest() != shifted.recipe_digest())
  let f32_only_target = lower_video_vae_latent_destandardization_cuda_aot_candidate(
    requirements,
    parameters~,
    requirement_ordinal=video_vae_destandardization_ordinal(requirements),
    target=diffusion_aot_target(bf16=false),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(candidate.source_digest(), f32_only_target.source_digest())
  assert_true(candidate.recipe_digest() != f32_only_target.recipe_digest())
}

///|
test "video VAE latent destandardization rejects hostile joins and policy" {
  let requirements = diffusion_aot_requirements()
  let parameters = video_vae_destandardization_parameters()
  let ordinal = video_vae_destandardization_ordinal(requirements)
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_video_vae_latent_destandardization_cuda_aot_candidate(
        requirements,
        parameters~,
        requirement_ordinal=ordinal,
        target=diffusion_aot_target(),
        compiler~,
        limits=diffusion_aot_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected compiler rejection")
    } noraise {
      _ => fail("unsafe compiler admitted")
    }
  }
  try
    lower_video_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal + 1,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected decoder-body ordinal rejection")
  } noraise {
    _ => fail("decoder body weakened into destandardization")
  }
  try
    lower_video_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters=video_vae_destandardization_parameters(
        plan=diffusion_aot_plan(float32_output_heads=false),
      ),
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("unexpected foreign plan rejection")
  } noraise {
    _ => fail("foreign plan coefficient contract admitted")
  }
  try
    lower_video_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=23L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected element-limit rejection")
  } noraise {
    _ => fail("oversized request admitted")
  }
}
