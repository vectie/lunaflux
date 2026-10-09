///|
fn output_projection_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoOutputProjection(_, _, _, _, _) if latent == VideoLatent =>
        return ordinal
      AudioOutputProjection(_, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("test output projection requirement missing")
}

///|
test "video and audio output projections bind exact F32 selected-row ABI" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video = lower_output_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.numerical_contract(), OrderedFloat32DenseF32OutputV1)
  assert_eq(video.operation(), VideoOutputProjection(2, 1, 2, 2, 4))
  assert_eq(video.capability(), VideoOutputProjectionAot)
  assert_eq(video.latent(), VideoLatent)
  assert_eq(video.row_count(), 3L)
  assert_eq(video.input_width(), 4)
  assert_eq(video.output_width(), 8)
  assert_eq(video.input_element_count(), 12L)
  assert_eq(video.output_element_count(), 24L)
  assert_eq(video.projection_weight_element_count(), 32L)
  assert_eq(video.projection_bias_element_count(), 8L)
  assert_eq(video.model_weight_element_count(), 40L)
  assert_eq(video.dimensions().grid_x(), 1)
  assert_eq(video.operands().length(), 4)
  assert_eq(video.operands()[0].role(), VideoF32HeadInput)
  assert_eq(video.operands()[0].byte_count(), 48L)
  assert_eq(video.operands()[1].role(), ProjectionWeight)
  assert_eq(video.operands()[1].byte_count(), 128L)
  assert_eq(video.operands()[2].role(), ProjectionBias)
  assert_eq(video.operands()[2].byte_count(), 32L)
  assert_eq(video.operands()[3].role(), VideoPredictionOutput)
  assert_eq(video.operands()[3].byte_count(), 96L)
  assert_false(video.manifest_bindable())
  let source = try! @utf8.decode(video.source_bytes())
  assert_true(source.contains("const float *input"))
  assert_true(source.contains("const float *weight"))
  assert_true(source.contains("const float *bias"))
  assert_true(source.contains("float *output"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("__nv_bfloat16"))
  let audio = lower_output_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=output_projection_ordinal(requirements, AudioLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(audio.operation(), AudioOutputProjection(2, 4))
  assert_eq(audio.capability(), AudioOutputProjectionAot)
  assert_eq(audio.latent(), AudioLatent)
  assert_eq(audio.row_count(), 2L)
  assert_eq(audio.input_width(), 4)
  assert_eq(audio.output_width(), 2)
  assert_eq(audio.input_element_count(), 8L)
  assert_eq(audio.output_element_count(), 4L)
  assert_eq(audio.projection_weight_element_count(), 8L)
  assert_eq(audio.projection_bias_element_count(), 2L)
  assert_eq(audio.operands()[0].role(), AudioF32HeadInput)
  assert_eq(audio.operands()[0].byte_count(), 32L)
  assert_eq(audio.operands()[3].role(), AudioPredictionOutput)
  assert_eq(audio.operands()[3].byte_count(), 16L)
  assert_true(video.source_digest() != audio.source_digest())
  assert_true(video.recipe_digest() != audio.recipe_digest())
  let repeated = lower_output_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.source_digest(), repeated.source_digest())
  assert_eq(video.recipe_digest(), repeated.recipe_digest())
}

///|
test "output projection lowering rejects foreign and unsafe contracts" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let wrong_storage = diffusion_aot_requirements(float32_output_heads=false)
  try
    lower_output_projection_cuda_aot_candidate(
      wrong_storage,
      request_shape=request,
      requirement_ordinal=output_projection_ordinal(wrong_storage, VideoLatent),
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("unexpected output-storage rejection")
  } noraise {
    _ => fail("BF16 output-head storage was weakened into the F32 candidate")
  }
  try
    lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=diffusion_aot_foreign_request(),
      requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign-request rejection")
  } noraise {
    _ => fail("foreign request geometry was accepted")
  }
  try
    lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=3,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(3) => ()
    _ => fail("unexpected foreign-operation rejection")
  } noraise {
    _ => fail("input projection was weakened into an output head")
  }
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_output_projection_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
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
  try
    lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
      target=diffusion_aot_target(bf16=false),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("a target unable to execute the BF16 denoiser was accepted")
  }
  try
    lower_output_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=output_projection_ordinal(requirements, VideoLatent),
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=11L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected output-projection bound rejection")
  } noraise {
    _ => fail("oversized F32 hidden input was accepted")
  }
}
