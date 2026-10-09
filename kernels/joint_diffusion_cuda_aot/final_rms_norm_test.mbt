///|
fn final_norm_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoFinalRmsNorm(_, _, _) if latent == VideoLatent => return ordinal
      AudioFinalRmsNorm(_, _, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("test final RMSNorm requirement missing")
}

///|
test "video and audio final RMSNorm candidates preserve exact BF16 row ABI" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video = lower_final_rms_norm_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=final_norm_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.numerical_contract(), OrderedFloat32RmsNormBf16RoundV1)
  assert_eq(video.operation(), VideoFinalRmsNorm(4, 1, 100000))
  assert_eq(video.capability(), VideoFinalRmsNormAot)
  assert_eq(video.latent(), VideoLatent)
  assert_eq(video.row_count(), 3L)
  assert_eq(video.hidden_size(), 4)
  assert_eq(video.epsilon_numerator(), 1)
  assert_eq(video.epsilon_denominator(), 100000)
  assert_eq(video.element_count(), 12L)
  assert_eq(video.weight_element_count(), 4L)
  assert_eq(video.dimensions().grid_x(), 3)
  assert_eq(video.dimensions().block_x(), 256)
  assert_eq(video.dimensions().shared_memory_bytes(), 1024)
  assert_eq(video.operands().length(), 3)
  assert_eq(video.operands()[0].role(), VideoFinalNormInput)
  assert_eq(video.operands()[0].byte_count(), 24L)
  assert_eq(video.operands()[1].role(), FinalNormWeight)
  assert_eq(video.operands()[1].byte_count(), 8L)
  assert_eq(video.operands()[2].role(), VideoFinalNormOutput)
  assert_eq(video.operands()[2].byte_count(), 24L)
  assert_false(video.manifest_bindable())
  let source = try! @utf8.decode(video.source_bytes())
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__fdiv_rn"))
  assert_true(source.contains("rsqrtf"))
  assert_true(source.contains("__float2bfloat16_rn"))
  let audio = lower_final_rms_norm_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=final_norm_ordinal(requirements, AudioLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(audio.operation(), AudioFinalRmsNorm(4, 1, 100000))
  assert_eq(audio.capability(), AudioFinalRmsNormAot)
  assert_eq(audio.row_count(), 2L)
  assert_eq(audio.element_count(), 8L)
  assert_eq(audio.operands()[0].role(), AudioFinalNormInput)
  assert_eq(audio.operands()[0].byte_count(), 16L)
  assert_eq(audio.operands()[2].role(), AudioFinalNormOutput)
  assert_eq(audio.operands()[2].byte_count(), 16L)
  assert_true(video.source_digest() != audio.source_digest())
  assert_true(video.recipe_digest() != audio.recipe_digest())
  let repeated = lower_final_rms_norm_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=final_norm_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.source_digest(), repeated.source_digest())
  assert_eq(video.recipe_digest(), repeated.recipe_digest())
}

///|
test "final RMSNorm lowering rejects foreign operations and unsafe profiles" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video_ordinal = final_norm_ordinal(requirements, VideoLatent)
  try
    lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=diffusion_aot_foreign_request(),
      requirement_ordinal=video_ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign-request rejection")
  } noraise {
    _ => fail("foreign row geometry was accepted")
  }
  try
    lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=0,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected operation rejection")
  } noraise {
    _ => fail("conditioning was weakened into final RMSNorm")
  }
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_final_rms_norm_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=video_ordinal,
        target=diffusion_aot_target(),
        compiler~,
        limits=diffusion_aot_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected compiler rejection")
    } noraise {
      _ => fail("unsafe compiler policy was accepted")
    }
  }
  try
    lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=video_ordinal,
      target=diffusion_aot_target(bf16=false),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("non-BF16 target was accepted")
  }
  try
    lower_final_rms_norm_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=video_ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=11L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected size rejection")
  } noraise {
    _ => fail("oversized final norm was accepted")
  }
}
