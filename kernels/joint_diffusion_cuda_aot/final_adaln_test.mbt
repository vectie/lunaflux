///|
fn final_adaln_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoFinalAdaLnF32Boundary(_) if latent == VideoLatent => return ordinal
      AudioFinalAdaLnF32Boundary(_) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("test final AdaLN requirement missing")
}

///|
test "final AdaLN closes BF16 modulation to F32 head boundary exactly" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video = lower_final_adaln_f32_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=final_adaln_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.numerical_contract(), StagedBf16AdaLnF32BoundaryV1)
  assert_eq(video.operation(), VideoFinalAdaLnF32Boundary(4))
  assert_eq(video.row_count(), 3L)
  assert_eq(video.element_count(), 12L)
  assert_eq(video.dimensions().grid_x(), 1)
  assert_eq(video.operands().map(op => (op.role(), op.byte_count())), [
    (VideoFinalNormOutput, 24L),
    (FinalAdaLnShiftInput, 24L),
    (FinalAdaLnScaleInput, 24L),
    (VideoF32HeadInput, 48L),
  ])
  assert_false(video.manifest_bindable())
  let source = try! @utf8.decode(video.source_bytes())
  assert_true(source.contains("one_plus_scale = __float2bfloat16_rn"))
  assert_true(source.contains("scaled = __float2bfloat16_rn"))
  assert_true(source.contains("modulated = __float2bfloat16_rn"))
  assert_true(source.contains("float *output"))
  let audio = lower_final_adaln_f32_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=final_adaln_ordinal(requirements, AudioLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(audio.operation(), AudioFinalAdaLnF32Boundary(4))
  assert_eq(audio.row_count(), 2L)
  assert_eq(audio.element_count(), 8L)
  assert_eq(audio.operands()[0].role(), AudioFinalNormOutput)
  assert_eq(audio.operands()[3].role(), AudioF32HeadInput)
  assert_eq(audio.operands()[3].byte_count(), 32L)
  assert_true(video.recipe_digest() != audio.recipe_digest())
}

///|
test "final AdaLN lowering rejects hostile request ordinal target compiler and limit" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let ordinal = final_adaln_ordinal(requirements, VideoLatent)
  try
    lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=diffusion_aot_foreign_request(),
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected request rejection")
  } noraise {
    _ => fail("foreign request accepted")
  }
  try
    lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=0,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected ordinal rejection")
  } noraise {
    _ => fail("wrong ordinal accepted")
  }
  try
    lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(bf16=false),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("wrong target accepted")
  }
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_final_adaln_f32_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=ordinal,
        target=diffusion_aot_target(),
        compiler~,
        limits=diffusion_aot_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected compiler rejection")
    } noraise {
      _ => fail("unsafe compiler accepted")
    }
  }
  try
    lower_final_adaln_f32_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=11L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected limit rejection")
  } noraise {
    _ => fail("undersized limit accepted")
  }
}
