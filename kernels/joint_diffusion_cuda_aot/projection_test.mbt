///|
fn projection_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoLatentProjection(_, _, _, _, _) if latent == VideoLatent =>
        return ordinal
      AudioLatentProjection(_, _) if latent == AudioLatent => return ordinal
      _ => ()
    }
  }
  abort("test projection requirement missing")
}

///|
fn projection_test_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn projection_ordered_dense_bf16(
  input : ArrayView[Float],
  weights : ArrayView[Float],
  bias : Float,
) -> Int {
  let mut value = bias
  for index, input_value in input {
    let product = weights[index] * input_value
    value = value + product
  }
  projection_test_bf16_bits(value)
}

///|
test "video and audio projection candidates bind dense ABI and request rows" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  let video = lower_latent_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=projection_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(
    video.numerical_contract(),
    OrderedFloat32DenseF32BoundaryBf16RoundV1,
  )
  assert_eq(video.latent(), VideoLatent)
  assert_eq(video.row_count(), 3L)
  assert_eq(video.input_width(), 8)
  assert_eq(video.output_width(), 4)
  assert_eq(video.input_element_count(), 24L)
  assert_eq(video.output_element_count(), 12L)
  assert_eq(video.model_weight_element_count(), 36L)
  assert_eq(video.projection_weight_element_count(), 32L)
  assert_eq(video.projection_bias_element_count(), 4L)
  assert_eq(video.dimensions().grid_x(), 1)
  assert_eq(video.operands().length(), 4)
  assert_eq(video.operands()[0].role(), VideoLatentInput)
  assert_eq(video.operands()[0].byte_count(), 96L)
  assert_eq(video.operands()[1].role(), ProjectionWeight)
  assert_eq(video.operands()[1].byte_count(), 128L)
  assert_eq(video.operands()[2].role(), ProjectionBias)
  assert_eq(video.operands()[2].byte_count(), 16L)
  assert_eq(video.operands()[3].role(), JointHiddenOutput)
  assert_eq(video.operands()[3].byte_count(), 24L)
  assert_false(video.manifest_bindable())
  let source = try! @utf8.decode(video.source_bytes())
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("float accumulator = bias[column]"))
  let audio = lower_latent_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=projection_ordinal(requirements, AudioLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(audio.latent(), AudioLatent)
  assert_eq(audio.row_count(), 2L)
  assert_eq(audio.input_width(), 2)
  assert_eq(audio.output_width(), 4)
  assert_eq(audio.input_element_count(), 4L)
  assert_eq(audio.output_element_count(), 8L)
  assert_eq(audio.model_weight_element_count(), 12L)
  assert_eq(audio.operands()[0].role(), AudioLatentInput)
  assert_eq(audio.operands()[0].byte_count(), 16L)
  assert_eq(audio.operands()[1].byte_count(), 32L)
  assert_eq(audio.operands()[2].byte_count(), 16L)
  assert_eq(audio.operands()[3].byte_count(), 16L)
  assert_true(video.source_digest() != audio.source_digest())
  inspect(
    video.source_digest().as_hex(),
    content=(
      #|9cde313f80df4019385ccdb2a3dffbbfda9423c20e9ae41f1102f429e3872ebf
    ),
  )
  inspect(
    video.recipe_digest().as_hex(),
    content=(
      #|8a1a553a1aa22b17bd222f7f14fa64cf49dfdf87ea4341cd6744fdd7eb4c8a42
    ),
  )
  inspect(
    audio.source_digest().as_hex(),
    content=(
      #|752d44488d017b1fabedeb0cedd74736f97671e484963ea7eb86197e60671a45
    ),
  )
  inspect(
    audio.recipe_digest().as_hex(),
    content=(
      #|463a5f46281633eacec52fc8cb79ec87f725ddf2bc3fe88ac30d5e09f61df75c
    ),
  )
  let repeated = lower_latent_projection_cuda_aot_candidate(
    requirements,
    request_shape=request,
    requirement_ordinal=projection_ordinal(requirements, VideoLatent),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(video.source_digest(), repeated.source_digest())
  assert_eq(video.recipe_digest(), repeated.recipe_digest())
}

///|
test "ordered F32 dense contract has an independent BF16 bit oracle" {
  assert_eq(projection_ordered_dense_bf16([1.0, 2.0], [0.5, -1.0], 1.0), 0xbf00)
  assert_eq(projection_ordered_dense_bf16([1.0], [1.0], 0.0), 0x3f80)
}

///|
test "projection lowering rejects foreign operations and unsafe profiles" {
  let requirements = diffusion_aot_requirements()
  let request = diffusion_aot_request()
  try
    lower_latent_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=5,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(5) => ()
    _ => fail("unexpected unsupported-operation rejection")
  } noraise {
    _ => fail("shifted flow was lowered as a projection")
  }
  for
    compiler in [
      diffusion_aot_compiler(fused_multiply_add=true),
      diffusion_aot_compiler(reassociate=true),
    ] {
    try
      lower_latent_projection_cuda_aot_candidate(
        requirements,
        request_shape=request,
        requirement_ordinal=projection_ordinal(requirements, VideoLatent),
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
    lower_latent_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=projection_ordinal(requirements, VideoLatent),
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
    lower_latent_projection_cuda_aot_candidate(
      requirements,
      request_shape=request,
      requirement_ordinal=projection_ordinal(requirements, VideoLatent),
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=11L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected projection-bound rejection")
  } noraise {
    _ => fail("oversized projected output was accepted")
  }
}
