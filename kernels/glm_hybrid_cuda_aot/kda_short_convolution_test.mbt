///|
fn kda_short_convolution_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaShortConvolution(_, _) {
      return ordinal
    }
  }
  abort("missing KDA short convolution")
}

///|
fn kda_short_convolution_requirements(
  head_count? : Int = 64,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements {
  glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=head_count,
  )
}

///|
test "GLM Flash KDA short convolution binds explicit sequence and state ABI" {
  let requirements = kda_short_convolution_requirements()
  let ordinal = kda_short_convolution_ordinal(requirements)
  let candidate = lower_kda_short_convolution_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    maximum_sequences=8,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), ReusableDepthwiseCausalSiluBf16)
  assert_eq(candidate.operation(), KdaShortConvolution(8192, 4))
  assert_eq(candidate.capability(), KdaShortConvolutionAot)
  assert_eq(candidate.channel_count(), 8192)
  assert_eq(candidate.kernel_size(), 4)
  assert_eq(candidate.maximum_tokens(), 32)
  assert_eq(candidate.maximum_sequences(), 8)
  assert_eq(candidate.dimensions().grid_x(), 1024)
  assert_eq(candidate.operands().length(), 7)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), KdaSequenceOffsetsInput)
  assert_eq(candidate.operands()[1].byte_count(), 36L)
  assert_eq(candidate.operands()[1].alignment(), 4L)
  assert_eq(candidate.operands()[2].role(), KdaProjectedInput)
  assert_eq(candidate.operands()[2].byte_count(), 524288L)
  assert_eq(candidate.operands()[3].role(), KdaShortConvolutionWeights)
  assert_eq(candidate.operands()[3].byte_count(), 65536L)
  assert_eq(candidate.operands()[4].role(), KdaConvolutionInitialStateInput)
  assert_eq(candidate.operands()[4].byte_count(), 393216L)
  assert_eq(candidate.operands()[5].role(), KdaConvolvedOutput)
  assert_eq(candidate.operands()[5].byte_count(), 524288L)
  assert_eq(candidate.operands()[6].role(), KdaConvolutionFinalStateOutput)
  assert_eq(candidate.operands()[6].byte_count(), 393216L)
  assert_eq(
    candidate.numerical_contract(),
    OrderedFloat32AccumulationSiluBf16RoundV1,
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("counts[1] != sequence_count - counts[0]"))
  assert_true(source.contains("sequence_offsets[sequence_count] != live_rows"))
  assert_true(source.contains("local_row - 3 + offset"))
  assert_true(source.contains("source < 0 ? initial_state[i + length]"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("invocations=independent-q-k-v"))
  assert_true(
    recipe.contains("weights=channel-kernel-oldest-to-current-bf16-no-bias"),
  )
  assert_true(
    recipe.contains(
      "state=sequence-channel-history-oldest-first-bf16-explicit-in-out",
    ),
  )
  assert_true(
    recipe.contains(
      "excluded=projection-recurrent-update-residual-norm-cache-ownership",
    ),
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA short convolution rejects profile shape limit target compiler and ordinal" {
  let wrong_shape = kda_short_convolution_requirements(head_count=63)
  try
    lower_kda_short_convolution_cuda_aot_candidate(
      wrong_shape,
      requirement_ordinal=kda_short_convolution_ordinal(wrong_shape),
      maximum_tokens=32,
      maximum_sequences=8,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("non-official KDA geometry admitted")
  }
  let exact = kda_short_convolution_requirements()
  let ordinal = kda_short_convolution_ordinal(exact)
  for rejected_sequences in [0, 33] {
    try
      lower_kda_short_convolution_cuda_aot_candidate(
        exact,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        maximum_sequences=rejected_sequences,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    catch {
      Invalid(Limit) => ()
      _ => fail("wrong limit error")
    } noraise {
      _ => fail("hostile sequence limit admitted")
    }
  }
  try
    lower_kda_short_convolution_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      maximum_sequences=8,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("neighboring KDA operation admitted")
  }
  try
    lower_kda_short_convolution_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      maximum_sequences=8,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  for
    bad_compiler in [
      glm_aot_compiler(reassociate=true),
      glm_aot_compiler(fused_multiply_add=true),
    ] {
    try
      lower_kda_short_convolution_cuda_aot_candidate(
        exact,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        maximum_sequences=8,
        target=glm_aot_target(),
        compiler=bad_compiler,
        limits=glm_aot_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("wrong compiler error")
    } noraise {
      _ => fail("hostile compiler admitted")
    }
  }
}
