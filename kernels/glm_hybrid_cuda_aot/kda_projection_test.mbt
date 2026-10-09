///|
fn kda_projection_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaProjection(_, _) {
      return ordinal
    }
  }
  abort("missing KDA projection")
}

///|
test "KDA projection binds QKV explicit forget intermediates and beta" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_projection_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), SerialQkvForgetAndInputGateBf16)
  assert_eq(candidate.operation(), KdaProjection(64, 128))
  assert_eq(candidate.capability(), KdaProjectionAot)
  assert_eq(candidate.hidden_width(), 4096)
  assert_eq(candidate.control_rank(), 128)
  assert_eq(candidate.operands().length(), 14)
  assert_eq(candidate.operands()[11].role(), KdaForgetLowRankOutput)
  assert_eq(candidate.operands()[11].byte_count(), 8192L)
  assert_eq(candidate.operands()[12].role(), KdaRawForgetOutput)
  assert_eq(candidate.operands()[12].byte_count(), 524288L)
  assert_eq(candidate.operands()[13].role(), KdaBetaOutput)
  assert_eq(candidate.operands()[13].byte_count(), 4096L)
  assert_eq(candidate.dimensions().grid_x(), 1)
  assert_eq(candidate.dimensions().block_x(), 1)
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("excluded=f32-decay-control-short-convolution"))
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA projection rejects wrong geometry ordinal target and compiler" {
  let wrong = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=63,
  )
  try
    lower_kda_projection_cuda_aot_candidate(
      wrong,
      requirement_ordinal=kda_projection_ordinal(wrong),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("wrong projection geometry admitted")
  }
  let exact = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_projection_ordinal(exact)
  try
    lower_kda_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("short convolution admitted as projection")
  }
  try
    lower_kda_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
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
  try
    lower_kda_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
}
