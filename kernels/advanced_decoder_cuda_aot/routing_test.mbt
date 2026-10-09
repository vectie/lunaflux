///|
fn expert_score_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is ExpertScoreTransform(_, _) {
      return ordinal
    }
  }
  abort("test requirements lost expert score transform")
}

///|
test "sqrt-softplus expert scores lower with exact deterministic F32 ABI" {
  let requirements = aot_test_requirements(score_function=SqrtSoftplus)
  let ordinal = expert_score_ordinal(requirements)
  let candidate = lower_expert_score_transform_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), ExpertScoreTransformF32)
  assert_eq(candidate.numerical_contract(), FiniteInputStableSqrtSoftplusF32V1)
  assert_eq(candidate.operation(), ExpertScoreTransform(SqrtSoftplus, 8))
  assert_eq(candidate.capability(), ExpertScoreTransformAot)
  assert_eq(candidate.routed_expert_count(), 8)
  assert_eq(candidate.maximum_tokens(), 32)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.dimensions().shared_memory_bytes(), 0)
  assert_eq(candidate.operands().length(), 3)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), RoutingScoresInput)
  assert_eq(candidate.operands()[1].byte_count(), 1024L)
  assert_eq(candidate.operands()[2].role(), RoutingScoresOutput)
  assert_eq(candidate.operands()[2].byte_count(), 1024L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("live_tokens > 32"))
  assert_true(source.contains("value > 0.0f"))
  assert_true(source.contains("__fadd_rn(value, log1pf(expf(-value)))"))
  assert_true(source.contains("log1pf(expf(value))"))
  assert_true(source.contains("sqrtf(softplus)"))
  assert_false(source.contains("bias"))
  assert_false(source.contains("bfloat16"))
  inspect(
    candidate.source_digest().as_hex(),
    content=(
      #|b03a790bff598e8be47afb1ee017338f97978b92db3b8c24d9cd8ad86ec4651f
    ),
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content=(
      #|1ca890c6b54cc9e22fefe96056746d5072bbc0f46004bd3746dfcc533ff84be1
    ),
  )
  let repeated = lower_expert_score_transform_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "expert-score lowering rejects adjacent semantics and hostile policies" {
  let sqrt_requirements = aot_test_requirements(score_function=SqrtSoftplus)
  let ordinal = expert_score_ordinal(sqrt_requirements)
  try
    lower_expert_score_transform_cuda_aot_candidate(
      sqrt_requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected foreign-operation rejection")
  } noraise {
    _ => fail("embedding was lowered as an expert score transform")
  }
  let sigmoid_requirements = aot_test_requirements()
  try
    lower_expert_score_transform_cuda_aot_candidate(
      sigmoid_requirements,
      requirement_ordinal=expert_score_ordinal(sigmoid_requirements),
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected sigmoid rejection")
  } noraise {
    _ => fail("unsupported sigmoid score transform was weakened")
  }
  try
    lower_expert_score_transform_cuda_aot_candidate(
      sqrt_requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("non-BF16 advanced-decoder target was admitted")
  }
  try
    lower_expert_score_transform_cuda_aot_candidate(
      sqrt_requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating compiler policy was admitted")
  }
  try
    lower_expert_score_transform_cuda_aot_candidate(
      sqrt_requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=4097,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected maximum-token rejection")
  } noraise {
    _ => fail("oversized expert-score row envelope was admitted")
  }
}
