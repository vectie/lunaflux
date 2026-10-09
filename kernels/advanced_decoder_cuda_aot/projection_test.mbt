///|
test "language-model head lowers with exact deterministic BF16 ABI" {
  let requirements = aot_test_requirements()
  let ordinal = requirements.requirements().length() - 1
  let candidate = lower_language_model_head_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), LanguageModelHeadBf16)
  assert_eq(candidate.numerical_contract(), OrderedF32AccumulationBf16RoundV1)
  assert_eq(candidate.operation(), LanguageModelHead)
  assert_eq(candidate.capability(), DenseProjectionAot)
  assert_eq(candidate.input_width(), 16)
  assert_eq(candidate.output_width(), 128)
  assert_eq(candidate.maximum_tokens(), 32)
  assert_eq(candidate.dimensions().grid_x(), 16)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.dimensions().shared_memory_bytes(), 0)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), HiddenInput)
  assert_eq(candidate.operands()[1].byte_count(), 1024L)
  assert_eq(candidate.operands()[2].role(), ModelWeights)
  assert_eq(candidate.operands()[2].byte_count(), 4096L)
  assert_eq(candidate.operands()[3].role(), LogitsOutput)
  assert_eq(candidate.operands()[3].byte_count(), 8192L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("row_count > 32"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("bias"))
  inspect(
    candidate.source_digest().as_hex(),
    content=(
      #|357c2d3521d5ba68cf0bff3cc7c90049d1923818a78951e0c2cb6948f57beee8
    ),
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content=(
      #|2d0bc7c8cf39ad4ead5cb071bc8706dec63c057d2510560443bacabdd428592e
    ),
  )
  let repeated = lower_language_model_head_cuda_aot_candidate(
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
test "language-model head lowering rejects foreign requirements and policy" {
  let requirements = aot_test_requirements()
  try
    lower_language_model_head_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected foreign-requirement rejection")
  } noraise {
    _ => fail("embedding requirement was lowered as a language-model head")
  }
  let ordinal = requirements.requirements().length() - 1
  try
    lower_language_model_head_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler-policy rejection")
  } noraise {
    _ => fail("reassociating language-model-head compiler was admitted")
  }
}
