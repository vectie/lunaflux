///|
test "query and key-value RMSNorm lower as exact independent BF16 streams" {
  let requirements = aot_test_requirements()
  let candidate = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=3,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), QueryKeyRmsNormBf16)
  assert_eq(candidate.numerical_contract(), OrderedFloat32ReductionBf16RoundV1)
  assert_eq(candidate.operation(), QueryKeyRmsNorm(0.000001, 4, 8))
  assert_eq(candidate.capability(), QueryKeyRmsNormAot)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().grid_y(), 2)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.dimensions().shared_memory_bytes(), 1024)
  assert_eq(candidate.operands().length(), 7)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), QueryInput)
  assert_eq(candidate.operands()[1].byte_count(), 256L)
  assert_eq(candidate.operands()[2].role(), QueryNormWeights)
  assert_eq(candidate.operands()[2].byte_count(), 8L)
  assert_eq(candidate.operands()[3].role(), QueryOutput)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  assert_eq(candidate.operands()[4].role(), KeyValueInput)
  assert_eq(candidate.operands()[4].byte_count(), 512L)
  assert_eq(candidate.operands()[5].role(), KeyValueNormWeights)
  assert_eq(candidate.operands()[5].byte_count(), 16L)
  assert_eq(candidate.operands()[6].role(), KeyValueOutput)
  assert_eq(candidate.operands()[6].byte_count(), 512L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("live_rows > 32"))
  assert_true(source.contains("row * 4ULL"))
  assert_true(source.contains("row * 8ULL"))
  assert_true(source.contains("__fmul_rn(value, value)"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__fdiv_rn"))
  assert_true(source.contains("__float2bfloat16_rn"))
  let repeated = lower_foundation_cuda_aot_candidate(
    requirements,
    requirement_ordinal=3,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
  let foreign = lower_foundation_cuda_aot_candidate(
    aot_test_requirements(content_fill='c'),
    requirement_ordinal=3,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), foreign.source_digest())
  assert_true(candidate.recipe_digest() != foreign.recipe_digest())
  assert_true(candidate.source_identity() != foreign.source_identity())
}

///|
test "query and key-value RMSNorm rejects adjacent operation and hostile policy" {
  let requirements = aot_test_requirements()
  for ordinal in [2, 4] {
    try
      lower_foundation_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("unexpected adjacent-operation rejection")
    } noraise {
      _ => fail("adjacent operation was lowered as Q/K RMSNorm")
    }
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=3,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected Q/K target rejection")
  } noraise {
    _ => fail("non-BF16 Q/K RMSNorm target was admitted")
  }
  try
    lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=3,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected Q/K compiler rejection")
  } noraise {
    _ => fail("reassociating Q/K RMSNorm compiler was admitted")
  }
}
