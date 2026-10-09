///|
fn glm_aot_dsa_pooling_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexPooling(_, _, _) {
      return ordinal
    }
  }
  abort("missing DSA pooling requirement")
}

///|
test "GLM Flash DSA key pooling binds exact standalone primitive ABI" {
  let requirements = glm_aot_requirements(pooled=true)
  let ordinal = glm_aot_dsa_pooling_ordinal(requirements)
  let candidate = lower_dsa_key_pooling_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), DsaKeyPoolCompressionBf16)
  assert_eq(candidate.operation(), DsaIndexPooling(4, true, true))
  assert_eq(candidate.capability(), DsaIndexPoolingAot)
  assert_eq(candidate.head_dimension(), 128)
  assert_eq(candidate.pool_size(), 4)
  assert_eq(candidate.pool_capacity(), 8)
  assert_eq(candidate.dimensions().grid_x(), 1)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.operands().length(), 9)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), DsaIndexKeysInput)
  assert_eq(candidate.operands()[1].byte_count(), 8192L)
  assert_eq(candidate.operands()[2].role(), DsaIndexGateScoresInput)
  assert_eq(candidate.operands()[2].byte_count(), 8192L)
  assert_eq(candidate.operands()[3].role(), DsaIndexValidityInput)
  assert_eq(candidate.operands()[3].byte_count(), 128L)
  assert_eq(candidate.operands()[4].role(), ModelWeights)
  assert_eq(candidate.operands()[4].byte_count(), 1024L)
  assert_eq(candidate.operands()[5].role(), DsaPooledKeysOutput)
  assert_eq(candidate.operands()[5].byte_count(), 2048L)
  assert_eq(candidate.operands()[6].role(), DsaPooledRawIndicesOutput)
  assert_eq(candidate.operands()[6].byte_count(), 128L)
  assert_eq(candidate.operands()[7].role(), DsaPooledValidityOutput)
  assert_eq(candidate.operands()[7].byte_count(), 32L)
  assert_eq(candidate.operands()[8].role(), DsaPooledCountOutput)
  assert_eq(candidate.operands()[8].byte_count(), 4L)
  assert_eq(
    candidate.numerical_contract(),
    StableFloat32SoftmaxBf16ProbabilityOrderedFloat32AccumulationV1,
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const int live_tokens = counts[3];"))
  assert_true(source.contains("if (!all_valid) continue;"))
  assert_false(source.contains("topk"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("geometry=128:4:2048:32"))
  assert_true(
    recipe.contains(
      "tail-policy=downstream-always-select-incomplete-visible-tail",
    ),
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "GLM DSA pooling rejects semantic neighbors and hostile compiler policy" {
  let requirements = glm_aot_requirements(pooled=true)
  let ordinal = glm_aot_dsa_pooling_ordinal(requirements)
  try
    lower_dsa_key_pooling_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(rejected) => assert_eq(rejected, ordinal - 1)
    _ => fail("unexpected neighboring-operation rejection")
  } noraise {
    _ => fail("DSA index compute admitted as key pooling")
  }
  try
    lower_dsa_key_pooling_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler-policy rejection")
  } noraise {
    _ => fail("reassociated softmax admitted")
  }
}
