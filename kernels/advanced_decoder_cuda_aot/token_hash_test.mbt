///|
fn aot_test_token_hash_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is TokenHashRouting(_, _, _, _) {
      return ordinal
    }
  }
  abort("token-hash test requirement missing")
}

///|
test "token-hash routing lowers with exact deterministic I32 ABI" {
  let requirements = aot_test_requirements(hash_layer_count=1)
  let ordinal = aot_test_token_hash_ordinal(requirements)
  let candidate = lower_token_hash_routing_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), TokenHashRoutingI32)
  assert_eq(candidate.numerical_contract(), BitExactInt32TableLookupV1)
  assert_eq(candidate.operation(), TokenHashRouting(1, 128, 2, 8))
  assert_eq(candidate.capability(), TokenHashRoutingAot)
  assert_eq(candidate.routed_expert_count(), 8)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.dimensions().shared_memory_bytes(), 0)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), TokenIdsInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[1].alignment(), 4L)
  assert_eq(candidate.operands()[2].role(), TokenHashTableInput)
  assert_eq(candidate.operands()[2].byte_count(), 1024L)
  assert_eq(candidate.operands()[2].alignment(), 4L)
  assert_eq(candidate.operands()[3].role(), RoutingIndicesOutput)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  assert_eq(candidate.operands()[3].alignment(), 4L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("live_rows > 32"))
  assert_true(source.contains("token >= 128"))
  assert_true(source.contains("slot >= 2"))
  assert_true(source.contains("expert < 8"))
  assert_true(source.contains("expert_indices[output_index] = -1"))
  let repeated = lower_token_hash_routing_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
  let foreign = lower_token_hash_routing_cuda_aot_candidate(
    aot_test_requirements(content_fill='c', hash_layer_count=1),
    requirement_ordinal=ordinal,
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
test "token-hash routing rejects adjacent ordinal target and compiler" {
  let requirements = aot_test_requirements(hash_layer_count=1)
  let ordinal = aot_test_token_hash_ordinal(requirements)
  for hostile_ordinal in [ordinal - 1, ordinal + 1] {
    try
      lower_token_hash_routing_cuda_aot_candidate(
        requirements,
        requirement_ordinal=hostile_ordinal,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, hostile_ordinal)
      _ => fail("unexpected adjacent-operation rejection")
    } noraise {
      _ => fail("adjacent operation was lowered as token hash")
    }
  }
  try
    lower_token_hash_routing_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected token-hash target rejection")
  } noraise {
    _ => fail("non-BF16 decoder target was admitted")
  }
  try
    lower_token_hash_routing_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected token-hash compiler rejection")
  } noraise {
    _ => fail("reassociating compiler was admitted")
  }
}
