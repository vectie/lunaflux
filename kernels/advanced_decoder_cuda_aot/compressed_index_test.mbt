///|
fn compressed_index_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DeterministicCompressedIndex(_) {
      return ordinal
    }
  }
  abort("deterministic compressed-index requirement missing")
}

///|
test "deterministic compressed index lowers with exact five-operand I32 ABI" {
  let requirements = aot_test_requirements(
    maximum_position_count=1024,
    deterministic_compression_ratio=128,
  )
  let ordinal = compressed_index_test_ordinal(requirements)
  let candidate = lower_deterministic_compressed_index_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), CausalCompleteSlotsI32)
  assert_eq(candidate.numerical_contract(), BitExactCausalCompleteSlotsI32V1)
  assert_eq(candidate.operation(), DeterministicCompressedIndex(128))
  assert_eq(candidate.capability(), DeterministicCompressedIndexAot)
  assert_eq(candidate.maximum_position_count(), 1024)
  assert_eq(candidate.maximum_slots_per_row(), 8)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 256)
  let operands = candidate.operands()
  assert_eq(operands.length(), 5)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[0].byte_count(), 20L)
  assert_eq(operands[1].role(), CompressedQueryPositionsInput)
  assert_eq(operands[1].byte_count(), 128L)
  assert_eq(operands[2].role(), CompressedCacheOffsetsInput)
  assert_eq(operands[2].byte_count(), 128L)
  assert_eq(operands[3].role(), CompressedIndexCountsOutput)
  assert_eq(operands[3].byte_count(), 128L)
  assert_eq(operands[4].role(), CompressedIndicesOutput)
  assert_eq(operands[4].byte_count(), 1024L)
  for operand in operands {
    assert_eq(operand.alignment(), 4L)
  }
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("selected = (position + 1) / 128"))
  assert_true(source.contains("position < 1024"))
  assert_true(source.contains("slot < 8"))
  let repeated = lower_deterministic_compressed_index_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
  let foreign = lower_deterministic_compressed_index_cuda_aot_candidate(
    aot_test_requirements(
      content_fill='c',
      maximum_position_count=1024,
      deterministic_compression_ratio=128,
    ),
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), foreign.source_digest())
  assert_true(candidate.recipe_digest() != foreign.recipe_digest())
  assert_true(candidate.source_identity() != foreign.source_identity())
  let other_target = lower_deterministic_compressed_index_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(bf16=false),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), other_target.source_digest())
  assert_true(candidate.recipe_digest() != other_target.recipe_digest())
}

///|
test "deterministic compressed index rejects ordinal compiler and row bounds" {
  let requirements = aot_test_requirements(deterministic_compression_ratio=128)
  let ordinal = compressed_index_test_ordinal(requirements)
  for hostile_ordinal in [ordinal - 1, ordinal + 1] {
    try
      lower_deterministic_compressed_index_cuda_aot_candidate(
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
      _ => fail("adjacent operation was lowered as deterministic index")
    }
  }
  try
    lower_deterministic_compressed_index_cuda_aot_candidate(
      requirements,
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
    _ => fail("reassociating compiler was admitted")
  }
  try
    lower_deterministic_compressed_index_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=4097,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected row-envelope rejection")
  } noraise {
    _ => fail("oversized row envelope was admitted")
  }
}

///|
test "deterministic compressed index rejects dense output byte overflow" {
  let requirements = aot_test_requirements(
    maximum_position_count=2147483647,
    deterministic_compression_ratio=128,
  )
  let ordinal = compressed_index_test_ordinal(requirements)
  try
    lower_deterministic_compressed_index_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=33,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    SizeOverflow => ()
    _ => fail("unexpected dense compressed-index overflow rejection")
  } noraise {
    _ => fail("overflowing dense compressed-index extent was admitted")
  }
}
