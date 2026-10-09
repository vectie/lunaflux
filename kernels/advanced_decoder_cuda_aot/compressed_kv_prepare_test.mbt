///|
fn compressed_kv_prepare_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is NonOverlappingCompressedKeyValuePrepareV1(_, _) {
      return ordinal
    }
  }
  abort("compressed KV prepare requirement missing")
}

///|
test "compressed KV prepare lowers exact seven-operand BF16 ABI" {
  let requirements = aot_test_requirements(
    maximum_position_count=1024,
    deterministic_compression_ratio=128,
  )
  let ordinal = compressed_kv_prepare_test_ordinal(requirements)
  let candidate = lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_batches=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(
    candidate.operation(),
    NonOverlappingCompressedKeyValuePrepareV1(64, 128),
  )
  assert_eq(candidate.family(), NonOverlappingSharedKeyValuePrepareBf16V1)
  assert_eq(candidate.numerical_contract(), BitExactBf16SegmentCopyV1)
  assert_eq(candidate.maximum_compressed_slots(), 8)
  assert_eq(candidate.maximum_key_slots(), 1032)
  let operands = candidate.operands()
  assert_eq(operands.length(), 7)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[1].role(), PreparedKeyValueModesInput)
  assert_eq(operands[2].role(), PreparedKeyValueSequenceLengthsInput)
  assert_eq(operands[3].role(), UncompressedKeyValueInput)
  assert_eq(operands[3].byte_count(), 32768L)
  assert_eq(operands[4].role(), CompressedKeyValueInput)
  assert_eq(operands[4].byte_count(), 256L)
  assert_eq(operands[5].role(), PreparedKeyValueLengthsOutput)
  assert_eq(operands[6].role(), PreparedKeyValueOutput)
  assert_eq(operands[6].byte_count(), 33024L)
  assert_false(candidate.manifest_bindable())
  assert_true(
    @utf8.decode(candidate.source_bytes()).contains("ordinary_slots + slot"),
  )
  let replay = lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_batches=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), replay.source_digest())
  assert_eq(candidate.recipe_digest(), replay.recipe_digest())
  let other_requirements = aot_test_requirements(
    content_fill='c',
    maximum_position_count=1024,
    deterministic_compression_ratio=128,
  )
  let other = lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
    other_requirements,
    requirement_ordinal=compressed_kv_prepare_test_ordinal(other_requirements),
    maximum_batches=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), other.source_digest())
  assert_false(candidate.recipe_digest() == other.recipe_digest())
  assert_false(candidate.source_identity() == other.source_identity())
}

///|
test "compressed KV prepare rejects ordinal compiler target and overflow" {
  let requirements = aot_test_requirements(deterministic_compression_ratio=128)
  let ordinal = compressed_kv_prepare_test_ordinal(requirements)
  for hostile in [ordinal - 1, ordinal + 1] {
    try
      lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
        requirements,
        requirement_ordinal=hostile,
        maximum_batches=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, hostile)
      _ => fail("wrong ordinal failure")
    } noraise {
      _ => fail("adjacent operation lowered")
    }
  }
  try
    lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_batches=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler failure")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
  try
    lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_batches=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target failure")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
}
