///|
fn official_deterministic_index_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DeterministicCompressedIndex(128) {
      return ordinal
    }
  }
  abort("official profile lost deterministic compressed-index requirement")
}

///|
test "all five official profiles lower the exact inert deterministic index" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", 'a'),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 'b'),
    ("deepseek-ai/DeepSeek-V4-Pro", 'c'),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'd'),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'e'),
  ]
  for profile in profiles {
    let (model_id, fill) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_deterministic_index_ordinal(requirements)
    let candidate = @advanced_decoder_cuda_aot.lower_deterministic_compressed_index_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.operation(), DeterministicCompressedIndex(128))
    assert_eq(candidate.capability(), DeterministicCompressedIndexAot)
    assert_eq(candidate.compression_ratio(), 128)
    assert_eq(
      candidate.maximum_position_count(),
      requirements.geometry().maximum_position_count(),
    )
    assert_eq(
      candidate.maximum_slots_per_row(),
      requirements.geometry().maximum_position_count() / 128,
    )
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[1].role(), CompressedQueryPositionsInput)
    assert_eq(candidate.operands()[2].role(), CompressedCacheOffsetsInput)
    assert_eq(candidate.operands()[3].role(), CompressedIndexCountsOutput)
    assert_eq(candidate.operands()[4].role(), CompressedIndicesOutput)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "Flash-0731 deterministic index rejects wrong ordinal and compiler" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'f',
  )
  let ordinal = official_deterministic_index_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_deterministic_compressed_index_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal - 1)
    _ => fail("unexpected official-profile ordinal rejection")
  } noraise {
    _ => fail("adjacent official operation was lowered as deterministic index")
  }
}
