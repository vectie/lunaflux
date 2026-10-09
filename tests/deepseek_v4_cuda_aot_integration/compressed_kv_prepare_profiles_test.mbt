///|
fn official_compressed_kv_prepare_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is NonOverlappingCompressedKeyValuePrepareV1(128, 128) {
      return ordinal
    }
  }
  abort("official profile lost non-overlapping compressed KV prepare")
}

///|
test "all five official profiles lower exact inert compressed KV prepare" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 15),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 15),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 12),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 12),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 15),
  ]
  for profile in profiles {
    let (model_id, fill, expected_ordinal) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
      requirements,
      requirement_ordinal=official_compressed_kv_prepare_ordinal(requirements),
      maximum_batches=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      NonOverlappingCompressedKeyValuePrepareV1(128, 128),
    )
    assert_eq(candidate.requirement_ordinal(), expected_ordinal)
    assert_eq(candidate.head_dimension(), 512)
    assert_eq(candidate.maximum_position_count(), 1048576)
    assert_eq(candidate.maximum_compressed_slots(), 8192)
    assert_eq(candidate.maximum_key_slots(), 1056768)
    assert_eq(candidate.operands().length(), 7)
    assert_eq(candidate.operands()[3].byte_count(), 1073741824L)
    assert_eq(candidate.operands()[4].byte_count(), 8388608L)
    assert_eq(candidate.operands()[6].byte_count(), 1082130432L)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "official compressed KV prepare rejects adjacent operation" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", '6',
  )
  let ordinal = official_compressed_kv_prepare_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal + 1,
      maximum_batches=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal + 1)
    _ => fail("wrong adjacent-operation rejection")
  } noraise {
    _ => fail("sparse attention lowered as KV preparation")
  }
}
