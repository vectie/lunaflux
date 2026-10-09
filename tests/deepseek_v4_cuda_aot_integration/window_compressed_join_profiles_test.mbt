///|
fn official_window_join_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is DeterministicWindowCompressedIndexJoinV1(128, 128) {
      return ordinal
    }
  }
  abort("official profile lost deterministic window/compressed join")
}

///|
test "all five official profiles lower exact inert window-compressed join" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", 'a', 14),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 'b', 14),
    ("deepseek-ai/DeepSeek-V4-Pro", 'c', 11),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'd', 11),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'e', 14),
  ]
  for profile in profiles {
    let (model_id, fill, expected_ordinal) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_window_join_ordinal(requirements)
    let candidate = @advanced_decoder_cuda_aot.lower_deterministic_window_compressed_join_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      DeterministicWindowCompressedIndexJoinV1(128, 128),
    )
    assert_eq(
      candidate.capability(),
      DeterministicWindowCompressedIndexJoinV1Aot,
    )
    assert_eq(candidate.maximum_position_count(), 1048576)
    assert_eq(candidate.maximum_compressed_width(), 8192)
    assert_eq(candidate.maximum_joined_width(), 8320)
    assert_eq(candidate.requirement_ordinal(), expected_ordinal)
    assert_eq(candidate.operands().length(), 8)
    assert_eq(candidate.operands()[5].byte_count(), 65536L)
    assert_eq(candidate.operands()[7].byte_count(), 66560L)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "official window-compressed join rejects adjacent ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'f',
  )
  let ordinal = official_window_join_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_deterministic_window_compressed_join_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal + 1)
    _ => fail("unexpected adjacent-operation rejection")
  } noraise {
    _ => fail("KV-preparation gap lowered as index join")
  }
}
