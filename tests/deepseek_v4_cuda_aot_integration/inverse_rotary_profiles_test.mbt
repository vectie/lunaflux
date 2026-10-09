///|
fn official_inverse_rotary_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is AttentionOutputInverseRotaryV1(160000.0, 65536, 16.0, 32.0, 1.0) {
      return ordinal
    }
  }
  abort("official profile lost attention-output inverse RoPE")
}

///|
test "all five official profiles lower exact inert output inverse RoPE" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", '7', 17, 64),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '8', 17, 64),
    ("deepseek-ai/DeepSeek-V4-Pro", '9', 14, 128),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'a', 14, 128),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'b', 17, 64),
  ]
  for profile in profiles {
    let (model_id, fill, expected_ordinal, expected_heads) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_attention_output_inverse_rotary_cuda_aot_candidate(
      requirements,
      requirement_ordinal=official_inverse_rotary_ordinal(requirements),
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.requirement_ordinal(), expected_ordinal)
    assert_eq(candidate.head_count(), expected_heads)
    assert_eq(candidate.head_dimension(), 512)
    assert_eq(candidate.rope_dimension(), 64)
    assert_eq(candidate.maximum_positions(), 1048576)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().length(), 3)
    assert_eq(
      candidate.operands()[2].byte_count(),
      2L * expected_heads.to_int64() * 512L * 2L,
    )
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "official inverse RoPE rejects sparse-attention ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'c',
  )
  let ordinal = official_inverse_rotary_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_attention_output_inverse_rotary_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal - 1)
    _ => fail("wrong preceding-operation rejection")
  } noraise {
    _ => fail("sparse attention lowered as inverse RoPE")
  }
}
