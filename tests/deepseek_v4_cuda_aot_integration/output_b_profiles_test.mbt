///|
fn official_output_b_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputBProjectionV1(_, _, _) {
      return ordinal
    }
  }
  abort("official profile lost Output-B projection")
}

///|
test "all five profiles bind exact dense FP8 Output-B ABI" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 19, 8192, 4096, 33554432L, 2048L),
    (
      "deepseek-ai/DeepSeek-V4-Flash-Base", '2', 19, 8192, 4096, 33554432L, 2048L,
    ),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 18, 16384, 7168, 117440512L, 7168L),
    (
      "deepseek-ai/DeepSeek-V4-Pro-Base", '4', 18, 16384, 7168, 117440512L, 7168L,
    ),
    (
      "deepseek-ai/DeepSeek-V4-Flash-0731", '5', 19, 8192, 4096, 33554432L, 2048L,
    ),
  ]
  for profile in profiles {
    let (model_id, fill, ordinal, input_width, hidden, weight_bytes, scale_bytes
    ) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_attention_output_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=official_output_b_ordinal(requirements),
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.requirement_ordinal(), ordinal)
    assert_eq(candidate.input_width(), input_width)
    assert_eq(candidate.output_width(), hidden)
    assert_eq(candidate.block_width(), 128)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().grid_y(), hidden / 256)
    assert_eq(candidate.operands().map(op => op.role()), [
      StepCountsInput,
      AttentionOutputRankInput,
      AttentionOutputBWeightCodesInput,
      AttentionOutputBWeightScalesInput,
      HiddenOutput,
    ])
    assert_eq(candidate.operands().map(op => op.byte_count()), [
      20L,
      2L * input_width.to_int64() * 2L,
      weight_bytes,
      scale_bytes,
      2L * hidden.to_int64() * 2L,
    ])
    assert_eq(candidate.tensor_parallel_world_size(), 1)
    assert_false(candidate.owns_collective())
    assert_false(candidate.manifest_bindable())
    if model_id == "deepseek-ai/DeepSeek-V4-Flash-0731" {
      inspect(
        candidate.source_digest().as_hex(),
        content="3d24979ba6cb8e04e16211731eb641837b031623c70eb10404ef0cd9a2c26a2c",
      )
    } else if model_id == "deepseek-ai/DeepSeek-V4-Pro" {
      inspect(
        candidate.source_digest().as_hex(),
        content="63075862b3009b8c786a1d92d3cf48fd1de3129fb767ae93bf6dd0d35ba9c5a7",
      )
    }
  }
}

///|
test "official Output-B rejects wrong ordinal target and row overflow" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  let ordinal = official_output_b_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_attention_output_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal - 1)
    _ => fail("wrong ordinal failure")
  } noraise {
    _ => fail("Output-A ordinal admitted")
  }
  try
    @advanced_decoder_cuda_aot.lower_attention_output_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1048576,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
        maximum_tokens=1048576,
        maximum_source_bytes=65536,
        maximum_recipe_bytes=65536,
      ),
    )
  catch {
    SizeOverflow => ()
    _ => fail("wrong overflow failure")
  } noraise {
    _ => fail("oversized ABI admitted")
  }
}
