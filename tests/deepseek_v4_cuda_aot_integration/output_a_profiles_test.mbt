///|
fn official_output_a_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputAProjectionV1(_, _, _, _) {
      return ordinal
    }
  }
  abort("official profile lost output-A projection")
}

///|
test "all five profiles bind converted BF16 grouped output-A exactly" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 18, 64, 8),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 18, 64, 8),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 17, 128, 16),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 17, 128, 16),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 18, 64, 8),
  ]
  let source_digests : Array[String] = []
  let recipe_digests : Array[String] = []
  for profile in profiles {
    let (model_id, fill, expected_ordinal, heads, groups) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=official_output_a_ordinal(requirements),
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.requirement_ordinal(), expected_ordinal)
    assert_eq(
      candidate.operation(),
      AttentionOutputAProjectionV1(1024, groups, heads, 512),
    )
    assert_eq(candidate.group_count(), groups)
    assert_eq(candidate.input_width_per_group(), 4096)
    assert_eq(candidate.output_rank(), 1024)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().grid_y(), groups)
    assert_eq(candidate.operands().map(op => op.byte_count()), [
      20L,
      2L * heads.to_int64() * 512L * 2L,
      groups.to_int64() * 1024L * 4096L * 2L,
      2L * groups.to_int64() * 1024L * 2L,
    ])
    assert_false(candidate.manifest_bindable())
    source_digests.push(candidate.source_digest().as_hex())
    recipe_digests.push(candidate.recipe_digest().as_hex())
    let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
      model_id,
    )
    let converted = @deepseek_v4_weights.output_a_converted_layout(spec)
    assert_eq(converted.converted_grouped_shape(), [
      groups.to_int64(),
      1024L,
      4096L,
    ])
  }
  assert_eq(source_digests[0], source_digests[1])
  assert_eq(source_digests[0], source_digests[4])
  assert_eq(source_digests[2], source_digests[3])
  assert_true(source_digests[0] != source_digests[2])
  inspect(
    source_digests[0],
    content="2f25319ab2b30b6785d81f20fe4643cccbb90fb20534fd1c2404658de9b3d8b7",
  )
  inspect(
    source_digests[2],
    content="c79688d5aba6b8a911092235e65b937af888a177f2a46344bf86d35d0e875011",
  )
  for index, digest in recipe_digests {
    for earlier in 0..<index {
      assert_true(digest != recipe_digests[earlier])
    }
  }
}

///|
test "output-A group order is isolated from the distinct Output-B ABI" {
  let values = try! @advanced_decoder_reference.grouped_output_a_projection_v1(
    [1.0F, 2.0F, 10.0F, 20.0F],
    [1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F, 0.0F, 3.0F],
    row_count=1,
    group_count=2,
    input_width=2,
    output_width=2,
    limits=try! @advanced_decoder_reference.ReferenceLimits::new(
      max_values=64,
      max_width=8,
      max_iterations=1,
    ),
  )
  assert_eq(values, [1.0F, 2.0F, 20.0F, 60.0F])
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  let ordinal = official_output_a_ordinal(requirements)
  assert_eq(
    requirements.requirements()[ordinal + 1].operation(),
    AttentionOutputBProjectionV1(1024, 8, 4096),
  )
  try
    @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=1,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, ordinal + 1)
    _ => fail("wrong output-B rejection")
  } noraise {
    _ => fail("output-B used converted output-A ABI")
  }
}

///|
test "official output-A rejects byte overflow before source construction" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'b',
  )
  try
    @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=official_output_a_ordinal(requirements),
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
    _ => fail("wrong output-A overflow rejection")
  } noraise {
    _ => fail("oversized output-A ABI admitted")
  }
}
