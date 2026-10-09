///|
test "all five profiles bind exact replicated key/value projection ABI" {
  let profiles = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 4096, 2097152L, 128L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 4096, 2097152L, 128L),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 7168, 3670016L, 224L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 7168, 3670016L, 224L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 4096, 2097152L, 128L),
  ]
  let source_digests : Array[String] = []
  for profile in profiles {
    let (model_id, fill, hidden, weight_bytes, scale_bytes) = profile
    let requirements = official_deepseek_requirements(model_id, fill)
    guard requirements.requirements()[2].operation()
      is KeyValueProjection(heads, head_dimension, rope_dimension) else {
      fail("official ordinal 2 lost key/value projection")
    }
    assert_eq(heads, 1)
    assert_eq(head_dimension, 512)
    assert_eq(rope_dimension, 64)
    let candidate = @advanced_decoder_cuda_aot.lower_key_value_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=2,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.input_width(), hidden)
    assert_eq(candidate.output_width(), 512)
    assert_eq(candidate.rope_width(), 64)
    assert_eq(candidate.block_width(), 128)
    assert_true(candidate.tensor_parallel_replicated())
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().grid_y(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().map(op => op.role()), [
      StepCountsInput,
      HiddenInput,
      KeyValueWeightCodesInput,
      KeyValueWeightScalesInput,
      KeyValueProjectionOutput,
    ])
    assert_eq(candidate.operands().map(op => op.byte_count()), [
      20L,
      2L * hidden.to_int64() * 2L,
      weight_bytes,
      scale_bytes,
      2048L,
    ])
    assert_eq(candidate.operands().map(op => op.alignment()), [
      4L, 16L, 16L, 16L, 16L,
    ])
    assert_false(candidate.manifest_bindable())
    if !source_digests.contains(candidate.source_digest().as_hex()) {
      source_digests.push(candidate.source_digest().as_hex())
    }
    guard requirements.requirements()[3].operation()
      is QueryKeyRmsNorm(_, _, 512) else {
      fail("key/value RMSNorm was absorbed into projection")
    }
  }
  assert_eq(source_digests.length(), 2)
}

///|
test "official key/value projection source identities stay exact" {
  let flash = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  let pro = official_deepseek_requirements("deepseek-ai/DeepSeek-V4-Pro", 'b')
  let flash_candidate = @advanced_decoder_cuda_aot.lower_key_value_projection_cuda_aot_candidate(
    flash,
    requirement_ordinal=2,
    maximum_tokens=2,
    target=official_deepseek_target(),
    compiler=official_deepseek_compiler(),
    limits=official_deepseek_limits(),
  )
  let pro_candidate = @advanced_decoder_cuda_aot.lower_key_value_projection_cuda_aot_candidate(
    pro,
    requirement_ordinal=2,
    maximum_tokens=2,
    target=official_deepseek_target(),
    compiler=official_deepseek_compiler(),
    limits=official_deepseek_limits(),
  )
  inspect(
    flash_candidate.source_digest().as_hex(),
    content="1e632b2e89e0142c731eae2e432a2098574a2d5e1962eeb4b2e72c9bc49d5341",
  )
  inspect(
    flash_candidate.recipe_digest().as_hex(),
    content="cb0c6b8df5063656527101acc01082233cbb02735e8de00fd924abe02a6cad15",
  )
  inspect(
    pro_candidate.source_digest().as_hex(),
    content="6db3d4fa9e76b109999ddaf8da86a078dba95f9a08075532d55db3caf06ce1f5",
  )
  assert_false(
    flash_candidate.recipe_digest().as_hex() ==
    pro_candidate.recipe_digest().as_hex(),
  )
}
