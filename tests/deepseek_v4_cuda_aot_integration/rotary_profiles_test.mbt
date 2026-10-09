///|
test "all official DeepSeek profiles bind exact compressed YaRN RoPE slices" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 64),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 64),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 128),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 128),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 64),
  ]
  let recipes : Array[String] = []
  let sources : Array[String] = []
  for entry in cases {
    let (model_id, fill, query_heads) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_scaled_rotary_ordinal(requirements)
    assert_eq(ordinal, 5)
    let candidate = @advanced_decoder_cuda_aot.lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      RotaryPositionEncoding(10000.0, 160000.0, 65536, 16.0, 32.0, 1.0),
    )
    assert_eq(candidate.capability(), ScaledRotaryPositionAot)
    assert_eq(candidate.query_head_count(), query_heads)
    assert_eq(candidate.key_value_head_count(), 1)
    assert_eq(candidate.rope_dimension(), 64)
    assert_eq(candidate.maximum_positions(), 1048576)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().grid_y(), query_heads + 1)
    assert_eq(candidate.dimensions().block_x(), 32)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[1].byte_count(), 8L)
    assert_eq(
      candidate.operands()[2].byte_count(),
      query_heads.to_int64() * 256L,
    )
    assert_eq(
      candidate.operands()[3].byte_count(),
      query_heads.to_int64() * 256L,
    )
    assert_eq(candidate.operands()[4].byte_count(), 256L)
    assert_eq(candidate.operands()[5].byte_count(), 256L)
    assert_eq(candidate.source_identity(), requirements.source_identity())
    assert_eq(candidate.requirements_digest(), requirements.digest())
    assert_false(candidate.manifest_bindable())
    recipes.push(candidate.recipe_digest().as_hex())
    if !sources.contains(candidate.source_digest().as_hex()) {
      sources.push(candidate.source_digest().as_hex())
    }
  }
  assert_eq(recipes.length(), 5)
  assert_eq(sources.length(), 2)
}

///|
test "official scaled RoPE rejects wrong ordinal and hostile compiler identity" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", '6',
  )
  try
    @advanced_decoder_cuda_aot.lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(6) => ()
    _ => fail("wrong ordinal rejection")
  } noraise {
    _ => fail("wrong rotary ordinal was admitted")
  }
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'f'),
    major=12,
    minor=9,
    patch=1,
  )
  let hostile = try! @luna_specialization.LunaCompilerPolicy::new(
    identity~,
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=true,
    max_registers=128,
  )
  try
    @advanced_decoder_cuda_aot.lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=5,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=hostile,
      limits=official_deepseek_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("reassociating rotary compiler was admitted")
  }
}
