///|
fn official_deepseek_requirements(
  model_id : StringView,
  fill : Char,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements raise {
  let profile = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    model_id,
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    profile,
    @spec.ContentDigest::from_sha256(fill.to_string().repeat(64)),
  )
  @deepseek_v4_kernel_requirements.project(@deepseek_v4.build(metadata))
}

///|
fn official_deepseek_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn official_deepseek_compiler() -> @luna_specialization.LunaCompilerPolicy {
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'c'),
    major=12,
    minor=9,
    patch=1,
  )
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity~,
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn official_deepseek_limits() -> @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits {
  try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
fn official_expert_score_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is ExpertScoreTransform(_, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost expert score transform")
}

///|
fn official_query_key_norm_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is QueryKeyRmsNorm(_, _, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost Q/K RMSNorm")
}

///|
fn official_scaled_rotary_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is RotaryPositionEncoding(_, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost scaled rotary encoding")
}

///|
fn official_token_hash_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is TokenHashRouting(_, _, _, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost token-hash routing")
}

///|
test "all official DeepSeek V4 profiles bind exact I32 token-hash geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 21, 256),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 21, 256),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 20, 384),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 20, 384),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 21, 256),
  ]
  let recipe_digests : Array[String] = []
  let source_digests : Array[String] = []
  for entry in cases {
    let (model_id, fill, expected_ordinal, routed_expert_count) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_token_hash_ordinal(requirements)
    assert_eq(ordinal, expected_ordinal)
    let candidate = @advanced_decoder_cuda_aot.lower_token_hash_routing_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      TokenHashRouting(3, 129280, 6, routed_expert_count),
    )
    assert_eq(candidate.capability(), TokenHashRoutingAot)
    assert_eq(candidate.routed_expert_count(), routed_expert_count)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.dimensions().shared_memory_bytes(), 0)
    assert_eq(candidate.operands().length(), 4)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[0].alignment(), 4L)
    assert_eq(candidate.operands()[1].role(), TokenIdsInput)
    assert_eq(candidate.operands()[1].byte_count(), 8L)
    assert_eq(candidate.operands()[1].alignment(), 4L)
    assert_eq(candidate.operands()[2].role(), TokenHashTableInput)
    assert_eq(candidate.operands()[2].byte_count(), 3102720L)
    assert_eq(candidate.operands()[2].alignment(), 4L)
    assert_eq(candidate.operands()[3].role(), RoutingIndicesOutput)
    assert_eq(candidate.operands()[3].byte_count(), 48L)
    assert_eq(candidate.operands()[3].alignment(), 4L)
    assert_eq(candidate.source_identity(), requirements.source_identity())
    assert_eq(candidate.requirements_digest(), requirements.digest())
    assert_false(candidate.manifest_bindable())
    assert_false(recipe_digests.contains(candidate.recipe_digest().as_hex()))
    recipe_digests.push(candidate.recipe_digest().as_hex())
    if !source_digests.contains(candidate.source_digest().as_hex()) {
      source_digests.push(candidate.source_digest().as_hex())
    }
  }
  assert_eq(recipe_digests.length(), 5)
  assert_eq(source_digests.length(), 2)
}

///|
test "official DeepSeek token-hash lowering rejects adjacent ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", '6',
  )
  let ordinal = official_token_hash_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_token_hash_routing_cuda_aot_candidate(
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
    _ => fail("adjacent DeepSeek operation was lowered as token hash")
  }
}

///|
test "all official DeepSeek V4 profiles bind exact independent Q/K RMSNorm geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '6', 1024),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '7', 1024),
    ("deepseek-ai/DeepSeek-V4-Pro", '8', 1536),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '9', 1536),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '0', 1024),
  ]
  let recipe_digests : Array[String] = []
  let source_digests : Array[String] = []
  for entry in cases {
    let (model_id, fill, query_width) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_query_key_norm_ordinal(requirements)
    assert_eq(ordinal, 3)
    let candidate = @advanced_decoder_cuda_aot.lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      QueryKeyRmsNorm(0.000001, query_width, 512),
    )
    assert_eq(candidate.capability(), QueryKeyRmsNormAot)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().grid_y(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.dimensions().shared_memory_bytes(), 1024)
    assert_eq(candidate.operands().length(), 7)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[0].alignment(), 4L)
    assert_eq(candidate.operands()[1].role(), QueryInput)
    assert_eq(candidate.operands()[1].byte_count(), query_width.to_int64() * 4L)
    assert_eq(candidate.operands()[2].role(), QueryNormWeights)
    assert_eq(candidate.operands()[2].byte_count(), query_width.to_int64() * 2L)
    assert_eq(candidate.operands()[3].role(), QueryOutput)
    assert_eq(candidate.operands()[3].byte_count(), query_width.to_int64() * 4L)
    assert_eq(candidate.operands()[4].role(), KeyValueInput)
    assert_eq(candidate.operands()[4].byte_count(), 2048L)
    assert_eq(candidate.operands()[5].role(), KeyValueNormWeights)
    assert_eq(candidate.operands()[5].byte_count(), 1024L)
    assert_eq(candidate.operands()[6].role(), KeyValueOutput)
    assert_eq(candidate.operands()[6].byte_count(), 2048L)
    assert_eq(candidate.source_identity(), requirements.source_identity())
    assert_eq(candidate.requirements_digest(), requirements.digest())
    assert_false(candidate.manifest_bindable())
    assert_false(recipe_digests.contains(candidate.recipe_digest().as_hex()))
    recipe_digests.push(candidate.recipe_digest().as_hex())
    if !source_digests.contains(candidate.source_digest().as_hex()) {
      source_digests.push(candidate.source_digest().as_hex())
    }
  }
  assert_eq(recipe_digests.length(), 5)
  assert_eq(source_digests.length(), 2)
}

///|
test "official DeepSeek Q/K RMSNorm lowering rejects adjacent ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", '6',
  )
  let ordinal = official_query_key_norm_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_foundation_cuda_aot_candidate(
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
    _ => fail("adjacent DeepSeek operation was lowered as Q/K RMSNorm")
  }
}

///|
test "all official DeepSeek V4 profiles bind exact LM-head geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 31, 4096, 1059061760L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 31, 4096, 1059061760L),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 30, 7168, 1853358080L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 30, 7168, 1853358080L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 39, 4096, 1059061760L),
  ]
  let recipe_digests : Array[String] = []
  for entry in cases {
    let (model_id, fill, expected_ordinal, hidden, weight_bytes) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = requirements.requirements().length() - 1
    assert_eq(ordinal, expected_ordinal)
    let candidate = @advanced_decoder_cuda_aot.lower_language_model_head_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.operation(), LanguageModelHead)
    assert_eq(candidate.capability(), DenseProjectionAot)
    assert_eq(candidate.input_width(), hidden)
    assert_eq(candidate.output_width(), 129280)
    assert_eq(candidate.dimensions().grid_x(), 1010)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[0].alignment(), 4L)
    assert_eq(candidate.operands()[1].byte_count(), hidden.to_int64() * 4L)
    assert_eq(candidate.operands()[2].byte_count(), weight_bytes)
    assert_eq(candidate.operands()[3].byte_count(), 517120L)
    assert_eq(candidate.source_identity(), requirements.source_identity())
    assert_eq(candidate.requirements_digest(), requirements.digest())
    assert_false(candidate.manifest_bindable())
    assert_false(recipe_digests.contains(candidate.recipe_digest().as_hex()))
    recipe_digests.push(candidate.recipe_digest().as_hex())
  }
  assert_eq(recipe_digests.length(), 5)
}

///|
test "official vocabulary output capacity fails before operand truncation" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash", '6',
  )
  let limits = try! @advanced_decoder_cuda_aot.AdvancedDecoderCudaAotLimits::new(
    maximum_tokens=9000,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @advanced_decoder_cuda_aot.lower_language_model_head_cuda_aot_candidate(
      requirements,
      requirement_ordinal=requirements.requirements().length() - 1,
      maximum_tokens=8306,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits~,
    )
  catch {
    SizeOverflow => ()
    _ => fail("unexpected oversized-logits rejection")
  } noraise {
    _ => fail("oversized DeepSeek logits operand was admitted")
  }
}

///|
test "all official DeepSeek V4 profiles bind exact expert-score geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", 'a', 256),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 'b', 256),
    ("deepseek-ai/DeepSeek-V4-Pro", 'c', 384),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'd', 384),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'e', 256),
  ]
  let recipe_digests : Array[String] = []
  let source_digests : Array[String] = []
  for entry in cases {
    let (model_id, fill, routed_expert_count) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_expert_score_ordinal(requirements)
    let candidate = @advanced_decoder_cuda_aot.lower_expert_score_transform_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      ExpertScoreTransform(SqrtSoftplus, routed_expert_count),
    )
    assert_eq(candidate.capability(), ExpertScoreTransformAot)
    assert_eq(candidate.routed_expert_count(), routed_expert_count)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().length(), 3)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[0].alignment(), 4L)
    assert_eq(candidate.operands()[1].role(), RoutingScoresInput)
    assert_eq(
      candidate.operands()[1].byte_count(),
      routed_expert_count.to_int64() * 8L,
    )
    assert_eq(candidate.operands()[2].role(), RoutingScoresOutput)
    assert_eq(
      candidate.operands()[2].byte_count(),
      routed_expert_count.to_int64() * 8L,
    )
    assert_eq(candidate.source_identity(), requirements.source_identity())
    assert_eq(candidate.requirements_digest(), requirements.digest())
    assert_false(candidate.manifest_bindable())
    assert_false(recipe_digests.contains(candidate.recipe_digest().as_hex()))
    recipe_digests.push(candidate.recipe_digest().as_hex())
    if !source_digests.contains(candidate.source_digest().as_hex()) {
      source_digests.push(candidate.source_digest().as_hex())
    }
  }
  assert_eq(recipe_digests.length(), 5)
  assert_eq(source_digests.length(), 2)
}

///|
test "official DeepSeek expert-score lowering rejects adjacent ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", 'f',
  )
  let ordinal = official_expert_score_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_expert_score_transform_cuda_aot_candidate(
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
    _ => fail("adjacent DeepSeek operation was lowered as expert scores")
  }
}
