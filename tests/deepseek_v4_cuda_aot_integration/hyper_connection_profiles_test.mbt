///|
fn official_block_control_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is HyperConnectionBlockControlV1(_, _, _, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost mHC block control")
}

///|
fn hostile_reassociating_compiler() -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, 'e'),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=true,
    max_registers=128,
  )
}

///|
test "all official DeepSeek profiles bind exact mHC block-control geometry" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 4096, 1572864L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 4096, 1572864L),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 7168, 2752512L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 7168, 2752512L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 4096, 1572864L),
  ]
  let recipes : Array[String] = []
  for entry in cases {
    let (model_id, fill, hidden_width, function_bytes) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let ordinal = official_block_control_ordinal(requirements)
    assert_eq(ordinal, 6)
    let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      HyperConnectionBlockControlV1(4, 20, 0.000001, 0.000001),
    )
    assert_eq(candidate.capability(), HyperConnectionBlockControlV1Aot)
    assert_eq(candidate.stream_width(), 4)
    assert_eq(candidate.hidden_width(), hidden_width)
    assert_eq(candidate.sinkhorn_iterations(), 20)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 1)
    assert_eq(candidate.operands().length(), 8)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
    assert_eq(
      candidate.operands()[1].byte_count(),
      hidden_width.to_int64() * 16L,
    )
    assert_eq(
      candidate.operands()[2].role(),
      HyperConnectionFunctionWeightsInput,
    )
    assert_eq(candidate.operands()[2].byte_count(), function_bytes)
    assert_eq(candidate.operands()[3].role(), HyperConnectionBaseInput)
    assert_eq(candidate.operands()[3].byte_count(), 96L)
    assert_eq(candidate.operands()[4].role(), HyperConnectionScaleInput)
    assert_eq(candidate.operands()[4].byte_count(), 12L)
    assert_eq(candidate.operands()[5].role(), HyperConnectionPreOutput)
    assert_eq(candidate.operands()[5].byte_count(), 32L)
    assert_eq(candidate.operands()[6].role(), HyperConnectionPostOutput)
    assert_eq(candidate.operands()[6].byte_count(), 32L)
    assert_eq(candidate.operands()[7].role(), HyperConnectionCombinationOutput)
    assert_eq(candidate.operands()[7].byte_count(), 128L)
    assert_false(candidate.manifest_bindable())
    assert_false(recipes.contains(candidate.recipe_digest().as_hex()))
    recipes.push(candidate.recipe_digest().as_hex())
  }
}

///|
test "official mHC block control rejects adjacent ordinal compiler target and overflow" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", 'a',
  )
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(7) => ()
    _ => fail("wrong adjacent-operation rejection")
  } noraise {
    _ => fail("pre-reduction was lowered as block control")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=2,
      target=try! @catalog.DeviceTarget::new(
        compute_major=9,
        compute_minor=0,
        supports_bf16=false,
        supports_cublas_lt=true,
      ),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong non-BF16 target rejection")
  } noraise {
    _ => fail("non-BF16 target was admitted")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=hostile_reassociating_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong reassociating compiler rejection")
  } noraise {
    _ => fail("reassociating compiler was admitted")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
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
    _ => fail("wrong block-control operand overflow rejection")
  } noraise {
    _ => fail("oversized official block-control buffers were admitted")
  }
}

///|
test "all official DeepSeek profiles bind exact directional mHC pre-reduction" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '6', 4096, 16384L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '7', 4096, 16384L),
    ("deepseek-ai/DeepSeek-V4-Pro", '8', 7168, 28672L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '9', 7168, 28672L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 'a', 4096, 16384L),
  ]
  let recipes : Array[String] = []
  for entry in cases {
    let (model_id, fill, hidden_width, output_bytes) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_pre_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.operation(), HyperConnectionPreReduceV1(4))
    assert_eq(candidate.capability(), HyperConnectionPreReduceV1Aot)
    assert_eq(candidate.stream_width(), 4)
    assert_eq(candidate.hidden_width(), hidden_width)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().length(), 4)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
    assert_eq(
      candidate.operands()[1].byte_count(),
      hidden_width.to_int64() * 16L,
    )
    assert_eq(candidate.operands()[2].role(), HyperConnectionPreInput)
    assert_eq(candidate.operands()[2].byte_count(), 32L)
    assert_eq(candidate.operands()[3].role(), HyperConnectionReducedOutput)
    assert_eq(candidate.operands()[3].byte_count(), output_bytes)
    assert_false(candidate.manifest_bindable())
    assert_false(recipes.contains(candidate.recipe_digest().as_hex()))
    recipes.push(candidate.recipe_digest().as_hex())
  }
}

///|
test "official pre-reduction rejects post-combine and oversized row buffers" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", 'b',
  )
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_pre_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=8,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(8) => ()
    _ => fail("wrong post-combine rejection")
  } noraise {
    _ => fail("post-combine was lowered as pre-reduction")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_pre_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
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
    _ => fail("wrong pre-reduction buffer overflow rejection")
  } noraise {
    _ => fail("oversized pre-reduction buffers were admitted")
  }
}

///|
test "all official DeepSeek profiles bind exact mHC post-combination" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", 'c', 4096, 16384L, 65536L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 'd', 4096, 16384L, 65536L),
    ("deepseek-ai/DeepSeek-V4-Pro", 'e', 7168, 28672L, 114688L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 'f', 7168, 28672L, 114688L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '0', 4096, 16384L, 65536L),
  ]
  let recipes : Array[String] = []
  for entry in cases {
    let (model_id, fill, hidden_width, branch_bytes, state_bytes) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_post_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=8,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.operation(), HyperConnectionPostCombineV1(4))
    assert_eq(candidate.capability(), HyperConnectionPostCombineV1Aot)
    assert_eq(candidate.stream_width(), 4)
    assert_eq(candidate.hidden_width(), hidden_width)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[1].role(), HyperConnectionBranchInput)
    assert_eq(candidate.operands()[1].byte_count(), branch_bytes)
    assert_eq(candidate.operands()[2].role(), HyperConnectionStateInput)
    assert_eq(candidate.operands()[2].byte_count(), state_bytes)
    assert_eq(candidate.operands()[3].role(), HyperConnectionPostInput)
    assert_eq(candidate.operands()[3].byte_count(), 32L)
    assert_eq(candidate.operands()[4].role(), HyperConnectionCombinationInput)
    assert_eq(candidate.operands()[4].byte_count(), 128L)
    assert_eq(candidate.operands()[5].role(), HyperConnectionCombinedOutput)
    assert_eq(candidate.operands()[5].byte_count(), state_bytes)
    assert_false(candidate.manifest_bindable())
    assert_false(recipes.contains(candidate.recipe_digest().as_hex()))
    recipes.push(candidate.recipe_digest().as_hex())
  }
}

///|
test "official post-combination rejects head reduction and oversized buffers" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", '1',
  )
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_post_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(9) => ()
    _ => fail("wrong head-reduction rejection")
  } noraise {
    _ => fail("head reduction was lowered as post-combination")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_post_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=8,
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
    _ => fail("wrong post-combination buffer overflow rejection")
  } noraise {
    _ => fail("oversized post-combination buffers were admitted")
  }
}
