///|
test "all official DeepSeek profiles bind exact mHC head reduction" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 4096, 65536L, 262144L, 16384L),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 4096, 65536L, 262144L, 16384L),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 7168, 114688L, 458752L, 28672L),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 7168, 114688L, 458752L, 28672L),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 4096, 65536L, 262144L, 16384L),
  ]
  let recipes : Array[String] = []
  for entry in cases {
    let (model_id, fill, hidden_width, state_bytes, function_bytes, output_bytes
    ) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      candidate.operation(),
      HyperConnectionHeadReduceV1(4, 0.000001, 0.000001),
    )
    assert_eq(candidate.capability(), HyperConnectionHeadReduceV1Aot)
    assert_eq(candidate.stream_width(), 4)
    assert_eq(candidate.hidden_width(), hidden_width)
    assert_eq(candidate.dimensions().grid_x(), 2)
    assert_eq(candidate.dimensions().block_x(), 256)
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].byte_count(), 20L)
    assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
    assert_eq(candidate.operands()[1].byte_count(), state_bytes)
    assert_eq(
      candidate.operands()[2].role(),
      HyperConnectionFunctionWeightsInput,
    )
    assert_eq(candidate.operands()[2].byte_count(), function_bytes)
    assert_eq(candidate.operands()[3].role(), HyperConnectionBaseInput)
    assert_eq(candidate.operands()[3].byte_count(), 16L)
    assert_eq(candidate.operands()[4].role(), HyperConnectionScaleInput)
    assert_eq(candidate.operands()[4].byte_count(), 4L)
    assert_eq(candidate.operands()[4].alignment(), 4L)
    assert_eq(candidate.operands()[5].role(), HyperConnectionReducedOutput)
    assert_eq(candidate.operands()[5].byte_count(), output_bytes)
    assert_false(candidate.manifest_bindable())
    assert_false(recipes.contains(candidate.recipe_digest().as_hex()))
    recipes.push(candidate.recipe_digest().as_hex())
  }
}

///|
test "official head reduction rejects adjacent ordinals compiler target and overflow" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", 'a',
  )
  for ordinal in [8, 10] {
    try
      @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=2,
        target=official_deepseek_target(),
        compiler=official_deepseek_compiler(),
        limits=official_deepseek_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong adjacent-operation rejection")
    } noraise {
      _ => fail("adjacent operation was lowered as head reduction")
    }
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=hostile_reassociating_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("head reduction admitted reassociation")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
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
    _ => fail("wrong target rejection")
  } noraise {
    _ => fail("head reduction admitted a non-BF16 target")
  }
  try
    @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
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
    _ => fail("wrong head-reduction buffer overflow rejection")
  } noraise {
    _ => fail("oversized head-reduction buffers were admitted")
  }
}
