///|
test "head reduction lowers exact six-operand normalized-control ABI" {
  let requirements = aot_test_requirements()
  assert_true(
    requirements.requirements()[9].operation()
    is HyperConnectionHeadReduceV1(2, 0.000001, 0.000001),
  )
  let candidate = lower_hyper_connection_head_reduce_cuda_aot_candidate(
    requirements,
    requirement_ordinal=9,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), HeadReduceBf16F32V1)
  assert_eq(
    candidate.numerical_contract(),
    OrderedF32HeadControlReductionBf16RoundV1,
  )
  assert_eq(
    candidate.operation(),
    HyperConnectionHeadReduceV1(2, 0.000001, 0.000001),
  )
  assert_eq(candidate.capability(), HyperConnectionHeadReduceV1Aot)
  assert_eq(candidate.stream_width(), 2)
  assert_eq(candidate.hidden_width(), 16)
  assert_eq(candidate.dimensions().grid_x(), 2)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[2].role(), HyperConnectionFunctionWeightsInput)
  assert_eq(candidate.operands()[2].byte_count(), 256L)
  assert_eq(candidate.operands()[3].role(), HyperConnectionBaseInput)
  assert_eq(candidate.operands()[3].byte_count(), 8L)
  assert_eq(candidate.operands()[4].role(), HyperConnectionScaleInput)
  assert_eq(candidate.operands()[4].byte_count(), 4L)
  assert_eq(candidate.operands()[4].alignment(), 4L)
  assert_eq(candidate.operands()[5].role(), HyperConnectionReducedOutput)
  assert_eq(candidate.operands()[5].byte_count(), 64L)
  assert_false(candidate.manifest_bindable())
  assert_true(
    @utf8.decode(candidate.recipe_bytes()).contains(
      "ordered-f32-normalized-sigmoid-control-reduction-single-bf16-rne-v1",
    ),
  )
  let repeated = lower_hyper_connection_head_reduce_cuda_aot_candidate(
    requirements,
    requirement_ordinal=9,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "head reduction rejects adjacent operations target compiler and row bounds" {
  let requirements = aot_test_requirements()
  for ordinal in [8, 10] {
    try
      lower_hyper_connection_head_reduce_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=2,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong adjacent operation rejection")
    } noraise {
      _ => fail("another operation was lowered as head reduction")
    }
  }
  try
    lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=2,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong non-BF16 target rejection")
  } noraise {
    _ => fail("head reduction admitted a non-BF16 target")
  }
  try
    lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=2,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong reassociating compiler rejection")
  } noraise {
    _ => fail("head reduction admitted reassociation")
  }
  try
    lower_hyper_connection_head_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=9,
      maximum_tokens=3,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=try! AdvancedDecoderCudaAotLimits::new(
        maximum_tokens=2,
        maximum_source_bytes=65536,
        maximum_recipe_bytes=65536,
      ),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong live-row bound rejection")
  } noraise {
    _ => fail("head reduction exceeded its row limit")
  }
}
