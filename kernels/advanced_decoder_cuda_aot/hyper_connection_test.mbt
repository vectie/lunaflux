///|
test "block control lowers with exact eight-operand deterministic ABI" {
  let requirements = aot_test_requirements()
  assert_true(
    requirements.requirements()[6].operation()
    is HyperConnectionBlockControlV1(2, 4, 0.000001, 0.000001),
  )
  let candidate = lower_hyper_connection_block_control_cuda_aot_candidate(
    requirements,
    requirement_ordinal=6,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), BlockControlBf16F32V1)
  assert_eq(
    candidate.numerical_contract(),
    OrderedF32ControlProjectionSinkhornV1,
  )
  assert_eq(candidate.stream_width(), 2)
  assert_eq(candidate.hidden_width(), 16)
  assert_eq(candidate.sinkhorn_iterations(), 4)
  assert_eq(candidate.dimensions().grid_x(), 2)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.operands().length(), 8)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[2].role(), HyperConnectionFunctionWeightsInput)
  assert_eq(candidate.operands()[2].byte_count(), 1536L)
  assert_eq(candidate.operands()[3].role(), HyperConnectionBaseInput)
  assert_eq(candidate.operands()[3].byte_count(), 48L)
  assert_eq(candidate.operands()[4].role(), HyperConnectionScaleInput)
  assert_eq(candidate.operands()[4].byte_count(), 12L)
  assert_eq(candidate.operands()[4].alignment(), 4L)
  assert_eq(candidate.operands()[5].role(), HyperConnectionPreOutput)
  assert_eq(candidate.operands()[5].byte_count(), 16L)
  assert_eq(candidate.operands()[6].role(), HyperConnectionPostOutput)
  assert_eq(candidate.operands()[6].byte_count(), 16L)
  assert_eq(candidate.operands()[7].role(), HyperConnectionCombinationOutput)
  assert_eq(candidate.operands()[7].byte_count(), 32L)
  assert_false(candidate.manifest_bindable())
  assert_eq(candidate.source_digest().as_hex().length(), 64)
  assert_eq(candidate.recipe_digest().as_hex().length(), 64)
  let repeated = lower_hyper_connection_block_control_cuda_aot_candidate(
    requirements,
    requirement_ordinal=6,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "block control rejects adjacent phases target compiler and row bounds" {
  let requirements = aot_test_requirements()
  try
    lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
      maximum_tokens=2,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(7) => ()
    _ => fail("wrong adjacent-phase rejection")
  } noraise {
    _ => fail("pre-reduction was lowered as block control")
  }
  try
    lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=2,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target rejection")
  } noraise {
    _ => fail("non-BF16 target was admitted")
  }
  try
    lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=2,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("reassociating compiler was admitted")
  }
  try
    lower_hyper_connection_block_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=6,
      maximum_tokens=0,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong row-bound rejection")
  } noraise {
    _ => fail("empty row bound was admitted")
  }
}

///|
test "pre reduction lowers directional buffers with one BF16 output boundary" {
  let requirements = aot_test_requirements()
  let candidate = lower_hyper_connection_pre_reduce_cuda_aot_candidate(
    requirements,
    requirement_ordinal=7,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), PreReduceBf16V1)
  assert_eq(candidate.numerical_contract(), OrderedF32PreReductionBf16RoundV1)
  assert_eq(candidate.operation(), HyperConnectionPreReduceV1(2))
  assert_eq(candidate.capability(), HyperConnectionPreReduceV1Aot)
  assert_eq(candidate.stream_width(), 2)
  assert_eq(candidate.hidden_width(), 16)
  assert_eq(candidate.dimensions().grid_x(), 2)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[1].role(), HyperConnectionStateInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[2].role(), HyperConnectionPreInput)
  assert_eq(candidate.operands()[2].byte_count(), 16L)
  assert_eq(candidate.operands()[3].role(), HyperConnectionReducedOutput)
  assert_eq(candidate.operands()[3].byte_count(), 64L)
  assert_false(candidate.manifest_bindable())
  assert_true(
    @utf8.decode(candidate.recipe_bytes()).contains(
      "ordered-f32-pre-reduction-single-bf16-rne-v1",
    ),
  )
  let repeated = lower_hyper_connection_pre_reduce_cuda_aot_candidate(
    requirements,
    requirement_ordinal=7,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "pre reduction rejects block-control and post-combine ordinals" {
  let requirements = aot_test_requirements()
  for ordinal in [6, 8] {
    try
      lower_hyper_connection_pre_reduce_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=2,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong adjacent mHC phase rejection")
    } noraise {
      _ => fail("another mHC phase was lowered as pre-reduction")
    }
  }
  try
    lower_hyper_connection_pre_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
      maximum_tokens=2,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong pre-reduction target rejection")
  } noraise {
    _ => fail("pre-reduction admitted a non-BF16 target")
  }
  try
    lower_hyper_connection_pre_reduce_cuda_aot_candidate(
      requirements,
      requirement_ordinal=7,
      maximum_tokens=2,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong pre-reduction compiler rejection")
  } noraise {
    _ => fail("pre-reduction admitted reassociation")
  }
}

///|
test "post combination lowers directional controls and one BF16 output round" {
  let requirements = aot_test_requirements()
  let candidate = lower_hyper_connection_post_combine_cuda_aot_candidate(
    requirements,
    requirement_ordinal=8,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), PostCombineBf16V1)
  assert_eq(
    candidate.numerical_contract(),
    OrderedF32PostCombinationBf16RoundV1,
  )
  assert_eq(candidate.operation(), HyperConnectionPostCombineV1(2))
  assert_eq(candidate.capability(), HyperConnectionPostCombineV1Aot)
  assert_eq(candidate.stream_width(), 2)
  assert_eq(candidate.hidden_width(), 16)
  assert_eq(candidate.dimensions().grid_x(), 2)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[1].role(), HyperConnectionBranchInput)
  assert_eq(candidate.operands()[1].byte_count(), 64L)
  assert_eq(candidate.operands()[2].role(), HyperConnectionStateInput)
  assert_eq(candidate.operands()[2].byte_count(), 128L)
  assert_eq(candidate.operands()[3].role(), HyperConnectionPostInput)
  assert_eq(candidate.operands()[3].byte_count(), 16L)
  assert_eq(candidate.operands()[4].role(), HyperConnectionCombinationInput)
  assert_eq(candidate.operands()[4].byte_count(), 32L)
  assert_eq(candidate.operands()[5].role(), HyperConnectionCombinedOutput)
  assert_eq(candidate.operands()[5].byte_count(), 128L)
  assert_false(candidate.manifest_bindable())
  assert_true(
    @utf8.decode(candidate.recipe_bytes()).contains(
      "ordered-f32-post-combination-single-bf16-rne-v1",
    ),
  )
  let repeated = lower_hyper_connection_post_combine_cuda_aot_candidate(
    requirements,
    requirement_ordinal=8,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "post combination rejects pre-reduction head target and compiler aliases" {
  let requirements = aot_test_requirements()
  for ordinal in [7, 9] {
    try
      lower_hyper_connection_post_combine_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=2,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong adjacent mHC operation rejection")
    } noraise {
      _ => fail("another mHC phase was lowered as post-combination")
    }
  }
  try
    lower_hyper_connection_post_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=8,
      maximum_tokens=2,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong post-combination target rejection")
  } noraise {
    _ => fail("post-combination admitted a non-BF16 target")
  }
  try
    lower_hyper_connection_post_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=8,
      maximum_tokens=2,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong post-combination compiler rejection")
  } noraise {
    _ => fail("post-combination admitted reassociation")
  }
}
