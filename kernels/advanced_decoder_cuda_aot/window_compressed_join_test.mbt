///|
fn window_join_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DeterministicWindowCompressedIndexJoinV1(_, _) {
      return ordinal
    }
  }
  abort("window-compressed join requirement missing")
}

///|
test "window-compressed join lowers exact eight-operand I32 ABI" {
  let requirements = aot_test_requirements(
    maximum_position_count=1024,
    deterministic_compression_ratio=128,
  )
  let ordinal = window_join_test_ordinal(requirements)
  let candidate = lower_deterministic_window_compressed_join_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), DeterministicWindowCompressedJoinI32V1)
  assert_eq(
    candidate.numerical_contract(),
    BitExactOfficialWindowThenCompressedI32V1,
  )
  assert_eq(
    candidate.operation(),
    DeterministicWindowCompressedIndexJoinV1(64, 128),
  )
  assert_eq(candidate.capability(), DeterministicWindowCompressedIndexJoinV1Aot)
  assert_eq(candidate.maximum_compressed_width(), 8)
  assert_eq(candidate.maximum_joined_width(), 72)
  assert_eq(candidate.dimensions().block_x(), 1)
  let operands = candidate.operands()
  assert_eq(operands.length(), 8)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[1].role(), WindowQueryPositionsInput)
  assert_eq(operands[2].role(), WindowSequenceLengthsInput)
  assert_eq(operands[3].role(), WindowQueryModesInput)
  assert_eq(operands[4].role(), CompressedIndexCountsInput)
  assert_eq(operands[5].role(), CompressedIndicesInput)
  assert_eq(operands[5].byte_count(), 64L)
  assert_eq(operands[6].role(), JoinedIndexCountsOutput)
  assert_eq(operands[7].role(), JoinedIndicesOutput)
  assert_eq(operands[7].byte_count(), 576L)
  for operand in operands {
    assert_eq(operand.alignment(), 4L)
  }
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("compressed_base_value + slot"))
  let replay = lower_deterministic_window_compressed_join_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), replay.source_digest())
  assert_eq(candidate.recipe_digest(), replay.recipe_digest())
}

///|
test "window-compressed join rejects adjacent ordinal compiler and overflow" {
  let requirements = aot_test_requirements(deterministic_compression_ratio=128)
  let ordinal = window_join_test_ordinal(requirements)
  for hostile in [ordinal - 1, ordinal + 1] {
    try
      lower_deterministic_window_compressed_join_cuda_aot_candidate(
        requirements,
        requirement_ordinal=hostile,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, hostile)
      _ => fail("unexpected ordinal rejection")
    } noraise {
      _ => fail("adjacent operation lowered as index join")
    }
  }
  try
    lower_deterministic_window_compressed_join_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
}
