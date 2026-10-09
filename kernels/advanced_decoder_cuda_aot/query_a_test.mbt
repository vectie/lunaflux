///|
test "Query-A lowers exact replicated block-FP8 ABI" {
  let requirements = aot_test_requirements(hidden_size=128, query_low_rank=128)
  let candidate = lower_query_a_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=1,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), ReplicatedBlockFp8E4M3Ue8M0ProjectionV1)
  assert_eq(candidate.input_width(), 128)
  assert_eq(candidate.output_width(), 128)
  assert_eq(candidate.block_width(), 128)
  assert_true(candidate.tensor_parallel_replicated())
  assert_eq(candidate.operands().map(op => op.role()), [
    StepCountsInput,
    HiddenInput,
    QueryAWeightCodesInput,
    QueryAWeightScalesInput,
    QueryLowRankAOutput,
  ])
  assert_eq(candidate.operands().map(op => op.byte_count()), [
    20L, 512L, 16384L, 1L, 512L,
  ])
  assert_false(candidate.manifest_bindable())
}

///|
test "Query-A rejects Query-B ordinal unaligned geometry compiler and target" {
  let exact = aot_test_requirements(hidden_size=128, query_low_rank=128)
  try
    lower_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=4,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(4) => ()
    _ => fail("wrong ordinal failure")
  } noraise {
    _ => fail("Query-B ordinal admitted")
  }
  let unaligned = aot_test_requirements()
  try
    lower_query_a_projection_cuda_aot_candidate(
      unaligned,
      requirement_ordinal=1,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape failure")
  } noraise {
    _ => fail("unaligned geometry admitted")
  }
  try
    lower_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=1,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target failure")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=1,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(fused_multiply_add=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler failure")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
}
