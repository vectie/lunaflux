///|
test "key/value projection lowers exact replicated block-FP8 ABI" {
  let requirements = aot_test_requirements(
    hidden_size=128,
    query_low_rank=128,
    attention_head_dimension=512,
    rope_head_dimension=64,
  )
  let candidate = lower_key_value_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=2,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), ReplicatedKeyValueBlockFp8E4M3Ue8M0ProjectionV1)
  assert_eq(candidate.input_width(), 128)
  assert_eq(candidate.output_width(), 512)
  assert_eq(candidate.rope_width(), 64)
  assert_eq(candidate.block_width(), 128)
  assert_true(candidate.tensor_parallel_replicated())
  assert_eq(candidate.operands().map(operand => operand.role()), [
    StepCountsInput,
    HiddenInput,
    KeyValueWeightCodesInput,
    KeyValueWeightScalesInput,
    KeyValueProjectionOutput,
  ])
  assert_eq(candidate.operands().map(operand => operand.byte_count()), [
    20L, 512L, 65536L, 4L, 2048L,
  ])
  assert_false(candidate.manifest_bindable())
}

///|
test "key/value projection rejects wrong ordinal shape compiler and target" {
  let exact = aot_test_requirements(
    hidden_size=128,
    query_low_rank=128,
    attention_head_dimension=512,
    rope_head_dimension=64,
  )
  try
    lower_key_value_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=1,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(1) => ()
    _ => fail("wrong ordinal failure")
  } noraise {
    _ => fail("Query-A ordinal admitted")
  }
  try
    lower_key_value_projection_cuda_aot_candidate(
      aot_test_requirements(),
      requirement_ordinal=2,
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
    lower_key_value_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=2,
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
    lower_key_value_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=2,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler failure")
  } noraise {
    _ => fail("reassociation compiler admitted")
  }
}
