///|
fn output_b_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputBProjectionV1(_, _, _) {
      return ordinal
    }
  }
  abort("output-B requirement missing")
}

///|
test "Output-B lowers exact raw E4M3 UE8M0 directional ABI" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
    hidden_size=128,
    output_low_rank=128,
  )
  let ordinal = output_b_test_ordinal(requirements)
  let candidate = lower_attention_output_b_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), BlockFp8E4M3Ue8M0ProjectionV1)
  assert_eq(
    candidate.numerical_contract(),
    PerRowBlock128Ue8M0OrderedF32Bf16RoundV1,
  )
  assert_eq(candidate.input_width(), 128)
  assert_eq(candidate.output_width(), 128)
  assert_eq(candidate.block_width(), 128)
  assert_eq(candidate.tensor_parallel_world_size(), 1)
  assert_false(candidate.owns_collective())
  assert_false(candidate.manifest_bindable())
}

///|
test "Output-B rejects unauthenticated small geometry and compiler semantics" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
  )
  let ordinal = output_b_test_ordinal(requirements)
  try
    lower_attention_output_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong block geometry failure")
  } noraise {
    _ => fail("non-128 Output-B geometry admitted")
  }
  let official_like = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
    hidden_size=128,
    output_low_rank=128,
  )
  let official_ordinal = output_b_test_ordinal(official_like)
  try
    lower_attention_output_b_projection_cuda_aot_candidate(
      official_like,
      requirement_ordinal=official_ordinal,
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
