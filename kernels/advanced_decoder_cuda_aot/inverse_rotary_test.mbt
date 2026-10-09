///|
fn inverse_rotary_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputInverseRotaryV1(_, _, _, _, _) {
      return ordinal
    }
  }
  abort("inverse rotary requirement missing")
}

///|
test "attention output inverse RoPE lowers exact full-head ABI" {
  let requirements = aot_test_requirements(
    maximum_position_count=1024,
    attention_head_count=2,
    attention_head_dimension=64,
    rope_head_dimension=64,
    deterministic_compression_ratio=128,
  )
  let ordinal = inverse_rotary_test_ordinal(requirements)
  let candidate = lower_attention_output_inverse_rotary_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), AttentionOutputInverseScaledYarnBf16V1)
  assert_eq(
    candidate.numerical_contract(),
    PrefixBitExactAdjacentPairInverseF32YarnBf16RoundV1,
  )
  assert_eq(candidate.head_count(), 2)
  assert_eq(candidate.head_dimension(), 64)
  assert_eq(candidate.rope_dimension(), 64)
  assert_eq(candidate.dimensions().block_x(), 32)
  let operands = candidate.operands()
  assert_eq(operands.length(), 3)
  assert_eq(operands[0].role(), StepCountsInput)
  assert_eq(operands[1].role(), PositionInput)
  assert_eq(operands[2].role(), AttentionOutputRotaryInOut)
  assert_eq(operands[2].byte_count(), 512L)
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("if (pair < 0) { return; }"))
  assert_true(source.contains("-__fmul_rn(left, sine)"))
  assert_false(source.contains("__restrict__"))
}

///|
test "inverse RoPE rejects adjacent ordinal target compiler and overflow" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
    rope_head_dimension=64,
    deterministic_compression_ratio=128,
  )
  let ordinal = inverse_rotary_test_ordinal(requirements)
  for hostile in [ordinal - 1, ordinal + 1] {
    try
      lower_attention_output_inverse_rotary_cuda_aot_candidate(
        requirements,
        requirement_ordinal=hostile,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, hostile)
      _ => fail("wrong ordinal rejection")
    } noraise {
      _ => fail("adjacent operation lowered")
    }
  }
  try
    lower_attention_output_inverse_rotary_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target rejection")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_attention_output_inverse_rotary_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
}
