///|
fn rotary_test_requirements(
  content_fill? : Char = 'a',
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements {
  aot_test_requirements(
    content_fill~,
    attention_head_count=64,
    attention_head_dimension=128,
    rope_head_dimension=64,
    maximum_position_count=1048576,
  )
}

///|
test "scaled rotary lowering binds exact RoPE-only ABI" {
  let requirements = rotary_test_requirements()
  let candidate = lower_scaled_rotary_position_cuda_aot_candidate(
    requirements,
    requirement_ordinal=5,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), CompressedScaledYarnRopeBf16)
  assert_eq(candidate.numerical_contract(), AdjacentPairF32YarnBf16RoundV1)
  assert_eq(candidate.capability(), ScaledRotaryPositionAot)
  assert_eq(candidate.query_head_count(), 64)
  assert_eq(candidate.key_value_head_count(), 1)
  assert_eq(candidate.rope_dimension(), 64)
  assert_eq(candidate.maximum_positions(), 1048576)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().grid_y(), 65)
  assert_eq(candidate.dimensions().block_x(), 32)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[1].role(), PositionInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[2].role(), QueryInput)
  assert_eq(candidate.operands()[2].byte_count(), 262144L)
  assert_eq(candidate.operands()[3].role(), QueryOutput)
  assert_eq(candidate.operands()[4].role(), KeyValueInput)
  assert_eq(candidate.operands()[4].byte_count(), 4096L)
  assert_eq(candidate.operands()[5].role(), KeyValueOutput)
  for item in candidate.operands() {
    assert_true(
      match item.role() {
        HiddenInput | HiddenOutput | ModelWeights => false
        _ => true
      },
    )
  }
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("position < 0"))
  assert_true(source.contains("pair * 2"))
  assert_true(source.contains("cosf(angle)"))
  assert_true(source.contains("sinf(angle)"))
  assert_true(source.contains("__float2bfloat16_rn"))
  let foreign = lower_scaled_rotary_position_cuda_aot_candidate(
    rotary_test_requirements(content_fill='d'),
    requirement_ordinal=5,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), foreign.source_digest())
  assert_true(candidate.recipe_digest() != foreign.recipe_digest())
}

///|
test "scaled rotary lowering rejects hostile ordinal target compiler and width" {
  let requirements = rotary_test_requirements()
  for ordinal in [4, 6] {
    try
      lower_scaled_rotary_position_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler=aot_test_compiler(),
        limits=aot_test_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong ordinal error")
    } noraise {
      _ => fail("adjacent operation was admitted")
    }
  }
  try
    lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=5,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("non-BF16 target was admitted")
  }
  try
    lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=5,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(reassociate=true),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociation was admitted")
  }
  try
    lower_scaled_rotary_position_cuda_aot_candidate(
      aot_test_requirements(),
      requirement_ordinal=5,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("non-64 RoPE width was admitted")
  }
  try
    lower_scaled_rotary_position_cuda_aot_candidate(
      requirements,
      requirement_ordinal=5,
      maximum_tokens=4097,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("token capacity overflow was admitted")
  }
}
