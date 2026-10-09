///|
fn output_a_test_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputAProjectionV1(_, _, _, _) {
      return ordinal
    }
  }
  abort("output-A requirement missing")
}

///|
test "grouped output-A lowers exact directional BF16 ABI" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
  )
  let ordinal = output_a_test_ordinal(requirements)
  let candidate = lower_attention_output_a_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=2,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), GroupedOutputAProjectionBf16V1)
  assert_eq(candidate.group_count(), 1)
  assert_eq(candidate.input_width_per_group(), 128)
  assert_eq(candidate.output_rank(), 4)
  assert_eq(candidate.dimensions().grid_x(), 2)
  assert_eq(candidate.dimensions().grid_y(), 1)
  let operands = candidate.operands()
  assert_eq(operands.map(op => op.role()), [
    StepCountsInput,
    AttentionOutputProjectionInput,
    AttentionOutputAWeightsInput,
    AttentionOutputRankOutput,
  ])
  assert_eq(operands.map(op => op.byte_count()), [20L, 512L, 1024L, 16L])
  assert_false(candidate.manifest_bindable())
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("blockIdx.y"))
  assert_true(source.contains("__float2bfloat16_rn"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("|launch=2:1:1:256:1:1:0"))
  assert_true(recipe.contains("converted-grouped-weight@1024/16"))
}

///|
test "output-A rejects wrong ordinal target compiler and row limit" {
  let requirements = aot_test_requirements(
    attention_head_count=2,
    attention_head_dimension=64,
  )
  let ordinal = output_a_test_ordinal(requirements)
  try
    lower_attention_output_a_projection_cuda_aot_candidate(
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
    lower_attention_output_a_projection_cuda_aot_candidate(
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
  try
    lower_attention_output_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
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
    _ => fail("wrong row-limit rejection")
  } noraise {
    _ => fail("output-A exceeded its row limit")
  }
}
