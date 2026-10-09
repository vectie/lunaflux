///|
fn glm_aot_attention_output_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputProjection(_, _) {
      return ordinal
    }
  }
  abort("missing attention-output projection requirement")
}

///|
test "GLM DSA attention output binds exact counts-driven projection ABI" {
  let requirements = glm_aot_requirements()
  let ordinal = glm_aot_attention_output_ordinal(requirements)
  let candidate = lower_attention_output_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), AttentionOutputProjectionBf16)
  assert_eq(candidate.operation(), AttentionOutputProjection(64, 192))
  assert_eq(candidate.capability(), AttentionOutputProjectionAot)
  assert_eq(candidate.input_width(), 192)
  assert_eq(candidate.output_width(), 64)
  assert_eq(candidate.dimensions().grid_x(), 8)
  assert_eq(candidate.dimensions().block_x(), 256)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), HiddenInput)
  assert_eq(candidate.operands()[1].byte_count(), 12288L)
  assert_eq(candidate.operands()[2].role(), ModelWeights)
  assert_eq(candidate.operands()[2].byte_count(), 24576L)
  assert_eq(candidate.operands()[3].role(), HiddenOutput)
  assert_eq(candidate.operands()[3].byte_count(), 4096L)
  assert_eq(
    candidate.function_symbol(),
    "lunaflux_glm_hybrid_attention_output_projection_bf16_v1",
  )
  assert_eq(
    candidate.numerical_contract(),
    OrderedFloat32AccumulationBf16RoundV1,
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const int row_count = counts[3];"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("bias"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("family=attention-output-projection-bf16"))
  assert_true(recipe.contains("hidden-output@4096/16"))
  assert_false(candidate.manifest_bindable())
  let repeated = lower_attention_output_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
}

///|
test "GLM attention output rejects a neighboring dense operation" {
  let requirements = glm_aot_requirements()
  let head_ordinal = requirements.requirements().length() - 1
  try
    lower_attention_output_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=head_ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(ordinal) => assert_eq(ordinal, head_ordinal)
    _ => fail("unexpected neighboring-operation rejection")
  } noraise {
    _ => fail("language-model head admitted as attention output")
  }
}
