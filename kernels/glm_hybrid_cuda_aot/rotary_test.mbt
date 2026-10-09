///|
fn glm_rotary_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is GlmRotaryPosition(_, _) {
      return ordinal
    }
  }
  abort("missing GLM rotary requirement")
}

///|
test "official Full interleaved RoPE binds query and shared-key dataflow" {
  let requirements = glm_aot_requirements(
    official_index=true,
    official_sparse=true,
    rope_theta=8000000.0,
  )
  let candidate = lower_dsa_interleaved_rotary_cuda_aot_candidate(
    requirements,
    requirement_ordinal=glm_rotary_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), DsaInterleavedQuerySharedKeyRopeBf16)
  assert_eq(candidate.operation(), GlmRotaryPosition(Some(8000000.0), true))
  assert_eq(candidate.query_head_count(), 64)
  assert_eq(candidate.query_head_dimension(), 256)
  assert_eq(candidate.rope_dimension(), 64)
  assert_eq(candidate.maximum_positions(), 4096)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().grid_y(), 65)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), QueryPositionsInput)
  assert_eq(candidate.operands()[1].byte_count(), 128L)
  assert_eq(candidate.operands()[2].role(), DsaQueryHeadsOutput)
  assert_eq(candidate.operands()[2].byte_count(), 1048576L)
  assert_eq(candidate.operands()[3].role(), DsaKeyRopeSuffixOutput)
  assert_eq(candidate.operands()[3].byte_count(), 4096L)
  assert_eq(candidate.operands()[4].role(), DsaAttentionQueriesInput)
  assert_eq(candidate.operands()[4].byte_count(), 1048576L)
  assert_eq(candidate.operands()[5].role(), DsaAttentionKeyRopeInput)
  assert_eq(candidate.operands()[5].byte_count(), 4096L)
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("amplitude=1"))
  assert_true(recipe.contains("awaits-kv-b-key-nope-join"))
  assert_false(candidate.manifest_bindable())
}

///|
test "Full rotary rejects Flash no-op shape theta ordinal target and compiler" {
  let flash = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
  )
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      flash,
      requirement_ordinal=glm_rotary_ordinal(flash),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong Flash error")
  } noraise {
    _ => fail("Flash zero-work rotary admitted")
  }
  let wrong_shape = glm_aot_requirements(rope_theta=8000000.0)
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      wrong_shape,
      requirement_ordinal=glm_rotary_ordinal(wrong_shape),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("neighboring rotary geometry admitted")
  }
  let wrong_theta = glm_aot_requirements(
    official_index=true,
    official_sparse=true,
    rope_theta=10000.0,
  )
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      wrong_theta,
      requirement_ordinal=glm_rotary_ordinal(wrong_theta),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong theta error")
  } noraise {
    _ => fail("wrong theta admitted")
  }
  let exact = glm_aot_requirements(
    official_index=true,
    official_sparse=true,
    rope_theta=8000000.0,
  )
  let ordinal = glm_rotary_ordinal(exact)
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("neighbor ordinal admitted")
  }
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_dsa_interleaved_rotary_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(fused_multiply_add=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
}
