///|
fn glm_aot_dsa_index_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing DSA index-compute requirement")
}

///|
fn glm_aot_dsa_reuse_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexReuse {
      return ordinal
    }
  }
  abort("missing DSA index-reuse requirement")
}

///|
test "Full projected DSA selection and reuse bind exact separate ABIs" {
  let requirements = glm_aot_requirements(official_index=true)
  let selection = lower_dsa_index_selection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=glm_aot_dsa_index_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(selection.family(), RawTokenScoreStableTopKBf16)
  assert_eq(selection.operation(), DsaIndexCompute(32, 128, 2048, true))
  assert_eq(selection.pool_size(), 1)
  assert_eq(selection.candidate_capacity(), 32)
  assert_eq(selection.output_width(), 2048)
  assert_eq(selection.dimensions().block_x(), 1)
  assert_eq(selection.operands().length(), 10)
  assert_eq(selection.operands()[0].role(), StepCountsInput)
  assert_eq(selection.operands()[0].alignment(), 4L)
  assert_eq(selection.operands()[1].role(), QueryPositionsInput)
  assert_eq(selection.operands()[1].byte_count(), 128L)
  assert_eq(selection.operands()[2].byte_count(), 262144L)
  assert_eq(selection.operands()[3].byte_count(), 2048L)
  assert_eq(selection.operands()[4].byte_count(), 8192L)
  assert_eq(selection.operands()[5].byte_count(), 128L)
  assert_eq(selection.operands()[6].byte_count(), 128L)
  assert_eq(selection.operands()[7].byte_count(), 4L)
  assert_eq(selection.operands()[8].byte_count(), 128L)
  assert_eq(selection.operands()[9].byte_count(), 262144L)
  let source = @utf8.decode(selection.source_bytes())
  assert_true(source.contains("final_raw > query_position"))
  assert_true(source.contains("candidate < best"))
  assert_false(source.contains("tail_count"))
  let recipe = @utf8.decode(selection.recipe_bytes())
  assert_true(recipe.contains("qualification=serial-correctness-only"))
  assert_true(recipe.contains("key-validity=single-contiguous-span"))
  assert_true(
    recipe.contains("batch-contract=single-sequence-counts-2-equals-1"),
  )
  assert_false(selection.manifest_bindable())
  let reuse = lower_dsa_index_reuse_cuda_aot_candidate(
    requirements,
    requirement_ordinal=glm_aot_dsa_reuse_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(reuse.family(), SelectedIndexReuseI32)
  assert_eq(reuse.operation(), DsaIndexReuse)
  assert_eq(reuse.output_width(), 2048)
  assert_eq(reuse.dimensions().grid_x(), 256)
  assert_eq(reuse.dimensions().block_x(), 256)
  assert_eq(reuse.operands().length(), 3)
  assert_eq(reuse.operands()[1].role(), DsaSelectedIndicesInput)
  assert_eq(reuse.operands()[1].byte_count(), 262144L)
  assert_eq(reuse.operands()[2].role(), DsaSelectedIndicesOutput)
  assert_eq(reuse.numerical_contract(), BitExactI32Copy)
  assert_false(reuse.manifest_bindable())
}

///|
test "Flash projected DSA selection expands pools and appends bounded tail" {
  let requirements = glm_aot_requirements(pooled=true, official_index=true)
  let selection = lower_dsa_index_selection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=glm_aot_dsa_index_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(selection.family(), PooledScoreStableTopKWithTailBf16)
  assert_eq(selection.pool_size(), 4)
  assert_eq(selection.candidate_capacity(), 8)
  assert_eq(selection.output_width(), 2051)
  assert_eq(selection.operands()[4].byte_count(), 2048L)
  assert_eq(selection.operands()[5].byte_count(), 128L)
  assert_eq(selection.operands()[6].byte_count(), 32L)
  assert_eq(selection.operands()[9].byte_count(), 262528L)
  let source = @utf8.decode(selection.source_bytes())
  assert_true(source.contains("const int tail_count = visible_count % 4;"))
  assert_true(source.contains("tail_layout_valid"))
  assert_true(source.contains("rank * 4 + offset"))
  assert_false(source.contains("pool_ape"))
  assert_false(selection.manifest_bindable())
}

///|
test "DSA index selection rejects shape policy ordinal and compiler mismatches" {
  let ordinary = glm_aot_requirements()
  try
    lower_dsa_index_selection_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=glm_aot_dsa_index_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected nonofficial-shape rejection")
  } noraise {
    _ => fail("nonofficial DSA index geometry admitted")
  }
  let bad_policy = glm_aot_requirements(
    pooled=true,
    official_index=true,
    pool_tail=false,
  )
  try
    lower_dsa_index_selection_cuda_aot_candidate(
      bad_policy,
      requirement_ordinal=glm_aot_dsa_index_ordinal(bad_policy),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected pooled-policy rejection")
  } noraise {
    _ => fail("Flash pooling without required tail admitted")
  }
  let exact = glm_aot_requirements(official_index=true)
  let ordinal = glm_aot_dsa_index_ordinal(exact)
  try
    lower_dsa_index_selection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(rejected) => assert_eq(rejected, ordinal + 1)
    _ => fail("unexpected ordinal rejection")
  } noraise {
    _ => fail("neighboring DSA operation admitted as selection")
  }
  try
    lower_dsa_index_selection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(fused_multiply_add=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("FMA-enabled DSA scoring admitted")
  }
}
