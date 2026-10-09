///|
fn sparse_attention_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaSparseAttention(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing sparse attention")
}

///|
test "Full and Flash sparse attention bind exact selected widths and ABI" {
  for pooled in [false, true] {
    let requirements = glm_aot_requirements(
      pooled~,
      official_index=true,
      official_sparse=true,
    )
    let ordinal = sparse_attention_ordinal(requirements)
    let candidate = lower_dsa_sparse_attention_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    assert_eq(candidate.operation(), DsaSparseAttention(64, 2048, 256, 256))
    assert_eq(candidate.key_value_head_count(), 64)
    assert_eq(candidate.selected_width(), if pooled { 2051 } else { 2048 })
    assert_eq(candidate.operands().length(), 9)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[1].byte_count(), 128L)
    assert_eq(candidate.operands()[2].role(), DsaKeyCountInput)
    assert_eq(candidate.operands()[3].byte_count(), 128L)
    assert_eq(candidate.operands()[4].byte_count(), 1048576L)
    assert_eq(candidate.operands()[5].byte_count(), 1048576L)
    assert_eq(candidate.operands()[6].byte_count(), 1048576L)
    assert_eq(
      candidate.operands()[7].byte_count(),
      if pooled {
        262528L
      } else {
        262144L
      },
    )
    assert_eq(candidate.operands()[8].role(), DsaAttentionOutput)
    assert_eq(candidate.operands()[8].byte_count(), 1048576L)
    let source = @utf8.decode(candidate.source_bytes())
    assert_true(source.contains("key == -1"))
    assert_true(source.contains("row_valid = 0"))
    assert_true(source.contains("key > position"))
    let recipe = @utf8.decode(candidate.recipe_bytes())
    assert_true(recipe.contains("serial-correctness-only"))
    assert_true(recipe.contains("selected-width-distinct"))
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "sparse attention rejects nonofficial shape ordinal and compiler" {
  let ordinary = glm_aot_requirements()
  let ordinal = sparse_attention_ordinal(ordinary)
  try
    lower_dsa_sparse_attention_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected shape error")
  } noraise {
    _ => fail("nonofficial sparse attention admitted")
  }
  let exact = glm_aot_requirements(official_index=true, official_sparse=true)
  let exact_ordinal = sparse_attention_ordinal(exact)
  try
    lower_dsa_sparse_attention_cuda_aot_candidate(
      exact,
      requirement_ordinal=exact_ordinal + 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected ordinal error")
  } noraise {
    _ => fail("neighbor ordinal admitted")
  }
  try
    lower_dsa_sparse_attention_cuda_aot_candidate(
      exact,
      requirement_ordinal=exact_ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(fused_multiply_add=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler error")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
}
