///|
fn kv_split_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is QueryKeyRmsNorm(_) {
      return ordinal
    }
  }
  abort("missing KV split norm")
}

///|
test "Full and Flash KV split norm bind optional suffix ABIs" {
  for pooled in [false, true] {
    let requirements = glm_aot_requirements(
      pooled~,
      official_index=true,
      official_sparse=true,
    )
    let candidate = lower_dsa_key_value_split_norm_cuda_aot_candidate(
      requirements,
      requirement_ordinal=kv_split_norm_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    assert_eq(candidate.family(), DsaKeyValuePrefixRmsNormBf16)
    assert_eq(candidate.prefix_width(), 512)
    assert_eq(candidate.suffix_width(), if pooled { 0 } else { 64 })
    assert_eq(candidate.operands().length(), if pooled { 4 } else { 5 })
    assert_eq(candidate.operands()[1].role(), DsaKeyValueCompressedRopeInput)
    assert_eq(
      candidate.operands()[1].byte_count(),
      if pooled {
        32768L
      } else {
        36864L
      },
    )
    assert_eq(candidate.operands()[2].byte_count(), 1024L)
    assert_eq(candidate.operands()[3].role(), DsaKeyValueNormalizedOutput)
    assert_eq(candidate.operands()[3].byte_count(), 32768L)
    if !pooled {
      assert_eq(candidate.operands()[4].role(), DsaKeyRopeSuffixOutput)
      assert_eq(candidate.operands()[4].byte_count(), 4096L)
    }
    let source = @utf8.decode(candidate.source_bytes())
    assert_eq(source.contains("suffix_output"), !pooled)
    assert_true(
      @utf8.decode(candidate.recipe_bytes()).contains(
        "kv-b-normalized-input-compatible",
      ),
    )
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "KV split norm rejects shape epsilon target compiler and ordinal mismatches" {
  let ordinary = glm_aot_requirements()
  try
    lower_dsa_key_value_split_norm_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=kv_split_norm_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("nonofficial KV split admitted")
  }
  let exact = glm_aot_requirements(official_index=true, official_sparse=true)
  let ordinal = kv_split_norm_ordinal(exact)
  try
    lower_dsa_key_value_split_norm_cuda_aot_candidate(
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
  let wrong_epsilon = glm_aot_requirements(
    official_index=true,
    official_sparse=true,
    rms_epsilon=0.000001,
  )
  try
    lower_dsa_key_value_split_norm_cuda_aot_candidate(
      wrong_epsilon,
      requirement_ordinal=kv_split_norm_ordinal(wrong_epsilon),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong epsilon error")
  } noraise {
    _ => fail("wrong epsilon admitted")
  }
  try
    lower_dsa_key_value_split_norm_cuda_aot_candidate(
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
    lower_dsa_key_value_split_norm_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociation admitted")
  }
}
