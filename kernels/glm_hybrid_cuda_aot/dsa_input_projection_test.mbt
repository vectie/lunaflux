///|
fn query_a_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaQueryAProjection(_, _) {
      return ordinal
    }
  }
  abort("missing q-a projection")
}

///|
fn kv_a_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaKeyValueAProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing kv-a projection")
}

///|
fn query_a_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaQueryARmsNorm(_, _) {
      return ordinal
    }
  }
  abort("missing q-a RMSNorm")
}

///|
fn query_b_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaQueryBProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing q-b projection")
}

///|
fn key_value_b_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is KeyValueLowRankProjection(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing kv-b projection")
}

///|
test "official Full and Flash A projections bind exact dense shapes" {
  for pooled in [false, true] {
    let requirements = glm_aot_requirements(
      pooled~,
      official_index=true,
      official_sparse=true,
    )
    assert_eq(
      requirements.requirements()[kv_a_ordinal(requirements)].operation(),
      DsaKeyValueAProjection(
        if pooled {
          4096
        } else {
          6144
        },
        512,
        if pooled {
          0
        } else {
          64
        },
      ),
    )
    let query = lower_dsa_query_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=query_a_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let kv = lower_dsa_key_value_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=kv_a_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let query_norm = lower_foundation_cuda_aot_candidate(
      requirements,
      requirement_ordinal=query_a_norm_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let query_b = lower_dsa_query_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=query_b_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let kv_b = lower_dsa_key_value_b_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=key_value_b_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let hidden = if pooled { 4096 } else { 6144 }
    let query_rank = if pooled { 1536 } else { 2048 }
    let kv_output = if pooled { 512 } else { 576 }
    assert_eq(query.operation(), DsaQueryAProjection(hidden, query_rank))
    assert_eq(query.input_width(), hidden)
    assert_eq(query.output_width(), query_rank)
    assert_eq(query.operands()[3].role(), DsaQueryCompressedOutput)
    assert_eq(
      kv.operation(),
      DsaKeyValueAProjection(hidden, 512, if pooled { 0 } else { 64 }),
    )
    assert_eq(kv.output_width(), kv_output)
    assert_eq(kv.operands()[3].role(), DsaKeyValueCompressedRopeOutput)
    assert_eq(query_norm.family(), DsaQueryARmsNormBf16)
    assert_eq(query_norm.operation(), DsaQueryARmsNorm(query_rank, 0.00001))
    assert_eq(query_norm.operands()[1].role(), DsaQueryCompressedInput)
    assert_eq(
      query_norm.operands()[1].byte_count(),
      Int64::from_int(32 * query_rank * 2),
    )
    assert_eq(
      query_norm.operands()[2].byte_count(),
      Int64::from_int(query_rank * 2),
    )
    assert_eq(query_norm.operands()[3].role(), DsaQueryNormalizedOutput)
    assert_true(
      @utf8.decode(query_norm.recipe_bytes()).contains("stage=q-a-rmsnorm-only"),
    )
    assert_eq(query_b.operation(), DsaQueryBProjection(query_rank, 64, 256))
    assert_eq(query_b.output_width(), 16384)
    assert_eq(query_b.operands()[1].role(), DsaQueryNormalizedInput)
    assert_eq(query_b.operands()[3].role(), DsaQueryHeadsOutput)
    assert_true(
      @utf8.decode(query_b.recipe_bytes()).contains(
        "component-order=nope-then-rope",
      ),
    )
    assert_eq(
      kv_b.operation(),
      KeyValueLowRankProjection(512, 64, if pooled { 256 } else { 192 }, 256),
    )
    assert_eq(kv_b.output_width(), if pooled { 32768 } else { 28672 })
    assert_eq(kv_b.operands()[1].role(), DsaKeyValueNormalizedInput)
    assert_eq(kv_b.operands()[3].role(), DsaKeyValueHeadsOutput)
    assert_true(
      @utf8.decode(kv_b.recipe_bytes()).contains(
        "output=token-kv-head-key-nope-then-value",
      ),
    )
    assert_true(
      @utf8.decode(kv.recipe_bytes()).contains(
        "contiguous-compressed-kv-then-rope-key",
      ),
    )
    assert_false(query.manifest_bindable())
    assert_false(kv.manifest_bindable())
    assert_false(query_norm.manifest_bindable())
    assert_false(query_b.manifest_bindable())
    assert_false(kv_b.manifest_bindable())
  }
}

///|
test "A projections reject shape ordinal target compiler and overflow mismatches" {
  let ordinary = glm_aot_requirements()
  try
    lower_dsa_query_a_projection_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=query_a_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected shape rejection")
  } noraise {
    _ => fail("nonofficial q-a shape admitted")
  }
  try
    lower_foundation_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=query_a_norm_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected q-a norm shape rejection")
  } noraise {
    _ => fail("nonofficial q-a RMSNorm admitted")
  }
  try
    lower_dsa_query_b_projection_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=query_b_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected q-b shape rejection")
  } noraise {
    _ => fail("nonofficial q-b projection admitted")
  }
  try
    lower_dsa_key_value_b_projection_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=key_value_b_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("unexpected kv-b shape rejection")
  } noraise {
    _ => fail("nonofficial kv-b projection admitted")
  }
  let exact = glm_aot_requirements(official_index=true, official_sparse=true)
  let ordinal = query_a_ordinal(exact)
  try
    lower_dsa_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal + 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected ordinal rejection")
  } noraise {
    _ => fail("neighbor ordinal admitted")
  }
  try
    lower_dsa_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_dsa_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
  try
    lower_foundation_cuda_aot_candidate(
      exact,
      requirement_ordinal=query_a_norm_ordinal(exact),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected q-a norm compiler rejection")
  } noraise {
    _ => fail("reassociating q-a RMSNorm admitted")
  }
  try
    lower_dsa_query_a_projection_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=4097,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected limit rejection")
  } noraise {
    _ => fail("oversized projection admitted")
  }
}
