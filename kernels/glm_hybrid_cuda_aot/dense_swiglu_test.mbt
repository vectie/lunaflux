///|
fn dense_swiglu_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DenseSwiGlu(_, _) {
      return ordinal
    }
  }
  abort("missing dense SwiGLU requirement")
}

///|
test "official Full and Flash dense SwiGLU bind separate weight operands" {
  for pooled in [false, true] {
    let requirements = glm_aot_requirements(
      pooled~,
      official_index=true,
      official_sparse=true,
    )
    let candidate = lower_dense_swiglu_cuda_aot_candidate(
      requirements,
      requirement_ordinal=dense_swiglu_ordinal(requirements),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
    let hidden = if pooled { 4096 } else { 6144 }
    let row_bytes = if pooled { 262144L } else { 393216L }
    let weight_bytes = if pooled { 100663296L } else { 150994944L }
    assert_eq(candidate.family(), SeparateGateUpDownWeightsBf16)
    assert_eq(candidate.operation(), DenseSwiGlu(hidden, 12288))
    assert_eq(candidate.hidden_size(), hidden)
    assert_eq(candidate.intermediate_size(), 12288)
    assert_eq(candidate.dimensions().grid_x(), if pooled { 512 } else { 768 })
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].role(), StepCountsInput)
    assert_eq(candidate.operands()[0].alignment(), 4L)
    assert_eq(candidate.operands()[1].role(), HiddenInput)
    assert_eq(candidate.operands()[1].byte_count(), row_bytes)
    assert_eq(candidate.operands()[2].role(), DenseGateWeights)
    assert_eq(candidate.operands()[3].role(), DenseUpWeights)
    assert_eq(candidate.operands()[4].role(), DenseDownWeights)
    assert_eq(candidate.operands()[2].byte_count(), weight_bytes)
    assert_eq(candidate.operands()[3].byte_count(), weight_bytes)
    assert_eq(candidate.operands()[4].byte_count(), weight_bytes)
    assert_eq(candidate.operands()[5].role(), HiddenOutput)
    assert_eq(candidate.operands()[5].byte_count(), row_bytes)
    assert_true(
      @utf8.decode(candidate.recipe_bytes()).contains(
        "weights=separate-row-major-gate-up-down",
      ),
    )
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "dense SwiGLU rejects shape ordinal target and compiler mismatches" {
  let ordinary = glm_aot_requirements()
  try
    lower_dense_swiglu_cuda_aot_candidate(
      ordinary,
      requirement_ordinal=dense_swiglu_ordinal(ordinary),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("neighboring dense SwiGLU shape admitted")
  }
  let exact = glm_aot_requirements(official_index=true, official_sparse=true)
  let ordinal = dense_swiglu_ordinal(exact)
  try
    lower_dense_swiglu_cuda_aot_candidate(
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
    lower_dense_swiglu_cuda_aot_candidate(
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
    lower_dense_swiglu_cuda_aot_candidate(
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
    _ => fail("reassociating compiler admitted")
  }
  try
    lower_dense_swiglu_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(fused_multiply_add=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong FMA compiler error")
  } noraise {
    _ => fail("FMA-contract compiler admitted")
  }
  try
    lower_dense_swiglu_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=4097,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong row-limit error")
  } noraise {
    _ => fail("out-of-bound maximum rows admitted")
  }
}
