///|
fn kda_gated_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaGatedRmsNorm(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA gated RMSNorm")
}

///|
test "KDA gated RMSNorm binds exact per-head gate weight and output ABI" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_gated_rms_norm_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_gated_norm_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.operation(), KdaGatedRmsNorm(64, 128, 0.00001))
  assert_eq(candidate.family(), SerialPerHeadSigmoidGatedRmsNormBf16)
  assert_eq(candidate.operands().length(), 5)
  assert_eq(candidate.operands()[1].role(), KdaGatedNormRecurrentInput)
  assert_eq(candidate.operands()[1].byte_count(), 524288L)
  assert_eq(candidate.operands()[2].role(), KdaGatedNormGateInput)
  assert_eq(candidate.operands()[3].role(), KdaGatedNormWeights)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  assert_eq(candidate.operands()[4].role(), KdaGatedNormOutput)
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA gated RMSNorm rejects adjacent ordinal and hostile compiler" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_gated_norm_ordinal(requirements)
  try
    lower_kda_gated_rms_norm_cuda_aot_candidate(
      requirements,
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
    _ => fail("output projection admitted as gated norm")
  }
  try
    lower_kda_gated_rms_norm_cuda_aot_candidate(
      requirements,
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
    lower_kda_gated_rms_norm_cuda_aot_candidate(
      requirements,
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
}
