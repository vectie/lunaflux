///|
fn kda_output_gate_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaOutputGateProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA output gate")
}

///|
test "KDA output gate binds explicit BF16 intermediate" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_output_gate_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_output_gate_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.operation(), KdaOutputGateProjection(4096, 64, 128))
  assert_eq(candidate.family(), SerialBiasFreeTwoStageBf16)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[4].role(), KdaOutputGateLowRankOutput)
  assert_eq(candidate.operands()[4].byte_count(), 8192L)
  assert_eq(candidate.operands()[5].role(), KdaOutputGateOutput)
  assert_eq(candidate.operands()[5].byte_count(), 524288L)
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA output gate rejects neighboring norm and hostile compiler" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_output_gate_ordinal(requirements)
  try
    lower_kda_output_gate_cuda_aot_candidate(
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
    _ => fail("gated norm admitted as gate projection")
  }
  try
    lower_kda_output_gate_cuda_aot_candidate(
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
}
