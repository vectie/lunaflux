///|
fn kda_decay_control_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaDecayControl(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA decay control")
}

///|
test "KDA decay control binds raw BF16 F32 parameters and F32 output" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_decay_control_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_decay_control_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.operation(), KdaDecayControl(64, 128, -5.0))
  assert_eq(candidate.family(), PointwiseSafeLowerBoundSigmoidF32)
  assert_eq(candidate.operands()[1].role(), KdaDecayRawForgetInput)
  assert_eq(candidate.operands()[1].byte_count(), 524288L)
  assert_eq(candidate.operands()[2].role(), KdaDecayBiasInput)
  assert_eq(candidate.operands()[2].byte_count(), 32768L)
  assert_eq(candidate.operands()[3].role(), KdaDecayLogRateInput)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  assert_eq(candidate.operands()[4].role(), KdaDecayOutput)
  assert_eq(candidate.operands()[4].byte_count(), 1048576L)
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA decay control rejects neighboring ordinal compiler target and shape" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_decay_control_ordinal(requirements)
  try
    lower_kda_decay_control_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("convolution admitted as decay control")
  }
  try
    lower_kda_decay_control_cuda_aot_candidate(
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
    lower_kda_decay_control_cuda_aot_candidate(
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
  let hostile = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=8,
  )
  try
    lower_kda_decay_control_cuda_aot_candidate(
      hostile,
      requirement_ordinal=kda_decay_control_ordinal(hostile),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("non-official heads admitted")
  }
}
