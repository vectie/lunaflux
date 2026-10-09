///|
fn kda_recurrent_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaRecurrentUpdate(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA recurrent update")
}

///|
test "KDA recurrent lowerer binds serial exact state transition" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_recurrent_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_recurrent_ordinal(requirements),
    maximum_tokens=32,
    maximum_sequences=8,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.family(), SerialNormalizedRecurrentDeltaBf16F32)
  assert_eq(candidate.operation(), KdaRecurrentUpdate(64, 128, -5.0))
  assert_eq(candidate.operands()[5].role(), KdaLogDecayInput)
  assert_eq(candidate.operands()[5].byte_count(), 1048576L)
  assert_eq(candidate.operands()[7].role(), KdaRecurrentInitialStateInput)
  assert_eq(candidate.operands()[7].byte_count(), 33554432L)
  assert_eq(candidate.operands()[9].role(), KdaRecurrentFinalStateOutput)
  assert_eq(candidate.dimensions().grid_x(), 1)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA recurrent lowerer rejects wrong shape limits ordinal target compiler" {
  let wrong = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=63,
  )
  try
    lower_kda_recurrent_cuda_aot_candidate(
      wrong,
      requirement_ordinal=kda_recurrent_ordinal(wrong),
      maximum_tokens=32,
      maximum_sequences=8,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("wrong KDA geometry admitted")
  }
  let exact = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_recurrent_ordinal(exact)
  for sequences in [0, 33] {
    try
      lower_kda_recurrent_cuda_aot_candidate(
        exact,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        maximum_sequences=sequences,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    catch {
      Invalid(Limit) => ()
      _ => fail("wrong limit error")
    } noraise {
      _ => fail("hostile sequence limit admitted")
    }
  }
  try
    lower_kda_recurrent_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      maximum_sequences=8,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("neighbor operation admitted")
  }
}
