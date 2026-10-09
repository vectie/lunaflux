///|
fn moe_score_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeSigmoidAndCorrectionBias(_) {
      return ordinal
    }
  }
  abort("missing MoE sigmoid correction")
}

///|
test "MoE score candidates bind official Full and Flash five-operand ABI" {
  let full_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=false,
    routed_expert_count=256,
  )
  let flash_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
  )
  let full = lower_moe_sigmoid_correction_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=moe_score_ordinal(full_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let flash = lower_moe_sigmoid_correction_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=moe_score_ordinal(flash_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(full.family(), FullSigmoidCorrection256)
  assert_eq(flash.family(), FlashSigmoidCorrection288)
  assert_eq(full.operation(), MoeSigmoidAndCorrectionBias(256))
  assert_eq(flash.operation(), MoeSigmoidAndCorrectionBias(288))
  assert_eq(full.capability(), MoeSigmoidAndCorrectionBiasAot)
  assert_eq(full.operands().length(), 5)
  assert_eq(full.operands()[0].role(), StepCountsInput)
  assert_eq(full.operands()[1].role(), MoeRouterFloat32LogitsOutput)
  assert_eq(full.operands()[1].byte_count(), 32768L)
  assert_eq(full.operands()[2].role(), MoeCorrectionBiasInput)
  assert_eq(full.operands()[2].byte_count(), 1024L)
  assert_eq(full.operands()[3].role(), MoeUncorrectedScoresOutput)
  assert_eq(full.operands()[4].role(), MoeChoiceScoresOutput)
  assert_eq(flash.operands()[1].byte_count(), 36864L)
  assert_eq(flash.operands()[2].byte_count(), 1152L)
  assert_eq(flash.operands()[3].byte_count(), 36864L)
  assert_eq(flash.operands()[4].byte_count(), 36864L)
  assert_eq(full.dimensions().grid_x(), 32)
  assert_eq(flash.dimensions().grid_x(), 36)
  let source = @utf8.decode(flash.source_bytes())
  assert_true(source.contains("scores[index] = sigmoid_score"))
  assert_true(source.contains("scores_for_choice[index] = __fadd_rn"))
  assert_true(source.contains("if (scores == scores_for_choice) return"))
  assert_false(source.contains("top_k"))
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}

///|
test "MoE score candidate rejects ordinal shape compiler and limits" {
  let requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
  )
  let ordinal = moe_score_ordinal(requirements)
  try
    lower_moe_sigmoid_correction_cuda_aot_candidate(
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
    _ => fail("neighboring requirement admitted")
  }
  for hostile_experts in [287, 289] {
    let hostile = glm_aot_requirements(
      official_sparse=true,
      pooled=true,
      routed_expert_count=hostile_experts,
    )
    try
      lower_moe_sigmoid_correction_cuda_aot_candidate(
        hostile,
        requirement_ordinal=moe_score_ordinal(hostile),
        maximum_tokens=32,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    catch {
      Invalid(Shape) => ()
      _ => fail("wrong shape error")
    } noraise {
      _ => fail("neighboring expert count admitted")
    }
  }
  try
    lower_moe_sigmoid_correction_cuda_aot_candidate(
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
    lower_moe_sigmoid_correction_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=0,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong limit error")
  } noraise {
    _ => fail("zero rows admitted")
  }
}
