///|
fn moe_grouped_routing_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is MoeGroupedNoAuxiliaryLossTopK(_, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing MoE grouped routing")
}

///|
test "grouped routing candidate binds official Full and Flash geometry" {
  let full_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=false,
    routed_expert_count=256,
    experts_per_token=8,
  )
  let flash_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    experts_per_token=8,
  )
  let full = lower_moe_grouped_routing_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=moe_grouped_routing_ordinal(full_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let flash = lower_moe_grouped_routing_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=moe_grouped_routing_ordinal(flash_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(full.family(), FullGroupedRouting256)
  assert_eq(flash.family(), FlashGroupedRouting288)
  assert_eq(
    full.operation(),
    MoeGroupedNoAuxiliaryLossTopK(256, 1, 2, 1, 8, true, 2.5),
  )
  assert_eq(full.capability(), MoeGroupedRoutingAot)
  assert_eq(full.operands().length(), 5)
  assert_eq(full.operands()[1].role(), MoeUncorrectedScoresOutput)
  assert_eq(full.operands()[2].role(), MoeChoiceScoresOutput)
  assert_eq(full.operands()[3].role(), MoeSelectedIndicesOutput)
  assert_eq(full.operands()[4].role(), MoeSelectedWeightsOutput)
  assert_eq(full.operands()[1].byte_count(), 32768L)
  assert_eq(full.operands()[3].byte_count(), 1024L)
  assert_eq(flash.operands()[1].byte_count(), 36864L)
  assert_eq(flash.operands()[3].byte_count(), 1024L)
  assert_eq(full.dimensions().grid_x(), 32)
  assert_eq(full.dimensions().block_x(), 1)
  let source = @utf8.decode(flash.source_bytes())
  assert_true(source.contains("float group_scores[1]"))
  assert_true(source.contains("int32_t group_top[2]"))
  assert_true(source.contains("int32_t selected_groups[1]"))
  assert_false(source.contains("expert_output"))
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}

///|
test "grouped routing rejects ordinal expert compiler and limit mismatches" {
  let requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    experts_per_token=8,
  )
  let ordinal = moe_grouped_routing_ordinal(requirements)
  try
    lower_moe_grouped_routing_cuda_aot_candidate(
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
  let hostile = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=287,
    experts_per_token=8,
  )
  try
    lower_moe_grouped_routing_cuda_aot_candidate(
      hostile,
      requirement_ordinal=moe_grouped_routing_ordinal(hostile),
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
  try
    lower_moe_grouped_routing_cuda_aot_candidate(
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
    lower_moe_grouped_routing_cuda_aot_candidate(
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
