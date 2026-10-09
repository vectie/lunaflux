///|
test "official Full and Flash grouped routing joins score producers and numeric plan" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full = @glm_hybrid_cuda_aot.lower_moe_grouped_routing_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_moe_grouped_routing_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_moe_grouped_routing_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_moe_grouped_routing_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    full.operation(),
    MoeGroupedNoAuxiliaryLossTopK(256, 1, 2, 1, 8, true, 2.5),
  )
  assert_eq(
    flash.operation(),
    MoeGroupedNoAuxiliaryLossTopK(288, 1, 2, 1, 8, true, 2.5),
  )
  let full_spec = @glm53_spec.Glm53FullSpec::official()
  let flash_spec = @glm53_spec.Glm53FlashSpec::official()
  assert_eq(full_spec.expert_group_count(), 1)
  assert_eq(full_spec.expert_group_score_topk(), 2)
  assert_eq(full_spec.selected_expert_group_count(), 1)
  assert_eq(flash_spec.expert_group_count(), 1)
  assert_eq(flash_spec.expert_group_score_topk(), 2)
  assert_eq(flash_spec.selected_expert_group_count(), 1)
  let score_stage = @glm_hybrid_cuda_aot.lower_moe_sigmoid_correction_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_moe_score_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(score_stage.operands()[3].role(), flash.operands()[1].role())
  assert_eq(score_stage.operands()[4].role(), flash.operands()[2].role())
  assert_eq(
    score_stage.operands()[3].byte_count(),
    flash.operands()[1].byte_count(),
  )
  assert_eq(
    score_stage.operands()[4].byte_count(),
    flash.operands()[2].byte_count(),
  )
  let plans = [
    @glm53_numeric_plan.build_full_numeric_plan(
      full_metadata,
      BFloat16WithFloat32Routing,
    ),
    @glm53_numeric_plan.build_flash_numeric_plan(
      flash_metadata,
      BFloat16WithFloat32Routing,
    ),
  ]
  for profile_index, plan in plans {
    let mut count = 0
    for ordinal in 0..<plan.operation_count() {
      guard plan.operation(ordinal) is Some(operation) else { continue }
      if operation.kind() == MoeF32GroupedRouting {
        count += 1
        assert_eq(
          operation.execution().activation_input(),
          @numeric_contract.ComputeDType::f32(),
        )
        assert_eq(
          operation.execution().output(),
          @numeric_contract.ComputeDType::f32(),
        )
      }
    }
    assert_eq(count, if profile_index == 0 { 76 } else { 43 })
  }
  let source = @utf8.decode(flash.source_bytes())
  assert_true(source.contains("choice_scores[score_base + expert]"))
  assert_true(
    source.contains("uncorrected_scores[score_base + selected_indices"),
  )
  assert_true(source.contains("__fadd_rn(denominator, 1.0e-20f)"))
  let recipe = @utf8.decode(flash.recipe_bytes())
  assert_true(recipe.contains("pytorch-unsorted-topk-parity-unqualified"))
  assert_true(
    recipe.contains("excluded=expert-ffn-shared-expert-combine-residual"),
  )
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_grouped_routing(full))
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_grouped_routing(flash))
  inspect(
    full.source_digest().as_hex(),
    content="da6cb856cc21d2164cd1a9ceece3f88e27001e92b0a1377e9c13c2d28fd7c331",
  )
  inspect(
    flash.source_digest().as_hex(),
    content="1a2789a3914f661ee3be70dab36801e96ee26d3be6f424207632bb7899545765",
  )
  inspect(
    full.recipe_digest().as_hex(),
    content="62e80841c91ad614c04146ef7817e71230068f712de8598a8875e367457d535c",
  )
  inspect(
    flash.recipe_digest().as_hex(),
    content="21feb6bc0a3cf9800a904a1a7dcc1d86cfedc5a967252f38a7ca6199cc719c83",
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
