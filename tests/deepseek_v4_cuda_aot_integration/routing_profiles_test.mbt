///|
fn official_top_k_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is NoAuxiliaryLossTopK(_, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost no-auxiliary-loss top-k")
}

///|
fn official_selected_weight_finalize_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is SelectedExpertWeightFinalize(_, _, _, _) {
      return ordinal
    }
  }
  abort("official DeepSeek requirements lost selected-weight finalization")
}

///|
test "all official DeepSeek profiles bind split selection and weight finalization" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '0', 22, 23, 256, 1.5),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '1', 22, 23, 256, 1.5),
    ("deepseek-ai/DeepSeek-V4-Pro", '2', 21, 22, 384, 2.5),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '3', 21, 22, 384, 2.5),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '4', 22, 23, 256, 1.5),
  ]
  let top_k_recipe_digests : Array[String] = []
  let top_k_source_digests : Array[String] = []
  let finalize_recipe_digests : Array[String] = []
  let finalize_source_digests : Array[String] = []
  for entry in cases {
    let (
      model_id,
      fill,
      expected_top_k_ordinal,
      expected_finalize_ordinal,
      routed_expert_count,
      scaling,
    ) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    let top_k_ordinal = official_top_k_ordinal(requirements)
    let finalize_ordinal = official_selected_weight_finalize_ordinal(
      requirements,
    )
    assert_eq(top_k_ordinal, expected_top_k_ordinal)
    assert_eq(finalize_ordinal, expected_finalize_ordinal)
    let top_k = @advanced_decoder_cuda_aot.lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
      requirements,
      requirement_ordinal=top_k_ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(top_k.operation(), NoAuxiliaryLossTopK(routed_expert_count, 6))
    assert_eq(top_k.capability(), StableTopKRoutingAot)
    assert_eq(top_k.dimensions().grid_x(), 2)
    assert_eq(top_k.dimensions().block_x(), 1)
    assert_eq(top_k.operands().length(), 4)
    assert_eq(top_k.operands()[0].role(), StepCountsInput)
    assert_eq(top_k.operands()[0].byte_count(), 20L)
    assert_eq(top_k.operands()[1].role(), RoutingScoresInput)
    assert_eq(
      top_k.operands()[1].byte_count(),
      routed_expert_count.to_int64() * 8L,
    )
    assert_eq(top_k.operands()[2].role(), RoutingSelectionBiasInput)
    assert_eq(
      top_k.operands()[2].byte_count(),
      routed_expert_count.to_int64() * 4L,
    )
    assert_eq(top_k.operands()[3].role(), RoutingIndicesOutput)
    assert_eq(top_k.operands()[3].byte_count(), 48L)
    assert_false(top_k.manifest_bindable())
    assert_false(top_k_recipe_digests.contains(top_k.recipe_digest().as_hex()))
    top_k_recipe_digests.push(top_k.recipe_digest().as_hex())
    if !top_k_source_digests.contains(top_k.source_digest().as_hex()) {
      top_k_source_digests.push(top_k.source_digest().as_hex())
    }
    let finalize = @advanced_decoder_cuda_aot.lower_selected_expert_weight_finalize_cuda_aot_candidate(
      requirements,
      requirement_ordinal=finalize_ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(
      finalize.operation(),
      SelectedExpertWeightFinalize(routed_expert_count, 6, true, scaling),
    )
    assert_eq(finalize.capability(), SelectedExpertWeightFinalizeAot)
    assert_eq(finalize.dimensions().grid_x(), 2)
    assert_eq(finalize.dimensions().block_x(), 1)
    assert_eq(finalize.operands().length(), 4)
    assert_eq(finalize.operands()[0].role(), StepCountsInput)
    assert_eq(finalize.operands()[0].byte_count(), 20L)
    assert_eq(finalize.operands()[1].role(), RoutingScoresInput)
    assert_eq(
      finalize.operands()[1].byte_count(),
      routed_expert_count.to_int64() * 8L,
    )
    assert_eq(finalize.operands()[2].role(), RoutingIndicesInput)
    assert_eq(finalize.operands()[2].byte_count(), 48L)
    assert_eq(finalize.operands()[3].role(), RoutingWeightsOutput)
    assert_eq(finalize.operands()[3].byte_count(), 48L)
    assert_false(finalize.manifest_bindable())
    assert_false(
      finalize_recipe_digests.contains(finalize.recipe_digest().as_hex()),
    )
    finalize_recipe_digests.push(finalize.recipe_digest().as_hex())
    if !finalize_source_digests.contains(finalize.source_digest().as_hex()) {
      finalize_source_digests.push(finalize.source_digest().as_hex())
    }
  }
  assert_eq(top_k_recipe_digests.length(), 5)
  assert_eq(top_k_source_digests.length(), 2)
  assert_eq(finalize_recipe_digests.length(), 5)
  assert_eq(finalize_source_digests.length(), 2)
}

///|
test "official DeepSeek split routing lowerers reject each other's ordinal" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Pro", '5',
  )
  let top_k_ordinal = official_top_k_ordinal(requirements)
  let finalize_ordinal = official_selected_weight_finalize_ordinal(requirements)
  try
    @advanced_decoder_cuda_aot.lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
      requirements,
      requirement_ordinal=finalize_ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, finalize_ordinal)
    _ => fail("unexpected official top-k ordinal rejection")
  } noraise {
    _ => fail("weight finalization was lowered as official top-k")
  }
  try
    @advanced_decoder_cuda_aot.lower_selected_expert_weight_finalize_cuda_aot_candidate(
      requirements,
      requirement_ordinal=top_k_ordinal,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, top_k_ordinal)
    _ => fail("unexpected official finalizer ordinal rejection")
  } noraise {
    _ => fail("top-k was lowered as official weight finalization")
  }
}
