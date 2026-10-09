///|
test "official Full and Flash MoE sigmoid correction joins manifest and numeric plan" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full = @glm_hybrid_cuda_aot.lower_moe_sigmoid_correction_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_moe_score_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_moe_sigmoid_correction_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_moe_score_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), MoeSigmoidAndCorrectionBias(256))
  assert_eq(flash.operation(), MoeSigmoidAndCorrectionBias(288))
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_bias = find_named_tensor(
    full_manifest, "model.layers.3.mlp.gate.e_score_correction_bias",
  )
  let flash_bias = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
  )
  assert_eq(full_bias.dtype(), Float32)
  assert_eq(full_bias.shape(), [256L])
  assert_eq(flash_bias.dtype(), Float32)
  assert_eq(flash_bias.shape(), [288L])
  assert_eq(full.operands()[1].role(), MoeRouterFloat32LogitsOutput)
  assert_eq(full.operands()[2].role(), MoeCorrectionBiasInput)
  assert_eq(full.operands()[3].role(), MoeUncorrectedScoresOutput)
  assert_eq(full.operands()[4].role(), MoeChoiceScoresOutput)
  let router = @glm_hybrid_cuda_aot.lower_moe_router_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_moe_router_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(router.operands()[3].role(), flash.operands()[1].role())
  assert_eq(router.operands()[3].byte_count(), flash.operands()[1].byte_count())
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
      if operation.kind() == MoeF32SigmoidAndCorrection {
        count += 1
        assert_eq(
          operation.execution().activation_input(),
          @numeric_contract.ComputeDType::f32(),
        )
        assert_eq(
          operation.execution().tensor_input(),
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
  assert_true(source.contains("expf(-raw_logits[index])"))
  assert_true(source.contains("scores[index] = sigmoid_score"))
  assert_true(source.contains("if (scores == scores_for_choice) return"))
  assert_false(source.contains("top_k"))
  let recipe = @utf8.decode(flash.recipe_bytes())
  assert_true(
    recipe.contains("separate-row-major-f32-uncorrected-and-choice-scores"),
  )
  assert_true(
    recipe.contains(
      "excluded=group-scoring-topk-masking-normalization-scaling-expert-ffn",
    ),
  )
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_sigmoid_correction(full))
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_sigmoid_correction(flash))
  inspect(
    full.source_digest().as_hex(),
    content="02d3fed7e3a23b3e67afbed88ce27b392314f678154e6ece1a20267cf19f2d01",
  )
  inspect(
    flash.source_digest().as_hex(),
    content="ac3106eec5c1559d332c26543b922a7570f31f8dd30a284cc8305cd4675f9327",
  )
  inspect(
    full.recipe_digest().as_hex(),
    content="9c882a1931140474e2236fd8eee2a57456ea9eca24fdb59e2cba39aeda790008",
  )
  inspect(
    flash.recipe_digest().as_hex(),
    content="391db382be2ee144e1fb59c85224ceaf57e809eca88194d74dacc0cab00db5f4",
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
