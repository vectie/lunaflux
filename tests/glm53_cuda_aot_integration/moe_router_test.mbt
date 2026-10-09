///|
test "official Full and Flash MoE router projection joins weights and numeric plan" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full = @glm_hybrid_cuda_aot.lower_moe_router_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_moe_router_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_moe_router_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_moe_router_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), MoeRouterProjection(6144, 256, true))
  assert_eq(flash.operation(), MoeRouterProjection(4096, 288, true))
  assert_eq(full.family(), FullFloat32Router256)
  assert_eq(flash.family(), FlashFloat32Router288)
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_weight = find_named_tensor(
    full_manifest, "model.layers.3.mlp.gate.weight",
  )
  let flash_weight = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.mlp.gate.weight",
  )
  assert_eq(full_weight.dtype(), BFloat16)
  assert_eq(full_weight.shape(), [256L, 6144L])
  assert_eq(flash_weight.dtype(), BFloat16)
  assert_eq(flash_weight.shape(), [288L, 4096L])
  assert_eq(full.operands()[1].role(), MoeRouterFloat32Input)
  assert_eq(full.operands()[2].role(), MoeRouterBf16Weights)
  assert_eq(full.operands()[3].role(), MoeRouterFloat32LogitsOutput)
  assert_eq(full.operands()[2].byte_count(), 3145728L)
  assert_eq(flash.operands()[2].byte_count(), 2359296L)
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
      if operation.kind() == MoeRouterProjection {
        count += 1
        assert_eq(
          operation.execution().activation_input(),
          @numeric_contract.ComputeDType::f32(),
        )
        assert_eq(
          operation.execution().activation_compute(),
          @numeric_contract.ComputeDType::f32(),
        )
        assert_eq(
          operation.execution().tensor_input(),
          @numeric_contract.ComputeDType::bfloat16(),
        )
        assert_eq(
          operation.execution().accumulator(),
          @numeric_contract.AccumulatorDType::f32(),
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
  assert_true(source.contains("weight_base = column * 4096ULL"))
  assert_true(source.contains("__bfloat162float"))
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_false(source.contains("sigmoid"))
  assert_false(source.contains("correction"))
  assert_false(source.contains("topk"))
  let recipe = @utf8.decode(flash.recipe_bytes())
  assert_true(recipe.contains("weights=row-major-bf16-expert-hidden-no-bias"))
  assert_true(recipe.contains("input=f32-activation-view"))
  assert_true(
    recipe.contains(
      "excluded=sigmoid-correction-bias-group-topk-expert-ffn-residual",
    ),
  )
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_router(full))
  ignore(try! @glm_hybrid_candidate_evidence.from_moe_router(flash))
  inspect(
    full.source_digest().as_hex(),
    content="85374a1554cc39e7330bea4f428a5e374b27f8d4f1b65b3a836de4ffbc4f5ffb",
  )
  inspect(
    flash.source_digest().as_hex(),
    content="0c0aa3bf6ab51d968fa29e37477addde621ecc1534b26d77390f947a54751f1a",
  )
  inspect(
    full.recipe_digest().as_hex(),
    content="e5c15f1b9c767e1346f8bacf456caedfbbeb6b1171576a6b6345488b6a84cc4c",
  )
  inspect(
    flash.recipe_digest().as_hex(),
    content="f5082b5b8b422f987e3b884513baef185fc64e5f63545b1bd6d2a96f8826686d",
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
