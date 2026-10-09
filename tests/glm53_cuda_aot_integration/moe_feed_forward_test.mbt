///|
fn find_moe_feed_forward_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  stage : Int,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match (stage, requirement.operation()) {
      (0, RoutedExpertFeedForward(_, _, _)) => return ordinal
      (1, SharedExpertFeedForward(_, _, _)) => return ordinal
      (2, RoutedSharedExpertCombine(_, _, _)) => return ordinal
      _ => ()
    }
  }
  abort("missing official MoE feed-forward stage")
}

///|
fn official_moe_feed_forward_candidates(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Array[@glm_hybrid_cuda_aot.GlmHybridMoeFeedForwardCudaCandidate] {
  [
    try! @glm_hybrid_cuda_aot.lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
      requirements,
      requirement_ordinal=find_moe_feed_forward_ordinal(requirements, 0),
      maximum_tokens=32,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    ),
    try! @glm_hybrid_cuda_aot.lower_moe_shared_expert_feed_forward_cuda_aot_candidate(
      requirements,
      requirement_ordinal=find_moe_feed_forward_ordinal(requirements, 1),
      maximum_tokens=32,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    ),
    try! @glm_hybrid_cuda_aot.lower_moe_routed_shared_combine_cuda_aot_candidate(
      requirements,
      requirement_ordinal=find_moe_feed_forward_ordinal(requirements, 2),
      maximum_tokens=32,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    ),
  ]
}

///|
test "official Full and Flash MoE feed-forward stages join exact tensors and producers" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full = official_moe_feed_forward_candidates(full_requirements)
  let flash = official_moe_feed_forward_candidates(flash_requirements)
  assert_eq(full[0].operation(), RoutedExpertFeedForward(256, 2048, SwiGlu))
  assert_eq(full[1].operation(), SharedExpertFeedForward(1, 2048, SwiGlu))
  assert_eq(full[2].operation(), RoutedSharedExpertCombine(8, true, 2.5))
  assert_eq(
    flash[0].operation(),
    RoutedExpertFeedForward(288, 2048, ClampedSwiGlu(10.0)),
  )
  assert_eq(
    flash[1].operation(),
    SharedExpertFeedForward(1, 2048, ClampedSwiGlu(10.0)),
  )
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_gate = find_named_tensor(
    full_manifest, "model.layers.3.mlp.experts.0.gate_proj.weight",
  )
  let full_up = find_named_tensor(
    full_manifest, "model.layers.3.mlp.experts.0.up_proj.weight",
  )
  let full_down = find_named_tensor(
    full_manifest, "model.layers.3.mlp.experts.0.down_proj.weight",
  )
  assert_eq(full_gate.dtype(), BFloat16)
  assert_eq(full_gate.shape(), [2048L, 6144L])
  assert_eq(full_up.shape(), [2048L, 6144L])
  assert_eq(full_down.shape(), [6144L, 2048L])
  let flash_gate = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
  )
  let flash_down = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
  )
  assert_eq(flash_gate.dtype(), BFloat16)
  assert_eq(flash_gate.shape(), [2048L, 4096L])
  assert_eq(flash_down.shape(), [4096L, 2048L])
  let full_shared = find_named_tensor(
    full_manifest, "model.layers.3.mlp.shared_experts.gate_proj.weight",
  )
  let flash_shared = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
  )
  assert_eq(full_shared.shape(), [2048L, 6144L])
  assert_eq(flash_shared.shape(), [2048L, 4096L])
  let full_banks = @glm53_weights.build_moe_expert_bank_layout(
    full_manifest,
    layer_index=3,
  )
  let flash_banks = @glm53_weights.build_moe_expert_bank_layout(
    flash_manifest,
    layer_index=3,
  )
  assert_eq(full_banks.execution_identity(), full_manifest.execution_identity())
  assert_eq(full_banks.manifest_layout_digest(), full_manifest.layout_digest())
  assert_eq(full_banks.routed_gate().slices().length(), 256)
  assert_eq(flash_banks.routed_gate().slices().length(), 288)
  assert_eq(
    full_banks.routed_gate().slices()[0].source_name(),
    "model.layers.3.mlp.experts.0.gate_proj.weight",
  )
  assert_eq(full_banks.routed_gate().slices()[0].target_byte_offset(), 0L)
  assert_eq(
    full_banks.routed_gate().slices()[255].target_byte_offset(),
    6417285120L,
  )
  assert_eq(full_banks.routed_gate().byte_count(), 6442450944L)
  assert_eq(flash_banks.routed_gate().byte_count(), 4831838208L)
  assert_eq(full_banks.shared_gate().byte_count(), 25165824L)
  assert_eq(flash_banks.shared_gate().byte_count(), 16777216L)
  assert_false(full_banks.materialization_authority())
  assert_eq(full[0].operands()[4].byte_count(), 6442450944L)
  assert_eq(flash[0].operands()[4].byte_count(), 4831838208L)
  assert_eq(full[1].operands()[2].byte_count(), 25165824L)
  assert_eq(flash[1].operands()[2].byte_count(), 16777216L)
  assert_eq(full[0].operands()[7].role(), MoeRoutedOutput)
  assert_eq(full[1].operands()[5].role(), MoeSharedOutput)
  assert_eq(full[2].operands()[1].role(), MoeRoutedOutput)
  assert_eq(full[2].operands()[2].role(), MoeSharedOutput)
  assert_eq(full[2].operands()[3].role(), MoeCombinedOutput)
  assert_eq(
    full[0].operands()[7].byte_count(),
    full[2].operands()[1].byte_count(),
  )
  assert_eq(
    full[1].operands()[5].byte_count(),
    full[2].operands()[2].byte_count(),
  )
  let full_grouped = @glm_hybrid_cuda_aot.lower_moe_grouped_routing_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_moe_grouped_routing_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    full_grouped.operands()[3].byte_count(),
    full[0].operands()[2].byte_count(),
  )
  assert_eq(
    full_grouped.operands()[4].byte_count(),
    full[0].operands()[3].byte_count(),
  )
  let full_recipe = @utf8.decode(full[2].recipe_bytes())
  assert_true(
    full_recipe.contains(
      "excluded=routing-expert-ffn-shared-ffn-decoder-residual",
    ),
  )
  assert_false(full_recipe.contains("executor-authority"))
  let flash_source = @utf8.decode(flash[0].source_bytes())
  assert_true(flash_source.contains("up_clamped"))
  assert_false(@utf8.decode(full[0].source_bytes()).contains("up_clamped"))
  for candidate in full {
    ignore(try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate))
    assert_false(candidate.manifest_bindable())
  }
  for candidate in flash {
    ignore(try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate))
    assert_false(candidate.manifest_bindable())
  }
  inspect(
    full[0].source_digest().as_hex() +
    ":" +
    full[1].source_digest().as_hex() +
    ":" +
    full[2].source_digest().as_hex() +
    ":" +
    flash[0].source_digest().as_hex() +
    ":" +
    flash[1].source_digest().as_hex() +
    ":" +
    flash[2].source_digest().as_hex() +
    ":" +
    full[0].recipe_digest().as_hex() +
    ":" +
    full[1].recipe_digest().as_hex() +
    ":" +
    full[2].recipe_digest().as_hex() +
    ":" +
    flash[0].recipe_digest().as_hex() +
    ":" +
    flash[1].recipe_digest().as_hex() +
    ":" +
    flash[2].recipe_digest().as_hex(),
    content="eddf47bf92b9103c88c112ba7a37272a9be1273cb09b6c0fce87b3c917300ccd:d1730f3573fe0f10a5ea8cb7a60f3b3e2feff4ff7fd8929d25b1662cd4ee0eaf:ec7f8a6a568c529df65169688337b2ef5b3518ce5d4fb223cea33f508c8d114f:aa242d42a09adcc74dc577ccd49d89f15b1822413eb238b3121120da3a63baa0:34182d963d3f6afc633bbbc12f34475e6a3f77b5f4ca6050155193f9b2132ee8:663fffc9ae2d4f51e8a5d9c6155d0d79d370388eb465ab3c69b9bfa6c94383da:b3b0275be4b542104104621c611e6489899781c8a02076250e7ab64c2fc22a58:d6047426e310a30dd1ab26653c6743acca7b04ed6b308fc5feb8b80b384c91c1:3cac20ab676226d6c600f2f147908f3079a23b0e75c801a0a33af1388b978f68:94f332a9f45b0f6bb2c1b6ffab8025b727a3c0e6ce616d99641a0a80592887c3:3040c792901123e09d564b410cf04a317d7884937e3c7d4987a09cf5bf72f3a4:04cf73bafd1f8b5b2c41f1e2b2daffc4d8e48740826f1f5978309cfe59df1468",
  )
}

///|
test "official numeric plans expose routed shared and composition boundaries per MoE layer" {
  let plans = [
    @glm53_numeric_plan.build_full_numeric_plan(
      full_metadata(),
      BFloat16WithFloat32Routing,
    ),
    @glm53_numeric_plan.build_flash_numeric_plan(
      flash_metadata(),
      BFloat16WithFloat32Routing,
    ),
  ]
  for profile_index, plan in plans {
    let mut routed = 0
    let mut shared = 0
    let mut combined = 0
    for ordinal in 0..<plan.operation_count() {
      guard plan.operation(ordinal) is Some(operation) else { continue }
      match operation.kind() {
        RoutedExpertSwiGlu => routed += 1
        SharedExpertSwiGlu => shared += 1
        RoutedSharedExpertBf16Combine => combined += 1
        _ => continue
      }
      assert_eq(
        operation.execution().activation_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        operation.execution().accumulator(),
        @numeric_contract.AccumulatorDType::f32(),
      )
      assert_eq(
        operation.execution().output(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
    }
    let expected = if profile_index == 0 { 76 } else { 43 }
    assert_eq(routed, expected)
    assert_eq(shared, expected)
    assert_eq(combined, expected)
  }
}
