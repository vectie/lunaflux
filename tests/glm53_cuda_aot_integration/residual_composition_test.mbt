///|
fn find_residual_composition_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeDecoderResidualAdd(_) ||
      requirement.operation() is HyperConnectionMix(_) {
      return ordinal
    }
  }
  abort("missing residual composition")
}

///|
test "official Full and Flash residual composition preserve distinct ordering" {
  let full = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full_candidate = try! @glm_hybrid_cuda_aot.lower_moe_decoder_residual_add_cuda_aot_candidate(
    full,
    requirement_ordinal=find_residual_composition_ordinal(full),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_candidate = try! @glm_hybrid_cuda_aot.lower_hyper_connection_mix_cuda_aot_candidate(
    flash,
    requirement_ordinal=find_residual_composition_ordinal(flash),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let mut full_combine_ordinal = -1
  let mut flash_combine_ordinal = -1
  for ordinal, requirement in full.requirements() {
    if requirement.operation() is RoutedSharedExpertCombine(_, _, _) {
      full_combine_ordinal = ordinal
    }
  }
  for ordinal, requirement in flash.requirements() {
    if requirement.operation() is RoutedSharedExpertCombine(_, _, _) {
      flash_combine_ordinal = ordinal
    }
  }
  let full_combine = try! @glm_hybrid_cuda_aot.lower_moe_routed_shared_combine_cuda_aot_candidate(
    full,
    requirement_ordinal=full_combine_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_combine = try! @glm_hybrid_cuda_aot.lower_moe_routed_shared_combine_cuda_aot_candidate(
    flash,
    requirement_ordinal=flash_combine_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full_candidate.operation(), MoeDecoderResidualAdd(6144))
  assert_eq(full_candidate.capability(), MoeDecoderResidualAddAot)
  assert_eq(flash_candidate.operation(), HyperConnectionMix(4))
  assert_eq(flash_candidate.capability(), Advanced(HyperConnectionMixAot))
  assert_eq(full_candidate.operands()[2].role(), MoeCombinedOutput)
  assert_eq(flash_candidate.operands()[4].role(), HyperSublayerOutputInput)
  assert_eq(
    full_combine.operands()[3].byte_count(),
    full_candidate.operands()[2].byte_count(),
  )
  assert_eq(
    flash_combine.operands()[3].byte_count(),
    flash_candidate.operands()[4].byte_count(),
  )
  assert_eq(full_candidate.requirements_digest(), full.digest())
  assert_eq(flash_candidate.requirements_digest(), flash.digest())
  let full_recipe = @utf8.decode(full_candidate.recipe_bytes())
  let flash_recipe = @utf8.decode(flash_candidate.recipe_bytes())
  assert_true(full_recipe.contains("stage=moe-decoder-outer-residual-add"))
  assert_true(
    flash_recipe.contains("stage=hyper-connection-outer-residual-mix"),
  )
  assert_true(flash_recipe.contains("next-layer-collapse-in-place-aliasing"))
  assert_true(full_combine_ordinal >= 0)
  assert_true(flash_combine_ordinal >= 0)
  assert_true(full_combine_ordinal < full_candidate.requirement_ordinal())
  ignore(
    try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(full_candidate),
  )
  ignore(
    try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(flash_candidate),
  )
  assert_true(
    @utf8.decode(flash_candidate.source_bytes()).contains(
      "__bfloat162float(__float2bfloat16_rn(residual_sum))",
    ),
  )
}
