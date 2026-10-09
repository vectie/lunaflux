///|
fn moe_feed_forward_ordinal(
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
  abort("missing MoE feed-forward stage")
}

///|
fn lower_moe_feed_forward_stage(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  stage : Int,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  let ordinal = moe_feed_forward_ordinal(requirements, stage)
  match stage {
    0 =>
      lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    1 =>
      lower_moe_shared_expert_feed_forward_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    2 =>
      lower_moe_routed_shared_combine_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=32,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    _ => abort("invalid stage")
  }
}

///|
test "MoE feed-forward candidates bind exact Full and Flash stages" {
  let full = glm_aot_requirements(
    official_sparse=true,
    pooled=false,
    routed_expert_count=256,
    experts_per_token=8,
    moe_intermediate_size=2048,
  )
  let flash = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    experts_per_token=8,
    moe_intermediate_size=2048,
    moe_activation=ClampedSwiGlu(10.0),
  )
  let full_routed = lower_moe_feed_forward_stage(full, 0)
  let flash_routed = lower_moe_feed_forward_stage(flash, 0)
  let full_shared = lower_moe_feed_forward_stage(full, 1)
  let flash_shared = lower_moe_feed_forward_stage(flash, 1)
  let full_add = lower_moe_feed_forward_stage(full, 2)
  let flash_add = lower_moe_feed_forward_stage(flash, 2)
  assert_eq(full_routed.family(), FullRoutedExperts256)
  assert_eq(flash_routed.family(), FlashRoutedExperts288)
  assert_eq(full_shared.family(), FullSharedExpert)
  assert_eq(flash_shared.family(), FlashSharedExpert)
  assert_eq(full_add.family(), FullRoutedSharedAdd)
  assert_eq(flash_add.family(), FlashRoutedSharedAdd)
  assert_eq(full_routed.operands().length(), 8)
  assert_eq(full_routed.operands()[2].role(), MoeSelectedIndicesOutput)
  assert_eq(full_routed.operands()[2].byte_count(), 1024L)
  assert_eq(full_routed.operands()[4].role(), MoeExpertGateWeights)
  assert_eq(full_routed.operands()[4].byte_count(), 6442450944L)
  assert_eq(flash_routed.operands()[4].byte_count(), 4831838208L)
  assert_eq(full_shared.operands()[2].byte_count(), 25165824L)
  assert_eq(flash_shared.operands()[2].byte_count(), 16777216L)
  assert_eq(full_add.operands()[1].role(), MoeRoutedOutput)
  assert_eq(full_add.operands()[2].role(), MoeSharedOutput)
  assert_eq(full_add.operands()[3].role(), MoeCombinedOutput)
  let routed_source = @utf8.decode(flash_routed.source_bytes())
  assert_true(routed_source.contains("for (int32_t expert = 0; expert < 288"))
  assert_true(
    routed_source.contains("fminf(__bfloat162float(gate_raw), 10.0f)"),
  )
  assert_true(routed_source.contains("input == output"))
  assert_false(routed_source.contains("shared_expert"))
  let shared_source = @utf8.decode(flash_shared.source_bytes())
  assert_true(shared_source.contains("const float gate_clamped = fminf"))
  assert_false(shared_source.contains("selected_indices"))
  let add_source = @utf8.decode(flash_add.source_bytes())
  assert_true(add_source.contains("output == left || output == right"))
  assert_false(add_source.contains("residual"))
  assert_false(full_routed.manifest_bindable())
  assert_false(flash_add.manifest_bindable())
}

///|
test "MoE feed-forward lowering rejects activation geometry ordinal and compiler mismatches" {
  let flash = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    experts_per_token=8,
    moe_intermediate_size=2048,
    moe_activation=ClampedSwiGlu(10.0),
  )
  try
    lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
      flash,
      requirement_ordinal=moe_feed_forward_ordinal(flash, 1),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("shared ordinal admitted as routed")
  }
  let wrong_activation = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    experts_per_token=8,
    moe_intermediate_size=2048,
  )
  try lower_moe_feed_forward_stage(wrong_activation, 0) catch {
    Invalid(Shape) => ()
    _ => fail("wrong activation error")
  } noraise {
    _ => fail("unclamped Flash experts admitted")
  }
  let wrong_k = glm_aot_requirements(
    official_sparse=true,
    pooled=false,
    routed_expert_count=256,
    experts_per_token=7,
    moe_intermediate_size=2048,
  )
  try lower_moe_feed_forward_stage(wrong_k, 0) catch {
    Invalid(Shape) => ()
    _ => fail("wrong top-k error")
  } noraise {
    _ => fail("wrong selected width admitted")
  }
  try
    lower_moe_routed_shared_combine_cuda_aot_candidate(
      flash,
      requirement_ordinal=moe_feed_forward_ordinal(flash, 2),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociation admitted")
  }
}
