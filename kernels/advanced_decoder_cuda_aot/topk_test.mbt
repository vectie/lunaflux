///|
fn aot_test_top_k_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is NoAuxiliaryLossTopK(_, _) {
      return ordinal
    }
  }
  abort("top-k test requirement missing")
}

///|
fn aot_test_finalize_ordinal(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is SelectedExpertWeightFinalize(_, _, _, _) {
      return ordinal
    }
  }
  abort("selected-weight finalizer test requirement missing")
}

///|
test "stable biased top-k lowers with exact deterministic index ABI" {
  let requirements = aot_test_requirements()
  let ordinal = aot_test_top_k_ordinal(requirements)
  let candidate = lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), NoAuxiliaryLossTopKI32)
  assert_eq(candidate.numerical_contract(), StableLowerExpertIdTopKV1)
  assert_eq(candidate.operation(), NoAuxiliaryLossTopK(8, 2))
  assert_eq(candidate.capability(), StableTopKRoutingAot)
  assert_eq(candidate.routed_expert_count(), 8)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.dimensions().shared_memory_bytes(), 0)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[0].alignment(), 4L)
  assert_eq(candidate.operands()[1].role(), RoutingScoresInput)
  assert_eq(candidate.operands()[1].byte_count(), 1024L)
  assert_eq(candidate.operands()[2].role(), RoutingSelectionBiasInput)
  assert_eq(candidate.operands()[2].byte_count(), 32L)
  assert_eq(candidate.operands()[3].role(), RoutingIndicesOutput)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("choice == best_value && expert < best"))
  assert_true(
    source.contains(
      "selected_indices[output_base + (unsigned long long)slot] = -1",
    ),
  )
  assert_false(candidate.manifest_bindable())
  let repeated = lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), repeated.source_digest())
  assert_eq(candidate.recipe_digest(), repeated.recipe_digest())
  let foreign = lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
    aot_test_requirements(content_fill='c'),
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.source_digest(), foreign.source_digest())
  assert_true(candidate.recipe_digest() != foreign.recipe_digest())
  assert_true(candidate.source_identity() != foreign.source_identity())
}

///|
test "selected-weight finalizer lowers exact gather normalize scale ABI" {
  let requirements = aot_test_requirements()
  let ordinal = aot_test_finalize_ordinal(requirements)
  let candidate = lower_selected_expert_weight_finalize_cuda_aot_candidate(
    requirements,
    requirement_ordinal=ordinal,
    maximum_tokens=32,
    target=aot_test_target(),
    compiler=aot_test_compiler(),
    limits=aot_test_limits(),
  )
  assert_eq(candidate.family(), SelectedExpertWeightFinalizeF32)
  assert_eq(candidate.numerical_contract(), OrderedF32NormalizeScaleV1)
  assert_eq(
    candidate.operation(),
    SelectedExpertWeightFinalize(8, 2, true, 1.5),
  )
  assert_eq(candidate.capability(), SelectedExpertWeightFinalizeAot)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.operands().length(), 4)
  assert_eq(candidate.operands()[0].role(), StepCountsInput)
  assert_eq(candidate.operands()[0].byte_count(), 20L)
  assert_eq(candidate.operands()[1].role(), RoutingScoresInput)
  assert_eq(candidate.operands()[1].byte_count(), 1024L)
  assert_eq(candidate.operands()[2].role(), RoutingIndicesInput)
  assert_eq(candidate.operands()[2].byte_count(), 256L)
  assert_eq(candidate.operands()[3].role(), RoutingWeightsOutput)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("__fadd_rn(sum, value)"))
  assert_true(source.contains("__fdiv_rn(value, sum)"))
  assert_true(source.contains("__fmul_rn(normalized, 1.5f)"))
  assert_true(source.contains("previous < slot"))
  assert_false(candidate.manifest_bindable())
}

///|
test "routing selection lowerers reject ordinal target compiler and byte overflow" {
  let requirements = aot_test_requirements()
  let top_k_ordinal = aot_test_top_k_ordinal(requirements)
  let finalize_ordinal = aot_test_finalize_ordinal(requirements)
  try
    lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
      requirements,
      requirement_ordinal=finalize_ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, finalize_ordinal)
    _ => fail("unexpected adjacent top-k ordinal rejection")
  } noraise {
    _ => fail("selected-weight finalizer was lowered as top-k")
  }
  try
    lower_selected_expert_weight_finalize_cuda_aot_candidate(
      requirements,
      requirement_ordinal=top_k_ordinal,
      maximum_tokens=1,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    UnsupportedRequirement(actual) => assert_eq(actual, top_k_ordinal)
    _ => fail("unexpected adjacent finalizer ordinal rejection")
  } noraise {
    _ => fail("top-k was lowered as selected-weight finalization")
  }
  for
    compiler in [
      aot_test_compiler(reassociate=true),
      aot_test_compiler(fused_multiply_add=true),
    ] {
    try
      lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
        requirements,
        requirement_ordinal=top_k_ordinal,
        maximum_tokens=1,
        target=aot_test_target(),
        compiler~,
        limits=aot_test_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected routing compiler rejection")
    } noraise {
      _ => fail("unsafe routing compiler policy was admitted")
    }
  }
  try
    lower_selected_expert_weight_finalize_cuda_aot_candidate(
      requirements,
      requirement_ordinal=finalize_ordinal,
      maximum_tokens=1,
      target=aot_test_target(bf16=false),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected routing target rejection")
  } noraise {
    _ => fail("non-BF16 decoder target was admitted")
  }
  let oversized = aot_test_requirements(routed_expert_count=600000)
  try
    lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
      oversized,
      requirement_ordinal=aot_test_top_k_ordinal(oversized),
      maximum_tokens=4096,
      target=aot_test_target(),
      compiler=aot_test_compiler(),
      limits=aot_test_limits(),
    )
  catch {
    SizeOverflow => ()
    _ => fail("unexpected routing byte-overflow rejection")
  } noraise {
    _ => fail("oversized routing score operand was admitted")
  }
}
