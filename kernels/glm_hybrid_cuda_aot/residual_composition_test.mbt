///|
fn residual_requirement_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeDecoderResidualAdd(_) ||
      requirement.operation() is HyperConnectionMix(_) {
      return ordinal
    }
  }
  abort("missing residual-composition requirement")
}

///|
test "Full residual add and Flash hyper mix bind distinct exact ABIs" {
  let full = glm_aot_requirements(official_sparse=true)
  let flash = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    hyper_connection=true,
    routed_expert_count=288,
    experts_per_token=8,
    moe_intermediate_size=2048,
    moe_activation=ClampedSwiGlu(10.0),
  )
  let full_candidate = lower_moe_decoder_residual_add_cuda_aot_candidate(
    full,
    requirement_ordinal=residual_requirement_ordinal(full),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let flash_candidate = lower_hyper_connection_mix_cuda_aot_candidate(
    flash,
    requirement_ordinal=residual_requirement_ordinal(flash),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(full_candidate.family(), FullDecoderResidualAdd)
  assert_eq(full_candidate.operation(), MoeDecoderResidualAdd(6144))
  assert_eq(full_candidate.operands()[1].role(), DecoderResidualInput)
  assert_eq(full_candidate.operands()[2].role(), MoeCombinedOutput)
  assert_eq(full_candidate.operands()[3].role(), DecoderResidualOutput)
  assert_eq(flash_candidate.family(), FlashHyperConnectionMix4)
  assert_eq(flash_candidate.operation(), HyperConnectionMix(4))
  assert_eq(flash_candidate.operands()[1].role(), HyperPostControlInput)
  assert_eq(flash_candidate.operands()[1].byte_count(), 512L)
  assert_eq(flash_candidate.operands()[2].byte_count(), 2048L)
  assert_eq(flash_candidate.operands()[3].byte_count(), 1048576L)
  assert_eq(flash_candidate.operands()[4].byte_count(), 262144L)
  assert_eq(flash_candidate.operands()[5].role(), HyperMixedStreamsOutput)
  let source = @utf8.decode(flash_candidate.source_bytes())
  assert_true(source.contains("output == residual || output == branch"))
  assert_true(source.contains("source) * 4ULL + destination"))
  assert_false(source.contains("sigmoid"))
  assert_false(full_candidate.manifest_bindable())
  assert_false(flash_candidate.manifest_bindable())
}

///|
test "residual composition lowerers reject wrong stage and compiler" {
  let full = glm_aot_requirements(official_sparse=true)
  let ordinal = residual_requirement_ordinal(full)
  try
    lower_hyper_connection_mix_cuda_aot_candidate(
      full,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected operation error")
  } noraise {
    _ => fail("plain residual admitted as hyper mix")
  }
  try
    lower_moe_decoder_residual_add_cuda_aot_candidate(
      full,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=try! @luna_specialization.LunaCompilerPolicy::new(
        identity=glm_aot_compiler().identity(),
        optimization_level=3,
        fused_multiply_add=true,
        reassociate=false,
        max_registers=128,
      ),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler error")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
}
