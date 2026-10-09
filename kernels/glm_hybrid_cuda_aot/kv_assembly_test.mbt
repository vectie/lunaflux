///|
fn kv_assembly_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, item in requirements.requirements() {
    if item.operation() is DsaKeyValueAttentionAssembly(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing KV assembly")
}

///|
test "KV assembly lowerer binds exact Full and Flash layouts" {
  let full = glm_aot_requirements(official_sparse=true, rope_theta=8000000.0)
  let flash = glm_aot_requirements(official_sparse=true, pooled=true)
  let full_candidate = lower_dsa_kv_attention_assembly_cuda_aot_candidate(
    full,
    requirement_ordinal=kv_assembly_ordinal(full),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let flash_candidate = lower_dsa_kv_attention_assembly_cuda_aot_candidate(
    flash,
    requirement_ordinal=kv_assembly_ordinal(flash),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(full_candidate.family(), FullSharedRopeSuffixBroadcast)
  assert_eq(flash_candidate.family(), FlashNoRopeSplit)
  assert_eq(full_candidate.operands().length(), 5)
  assert_eq(flash_candidate.operands().length(), 4)
  assert_eq(full_candidate.operands()[1].role(), DsaKeyValueHeadsOutput)
  assert_eq(full_candidate.operands()[1].byte_count(), 1835008L)
  assert_eq(full_candidate.operands()[2].role(), DsaAttentionKeyRopeInput)
  assert_eq(full_candidate.operands()[2].byte_count(), 4096L)
  assert_eq(full_candidate.operands()[3].role(), DsaAttentionKeysInput)
  assert_eq(full_candidate.operands()[3].byte_count(), 1048576L)
  assert_eq(full_candidate.operands()[4].role(), DsaAttentionValuesInput)
  assert_eq(flash_candidate.operands()[1].byte_count(), 2097152L)
  assert_eq(flash_candidate.operands()[2].byte_count(), 1048576L)
  let full_source = @utf8.decode(full_candidate.source_bytes())
  let flash_source = @utf8.decode(flash_candidate.source_bytes())
  assert_true(full_source.contains("shared_key_suffix"))
  assert_false(flash_source.contains("shared_key_suffix"))
  assert_false(full_candidate.manifest_bindable())
}

///|
test "KV assembly lowering rejects neighboring operation and unsafe compiler" {
  let full = glm_aot_requirements(official_sparse=true, rope_theta=8000000.0)
  try
    lower_dsa_kv_attention_assembly_cuda_aot_candidate(
      full,
      requirement_ordinal=kv_assembly_ordinal(full) - 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected ordinal error")
  } noraise {
    _ => fail("neighboring operation admitted")
  }
  try
    lower_dsa_kv_attention_assembly_cuda_aot_candidate(
      full,
      requirement_ordinal=kv_assembly_ordinal(full),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=try! @luna_specialization.LunaCompilerPolicy::new(
        identity=glm_aot_compiler().identity(),
        optimization_level=3,
        fused_multiply_add=false,
        reassociate=true,
        max_registers=128,
      ),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected compiler error")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
}
