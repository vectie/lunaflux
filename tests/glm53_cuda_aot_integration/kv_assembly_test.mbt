///|
fn find_kv_assembly_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaKeyValueAttentionAssembly(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing DSA KV assembly")
}

///|
test "official Full and Flash KV assembly joins KV-B, rotary, and sparse attention" {
  let full = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full_candidate = try! @glm_hybrid_cuda_aot.lower_dsa_kv_attention_assembly_cuda_aot_candidate(
    full,
    requirement_ordinal=find_kv_assembly_ordinal(full),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_candidate = try! @glm_hybrid_cuda_aot.lower_dsa_kv_attention_assembly_cuda_aot_candidate(
    flash,
    requirement_ordinal=find_kv_assembly_ordinal(flash),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_kv_b = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    full,
    requirement_ordinal=find_key_value_b_ordinal(full),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_kv_b = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    flash,
    requirement_ordinal=find_key_value_b_ordinal(flash),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_rope = @glm_hybrid_cuda_aot.lower_dsa_interleaved_rotary_cuda_aot_candidate(
    full,
    requirement_ordinal=find_rotary_ordinal(full),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_sparse = @glm_hybrid_cuda_aot.lower_dsa_sparse_attention_cuda_aot_candidate(
    full,
    requirement_ordinal=find_dsa_sparse_attention_ordinal(full),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_sparse = @glm_hybrid_cuda_aot.lower_dsa_sparse_attention_cuda_aot_candidate(
    flash,
    requirement_ordinal=find_dsa_sparse_attention_ordinal(flash),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    full_candidate.operation(),
    DsaKeyValueAttentionAssembly(64, 192, 64, 256),
  )
  assert_eq(
    flash_candidate.operation(),
    DsaKeyValueAttentionAssembly(64, 256, 0, 256),
  )
  assert_true(
    full.requirements()[find_kv_assembly_ordinal(full) - 1].operation()
    is GlmRotaryPosition(_, _),
  )
  assert_eq(full_candidate.operands()[1].role(), DsaKeyValueHeadsOutput)
  assert_eq(full_candidate.operands()[2].role(), DsaAttentionKeyRopeInput)
  assert_eq(full_candidate.operands()[3].role(), DsaAttentionKeysInput)
  assert_eq(full_candidate.operands()[4].role(), DsaAttentionValuesInput)
  assert_eq(flash_candidate.operands()[2].role(), DsaAttentionKeysInput)
  assert_eq(flash_candidate.operands()[3].role(), DsaAttentionValuesInput)
  assert_eq(
    full_kv_b.operands()[3].byte_count(),
    full_candidate.operands()[1].byte_count(),
  )
  assert_eq(
    flash_kv_b.operands()[3].byte_count(),
    flash_candidate.operands()[1].byte_count(),
  )
  assert_eq(
    full_rope.operands()[5].byte_count(),
    full_candidate.operands()[2].byte_count(),
  )
  assert_eq(
    full_candidate.operands()[3].byte_count(),
    full_sparse.operands()[5].byte_count(),
  )
  assert_eq(
    full_candidate.operands()[4].byte_count(),
    full_sparse.operands()[6].byte_count(),
  )
  assert_eq(
    flash_candidate.operands()[2].byte_count(),
    flash_sparse.operands()[5].byte_count(),
  )
  assert_eq(
    flash_candidate.operands()[3].byte_count(),
    flash_sparse.operands()[6].byte_count(),
  )
  assert_true(
    full
    .requirements()
    .any(item => item.operation() is DsaSparseAttention(_, _, _, _)),
  )
  assert_false(full_candidate.manifest_bindable())
  assert_false(flash_candidate.manifest_bindable())
  ignore(try! @glm_hybrid_candidate_evidence.from_kv_assembly(full_candidate))
  ignore(try! @glm_hybrid_candidate_evidence.from_kv_assembly(flash_candidate))
  inspect(
    full_candidate.source_digest().as_hex() +
    ":" +
    full_candidate.recipe_digest().as_hex() +
    ":" +
    flash_candidate.source_digest().as_hex() +
    ":" +
    flash_candidate.recipe_digest().as_hex(),
    content="dddbdff15931e0181ce2931d405932da33a4e11afae3106fee47e970eec33d38:6f2e0af68c97ce60a2940229fad8ae9e9e0d42708a5d9421784d05e1a42ba951:378ed733a6e112f06ae0992b1ad0427fdfe76687e53f64b6a63a6ba0cdcdd015:7fcc2003617dc8d4bab22bc529994549fcd726a697da85e9b73f384b16d8f81c",
  )
}
