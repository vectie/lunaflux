///|
test "official Full interleaved RoPE joins Query-B and preserved KV-A suffix" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let rotary = @glm_hybrid_cuda_aot.lower_dsa_interleaved_rotary_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_rotary_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let query_b = @glm_hybrid_cuda_aot.lower_dsa_query_b_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_query_b_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let kv_split = @glm_hybrid_cuda_aot.lower_dsa_key_value_split_norm_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kv_split_norm_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(rotary.operation(), GlmRotaryPosition(Some(8000000.0), true))
  assert_eq(rotary.maximum_positions(), 1048576)
  assert_eq(
    rotary.operands()[2].byte_count(),
    query_b.operands()[3].byte_count(),
  )
  assert_eq(
    rotary.operands()[3].byte_count(),
    kv_split.operands()[4].byte_count(),
  )
  assert_eq(rotary.operands()[4].role(), DsaAttentionQueriesInput)
  assert_eq(rotary.operands()[5].role(), DsaAttentionKeyRopeInput)
  let manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let query_weight = find_named_tensor(
    manifest, "model.layers.0.self_attn.q_b_proj.weight",
  )
  let key_weight = find_named_tensor(
    manifest, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
  )
  assert_eq(query_weight.shape(), [16384L, 2048L])
  assert_eq(key_weight.shape(), [576L, 6144L])
  assert_eq(query_weight.dtype(), BFloat16)
  assert_eq(key_weight.dtype(), BFloat16)
  let source = @utf8.decode(rotary.source_bytes())
  assert_true(source.contains("output_base + 32ULL"))
  assert_true(
    source.contains("query_output[query_base + column] = query_input"),
  )
  assert_true(
    @utf8.decode(rotary.recipe_bytes()).contains(
      "excluded=kv-b-key-nope-value-expansion-cache-and-attention",
    ),
  )
  assert_false(rotary.manifest_bindable())
}

///|
test "official Flash zero-width RoPE remains an explicit candidate gap" {
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  try
    @glm_hybrid_cuda_aot.lower_dsa_interleaved_rotary_cuda_aot_candidate(
      requirements,
      requirement_ordinal=find_rotary_ordinal(requirements),
      maximum_tokens=32,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong Flash no-op rotary error")
  } noraise {
    _ => fail("Flash no-op rotary received invented executable work")
  }
}
