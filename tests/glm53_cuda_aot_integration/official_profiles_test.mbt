///|
test "official Flash DSA key pooling is exact while Full fails closed" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  assert_eq(find_dsa_pooling_ordinal(full_requirements), None)
  guard find_dsa_pooling_ordinal(flash_requirements) is Some(flash_ordinal) else {
    fail("official Flash pooling requirement missing")
  }
  let flash = @glm_hybrid_cuda_aot.lower_dsa_key_pooling_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=flash_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(flash.operation(), DsaIndexPooling(4, true, true))
  assert_eq(flash.head_dimension(), 128)
  assert_eq(flash.pool_size(), 4)
  assert_eq(flash.pool_capacity(), 8)
  assert_eq(flash.operands()[0].alignment(), 4L)
  assert_eq(flash.operands()[1].byte_count(), 8192L)
  assert_eq(flash.operands()[2].byte_count(), 8192L)
  assert_eq(flash.operands()[3].byte_count(), 128L)
  assert_eq(flash.operands()[4].byte_count(), 1024L)
  assert_eq(flash.operands()[5].byte_count(), 2048L)
  assert_eq(flash.operands()[6].byte_count(), 128L)
  assert_eq(flash.operands()[7].byte_count(), 32L)
  assert_eq(flash.operands()[8].byte_count(), 4L)
  assert_false(flash.manifest_bindable())
  let full_neighbor = find_dsa_index_compute_ordinal(full_requirements)
  try
    @glm_hybrid_cuda_aot.lower_dsa_key_pooling_cuda_aot_candidate(
      full_requirements,
      requirement_ordinal=full_neighbor,
      maximum_tokens=32,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    )
  catch {
    UnsupportedRequirement(ordinal) => assert_eq(ordinal, full_neighbor)
    _ => fail("unexpected official Full rejection")
  } noraise {
    _ => fail("official Full admitted a nonexistent pooling stage")
  }
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_numeric = @glm53_numeric_plan.build_full_numeric_plan(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  assert_eq(count_exact_pool_ape_tensors(full_manifest), 0)
  assert_eq(count_exact_pool_ape_tensors(flash_manifest), 12)
  assert_eq(count_numeric_dsa_pooling(full_numeric), 0)
  assert_eq(count_numeric_dsa_pooling(flash_numeric), 12)
}

///|
test "official Full and Flash DSA index selection preserve profile differences" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full = @glm_hybrid_cuda_aot.lower_dsa_index_selection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_dsa_index_compute_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_dsa_index_selection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_dsa_index_compute_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), DsaIndexCompute(32, 128, 2048, true))
  assert_eq(flash.operation(), DsaIndexCompute(32, 128, 2048, true))
  assert_eq(full.family(), RawTokenScoreStableTopKBf16)
  assert_eq(flash.family(), PooledScoreStableTopKWithTailBf16)
  assert_eq(full.pool_size(), 1)
  assert_eq(flash.pool_size(), 4)
  assert_eq(full.candidate_capacity(), 32)
  assert_eq(flash.candidate_capacity(), 8)
  assert_eq(full.output_width(), 2048)
  assert_eq(flash.output_width(), 2051)
  assert_eq(full.operands()[2].byte_count(), 262144L)
  assert_eq(flash.operands()[2].byte_count(), 262144L)
  assert_eq(full.operands()[4].byte_count(), 8192L)
  assert_eq(flash.operands()[4].byte_count(), 2048L)
  assert_eq(full.operands()[9].byte_count(), 262144L)
  assert_eq(flash.operands()[9].byte_count(), 262528L)
  guard find_dsa_reuse_ordinal(full_requirements) is Some(reuse_ordinal) else {
    fail("official Full DSA reuse requirement missing")
  }
  assert_eq(find_dsa_reuse_ordinal(flash_requirements), None)
  let reuse = @glm_hybrid_cuda_aot.lower_dsa_index_reuse_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=reuse_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(reuse.family(), SelectedIndexReuseI32)
  assert_eq(reuse.output_width(), 2048)
  assert_eq(reuse.operands()[1].byte_count(), 262144L)
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
  assert_false(reuse.manifest_bindable())
}

///|
test "official Full and Flash DSA A projections bind exact weight geometry" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full_q = @glm_hybrid_cuda_aot.lower_dsa_query_a_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_query_a_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_q = @glm_hybrid_cuda_aot.lower_dsa_query_a_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_query_a_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_kv = @glm_hybrid_cuda_aot.lower_dsa_key_value_a_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_kv_a_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_kv = @glm_hybrid_cuda_aot.lower_dsa_key_value_a_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_kv_a_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_q_norm = @glm_hybrid_cuda_aot.lower_foundation_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_query_a_norm_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_q_norm = @glm_hybrid_cuda_aot.lower_foundation_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_query_a_norm_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let full_q_b = @glm_hybrid_cuda_aot.lower_dsa_query_b_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_query_b_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_q_b = @glm_hybrid_cuda_aot.lower_dsa_query_b_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_query_b_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full_q.operation(), DsaQueryAProjection(6144, 2048))
  assert_eq(flash_q.operation(), DsaQueryAProjection(4096, 1536))
  assert_eq(full_kv.operation(), DsaKeyValueAProjection(6144, 512, 64))
  assert_eq(flash_kv.operation(), DsaKeyValueAProjection(4096, 512, 0))
  assert_eq(full_q_norm.operation(), DsaQueryARmsNorm(2048, 0.00001))
  assert_eq(flash_q_norm.operation(), DsaQueryARmsNorm(1536, 0.00001))
  assert_eq(full_q_b.operation(), DsaQueryBProjection(2048, 64, 256))
  assert_eq(flash_q_b.operation(), DsaQueryBProjection(1536, 64, 256))
  assert_eq(full_kv.output_width(), 576)
  assert_eq(flash_kv.output_width(), 512)
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(
    find_named_tensor(full_manifest, "model.layers.0.self_attn.q_a_proj.weight").shape(),
    [2048L, 6144L],
  )
  assert_eq(
    find_named_tensor(
      flash_manifest, "model.language_model.layers.3.self_attn.q_a_proj.weight",
    ).shape(),
    [1536L, 4096L],
  )
  assert_eq(
    find_named_tensor(
      full_manifest, "model.layers.0.self_attn.q_a_layernorm.weight",
    ).shape(),
    [2048L],
  )
  assert_eq(
    find_named_tensor(
      flash_manifest, "model.language_model.layers.3.self_attn.q_a_layernorm.weight",
    ).shape(),
    [1536L],
  )
  assert_eq(
    find_named_tensor(full_manifest, "model.layers.0.self_attn.q_b_proj.weight").shape(),
    [16384L, 2048L],
  )
  assert_eq(
    find_named_tensor(
      flash_manifest, "model.language_model.layers.3.self_attn.q_b_proj.weight",
    ).shape(),
    [16384L, 1536L],
  )
  assert_eq(
    find_named_tensor(
      full_manifest, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    ).shape(),
    [576L, 6144L],
  )
  assert_eq(
    find_named_tensor(
      flash_manifest, "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    ).shape(),
    [512L, 4096L],
  )
  let full_numeric = @glm53_numeric_plan.build_full_numeric_plan(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let flash_numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(count_numeric_dsa_low_rank_projection(full_numeric), 79)
  assert_eq(count_numeric_dsa_low_rank_projection(flash_numeric), 12)
  assert_false(full_q.manifest_bindable())
  assert_false(flash_q.manifest_bindable())
  assert_false(full_kv.manifest_bindable())
  assert_false(flash_kv.manifest_bindable())
  assert_false(full_q_norm.manifest_bindable())
  assert_false(flash_q_norm.manifest_bindable())
  assert_false(full_q_b.manifest_bindable())
  assert_false(flash_q_b.manifest_bindable())
}

///|
test "official Full and Flash selected-index attention share exact QKV geometry" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full = @glm_hybrid_cuda_aot.lower_dsa_sparse_attention_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_dsa_sparse_attention_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_dsa_sparse_attention_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_dsa_sparse_attention_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), DsaSparseAttention(64, 2048, 256, 256))
  assert_eq(flash.operation(), DsaSparseAttention(64, 2048, 256, 256))
  assert_eq(full.key_value_head_count(), 64)
  assert_eq(flash.key_value_head_count(), 64)
  assert_eq(full.selected_width(), 2048)
  assert_eq(flash.selected_width(), 2051)
  assert_eq(full.operands()[7].byte_count(), 262144L)
  assert_eq(flash.operands()[7].byte_count(), 262528L)
  assert_eq(full.operands()[8].byte_count(), 1048576L)
  assert_eq(flash.operands()[8].byte_count(), 1048576L)
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}

///|
test "official full and Flash DSA output projections share one exact adapter" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full_ordinal = find_attention_output_ordinal(full_requirements)
  let flash_ordinal = find_attention_output_ordinal(flash_requirements)
  let full = @glm_hybrid_cuda_aot.lower_attention_output_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=full_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_attention_output_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=flash_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.family(), AttentionOutputProjectionBf16)
  assert_eq(flash.family(), AttentionOutputProjectionBf16)
  assert_eq(full.operation(), AttentionOutputProjection(6144, 16384))
  assert_eq(flash.operation(), AttentionOutputProjection(4096, 16384))
  assert_eq(full.input_width(), 16384)
  assert_eq(flash.input_width(), 16384)
  assert_eq(full.output_width(), 6144)
  assert_eq(flash.output_width(), 4096)
  assert_eq(full.dimensions().grid_x(), 768)
  assert_eq(flash.dimensions().grid_x(), 512)
  assert_eq(full.operands()[0].alignment(), 4L)
  assert_eq(full.operands()[1].byte_count(), 1048576L)
  assert_eq(flash.operands()[1].byte_count(), 1048576L)
  assert_eq(full.operands()[2].byte_count(), 201326592L)
  assert_eq(flash.operands()[2].byte_count(), 134217728L)
  assert_eq(full.operands()[3].byte_count(), 393216L)
  assert_eq(flash.operands()[3].byte_count(), 262144L)
  assert_eq(full.operands()[3].role(), HiddenOutput)
  assert_eq(full.function_symbol(), flash.function_symbol())
  assert_true(full.source_digest() != flash.source_digest())
  assert_true(full.recipe_digest() != flash.recipe_digest())
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  assert_eq(count_attention_output_tensors(full_manifest, [6144L, 16384L]), 79)
  assert_eq(count_attention_output_tensors(flash_manifest, [4096L, 16384L]), 12)
}

///|
test "official full and Flash LM heads lower through one modular contract" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata,
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata,
  )
  let full_ordinal = find_head_ordinal(full_requirements)
  let flash_ordinal = find_head_ordinal(flash_requirements)
  assert_eq(full_ordinal, 27)
  assert_eq(flash_ordinal, 46)
  let full = @glm_hybrid_cuda_aot.lower_language_model_head_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=full_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_language_model_head_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=flash_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), LanguageModelHead(6144, 154880))
  assert_eq(flash.operation(), LanguageModelHead(4096, 154880))
  assert_eq(full.dimensions().grid_x(), 19360)
  assert_eq(flash.dimensions().grid_x(), 19360)
  assert_eq(full.operands()[0].alignment(), 4L)
  assert_eq(full.operands()[1].byte_count(), 393216L)
  assert_eq(flash.operands()[1].byte_count(), 262144L)
  assert_eq(full.operands()[2].byte_count(), 1903165440L)
  assert_eq(flash.operands()[2].byte_count(), 1268776960L)
  assert_eq(full.operands()[3].byte_count(), 9912320L)
  assert_eq(full.operands()[3].role(), LogitsOutput)
  assert_eq(full.function_symbol(), flash.function_symbol())
  assert_true(full.source_digest() != flash.source_digest())
  assert_true(full.recipe_digest() != flash.recipe_digest())
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}

///|
test "official manifests and numeric plans agree without leaking authority" {
  let full_metadata = full_metadata()
  let flash_metadata = flash_metadata()
  let full_manifest = @glm53_weights.build_full_tensor_manifest(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = @glm53_weights.build_flash_tensor_manifest(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_numeric = @glm53_numeric_plan.build_full_numeric_plan(
    full_metadata,
    BFloat16WithFloat32Routing,
  )
  let flash_numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    flash_metadata,
    BFloat16WithFloat32Routing,
  )
  let full_head = find_head_tensor(full_manifest)
  let flash_head = find_head_tensor(flash_manifest)
  assert_eq(full_head.dtype(), BFloat16)
  assert_eq(flash_head.dtype(), BFloat16)
  assert_eq(full_head.shape(), [154880L, 6144L])
  assert_eq(flash_head.shape(), [154880L, 4096L])
  assert_eq(
    full_manifest.execution_identity(),
    full_numeric.execution_identity(),
  )
  assert_eq(
    flash_manifest.execution_identity(),
    flash_numeric.execution_identity(),
  )
  assert_eq(
    full_manifest.layout_digest(),
    full_numeric.manifest_layout_digest(),
  )
  assert_eq(
    flash_manifest.layout_digest(),
    flash_numeric.manifest_layout_digest(),
  )
  assert_true(full_numeric.binding_digest() != flash_numeric.binding_digest())
}
