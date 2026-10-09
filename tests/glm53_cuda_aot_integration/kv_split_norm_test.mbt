///|
test "official Full and Flash KV-A split RMSNorm joins weights and KV-B input" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full = @glm_hybrid_cuda_aot.lower_dsa_key_value_split_norm_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_kv_split_norm_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_dsa_key_value_split_norm_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_kv_split_norm_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), QueryKeyRmsNorm(0.00001))
  assert_eq(flash.operation(), QueryKeyRmsNorm(0.00001))
  assert_eq(full.prefix_width(), 512)
  assert_eq(flash.prefix_width(), 512)
  assert_eq(full.suffix_width(), 64)
  assert_eq(flash.suffix_width(), 0)
  assert_eq(full.operands().length(), 5)
  assert_eq(flash.operands().length(), 4)
  assert_eq(full.operands()[1].byte_count(), 36864L)
  assert_eq(flash.operands()[1].byte_count(), 32768L)
  assert_eq(full.operands()[2].byte_count(), 1024L)
  assert_eq(flash.operands()[2].byte_count(), 1024L)
  assert_eq(full.operands()[3].byte_count(), 32768L)
  assert_eq(flash.operands()[3].byte_count(), 32768L)
  assert_eq(full.operands()[4].byte_count(), 4096L)
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  let full_weight = find_named_tensor(
    full_manifest, "model.layers.0.self_attn.kv_a_layernorm.weight",
  )
  let flash_weight = find_named_tensor(
    flash_manifest, "model.language_model.layers.3.self_attn.kv_a_layernorm.weight",
  )
  assert_eq(full_weight.shape(), [512L])
  assert_eq(flash_weight.shape(), [512L])
  assert_eq(full_weight.dtype(), BFloat16)
  assert_eq(flash_weight.dtype(), BFloat16)
  let full_kv_b = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_key_value_b_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash_kv_b = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_key_value_b_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    full.operands()[3].byte_count(),
    full_kv_b.operands()[1].byte_count(),
  )
  assert_eq(
    flash.operands()[3].byte_count(),
    flash_kv_b.operands()[1].byte_count(),
  )
  assert_true(
    @utf8.decode(full.recipe_bytes()).contains(
      "downstream=kv-b-normalized-input-compatible",
    ),
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
