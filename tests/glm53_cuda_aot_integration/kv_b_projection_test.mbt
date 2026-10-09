///|
test "official Full and Flash kv-B projection binds key-nope and value layout" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_key_value_b_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_key_value_b_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), KeyValueLowRankProjection(512, 64, 192, 256))
  assert_eq(flash.operation(), KeyValueLowRankProjection(512, 64, 256, 256))
  assert_eq(full.output_width(), 28672)
  assert_eq(flash.output_width(), 32768)
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  assert_eq(
    find_named_tensor(
      full_manifest, "model.layers.0.self_attn.kv_b_proj.weight",
    ).shape(),
    [28672L, 512L],
  )
  assert_eq(
    find_named_tensor(
      flash_manifest, "model.language_model.layers.3.self_attn.kv_b_proj.weight",
    ).shape(),
    [32768L, 512L],
  )
  assert_true(
    @utf8.decode(full.recipe_bytes()).contains(
      "excluded=kv-a-split-norm-and-rope-key",
    ),
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
