///|
test "official Full and Flash dense SwiGLU join separate BF16 weights" {
  let full_requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  let flash_requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let full = @glm_hybrid_cuda_aot.lower_dense_swiglu_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=find_dense_swiglu_ordinal(full_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let flash = @glm_hybrid_cuda_aot.lower_dense_swiglu_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=find_dense_swiglu_ordinal(flash_requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(full.operation(), DenseSwiGlu(6144, 12288))
  assert_eq(flash.operation(), DenseSwiGlu(4096, 12288))
  let full_manifest = try! @glm53_weights.build_full_tensor_manifest(
    full_metadata(),
    BFloat16WithFloat32Routing,
  )
  let flash_manifest = try! @glm53_weights.build_flash_tensor_manifest(
    flash_metadata(),
    BFloat16WithFloat32Routing,
  )
  let full_gate = find_named_tensor(
    full_manifest, "model.layers.0.mlp.gate_proj.weight",
  )
  let full_up = find_named_tensor(
    full_manifest, "model.layers.0.mlp.up_proj.weight",
  )
  let full_down = find_named_tensor(
    full_manifest, "model.layers.0.mlp.down_proj.weight",
  )
  let flash_gate = find_named_tensor(
    flash_manifest, "model.language_model.layers.0.mlp.gate_proj.weight",
  )
  let flash_up = find_named_tensor(
    flash_manifest, "model.language_model.layers.0.mlp.up_proj.weight",
  )
  let flash_down = find_named_tensor(
    flash_manifest, "model.language_model.layers.0.mlp.down_proj.weight",
  )
  assert_eq(full_gate.shape(), [12288L, 6144L])
  assert_eq(full_up.shape(), [12288L, 6144L])
  assert_eq(full_down.shape(), [6144L, 12288L])
  assert_eq(flash_gate.shape(), [12288L, 4096L])
  assert_eq(flash_up.shape(), [12288L, 4096L])
  assert_eq(flash_down.shape(), [4096L, 12288L])
  for
    tensor in [full_gate, full_up, full_down, flash_gate, flash_up, flash_down] {
    assert_eq(tensor.dtype(), BFloat16)
  }
  assert_eq(full.operands()[2].role(), DenseGateWeights)
  assert_eq(full.operands()[3].role(), DenseUpWeights)
  assert_eq(full.operands()[4].role(), DenseDownWeights)
  assert_eq(flash.operands()[2].byte_count(), 100663296L)
  assert_true(
    @utf8.decode(full.recipe_bytes()).contains(
      "excluded=residual-norm-routing-and-shared-experts",
    ),
  )
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}
