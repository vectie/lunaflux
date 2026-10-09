///|
test "official Flash KDA short convolution joins config weights numeric plan and state ABI" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_short_convolution_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_short_convolution_ordinal(requirements),
    maximum_tokens=32,
    maximum_sequences=8,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaShortConvolution(8192, 4))
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let q = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.q_conv1d.weight",
  )
  let k = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.k_conv1d.weight",
  )
  let v = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.v_conv1d.weight",
  )
  for tensor in [q, k, v] {
    assert_eq(tensor.dtype(), BFloat16)
    assert_eq(tensor.shape(), [8192L, 1L, 4L])
  }
  let mut convolution_weight_count = 0
  let mut convolution_bias_count = 0
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.name().contains(".self_attn.") &&
      tensor.name().has_suffix("_conv1d.weight") {
      convolution_weight_count += 1
    }
    if tensor.name().contains(".self_attn.") &&
      tensor.name().contains("_conv1d.bias") {
      convolution_bias_count += 1
    }
  }
  assert_eq(convolution_weight_count, 102)
  assert_eq(convolution_bias_count, 0)
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut numeric_short_convolution_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == KdaShortConvolution {
      numeric_short_convolution_count += 1
      assert_eq(
        operation.execution().accumulator(),
        @numeric_contract.AccumulatorDType::f32(),
      )
      assert_eq(
        operation.execution().output(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
    }
  }
  assert_eq(numeric_short_convolution_count, 34)
  assert_eq(candidate.operands()[3].byte_count(), 65536L)
  assert_eq(candidate.operands()[4].byte_count(), 393216L)
  assert_eq(candidate.operands()[6].byte_count(), 393216L)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("counts[2]"))
  assert_true(source.contains("counts[3]"))
  assert_true(source.contains("sequence_offsets[sequence_count] != live_rows"))
  assert_true(source.contains("__float2bfloat16_rn(__fdiv_rn"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("invocations=independent-q-k-v"))
  assert_true(recipe.contains("zero-history=all-zero-initial-state"))
  assert_true(recipe.contains("state-copy=bit-exact-bf16"))
  ignore(
    try! @glm_hybrid_candidate_evidence.from_kda_short_convolution(candidate),
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "Full profile has no KDA short-convolution seam" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements
    .requirements()
    .any(requirement => requirement.operation() is KdaShortConvolution(_, _)),
  )
}
