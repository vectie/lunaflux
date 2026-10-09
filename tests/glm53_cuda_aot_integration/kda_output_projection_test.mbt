///|
test "official Flash KDA output projection joins o_proj and gated norm output" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_output_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_output_projection_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaOutputProjection(8192, 4096))
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let tensor = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.o_proj.weight",
  )
  assert_eq(tensor.dtype(), BFloat16)
  assert_eq(tensor.shape(), [4096L, 8192L])
  assert_eq(candidate.operands()[2].byte_count(), 67108864L)
  let norm = @glm_hybrid_cuda_aot.lower_kda_gated_rms_norm_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_gated_norm_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    norm.operands()[4].byte_count(),
    candidate.operands()[1].byte_count(),
  )
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == KdaOutputProjection {
      count += 1
      assert_eq(
        operation.execution().activation_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        operation.execution().tensor_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
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
  assert_eq(count, 34)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("weight_base = column * 8192ULL"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__float2bfloat16_rn"))
  assert_false(source.contains("rsqrt"))
  assert_false(source.contains("residual"))
  inspect(
    candidate.source_digest().as_hex(),
    content="cb2f5b3f789edf53b06ad0c4eba306fd49de313b1726359baf0a1607fcde712a",
  )
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("excluded=gated-rmsnorm-residual-cache"))
  inspect(
    candidate.recipe_digest().as_hex(),
    content="20fa381266de752f26ed4e170892fdd1dcd15507b3172491082c8a50b3978d13",
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "official Full profile has no KDA output projection candidate" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is KdaOutputProjection(_, _)),
  )
}
