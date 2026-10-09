///|
test "official Flash decay control joins raw forget A_log dt_bias and recurrence" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_decay_control_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_decay_control_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaDecayControl(64, 128, -5.0))
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let prefix = "model.language_model.layers.0.self_attn."
  let a_log = find_named_tensor(manifest, prefix + "A_log")
  let bias = find_named_tensor(manifest, prefix + "dt_bias")
  assert_eq(a_log.dtype(), Float32)
  assert_eq(a_log.shape(), [64L])
  assert_eq(bias.dtype(), Float32)
  assert_eq(bias.shape(), [8192L])
  assert_eq(candidate.operands()[2].byte_count(), 32768L)
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  let projection = @glm_hybrid_cuda_aot.lower_kda_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_projection_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let recurrent = @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_recurrent_ordinal(requirements),
    maximum_tokens=32,
    maximum_sequences=8,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    projection.operands()[12].byte_count(),
    candidate.operands()[1].byte_count(),
  )
  assert_eq(
    candidate.operands()[4].byte_count(),
    recurrent.operands()[5].byte_count(),
  )
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == KdaF32DecayControl {
      count += 1
      assert_eq(
        operation.execution().activation_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        operation.execution().activation_compute(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        operation.execution().tensor_input(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        operation.execution().output(),
        @numeric_contract.ComputeDType::f32(),
      )
    }
  }
  assert_eq(count, 34)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("expf(a_log[head])"))
  assert_true(source.contains("dt_bias[component]"))
  assert_true(source.contains("expf(-scaled)"))
  assert_true(source.contains("__fmul_rn(-5.0f, sigmoid)"))
  assert_false(source.contains("initial_state"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("excluded=qkv-beta-projection-recurrence-cache"))
  inspect(
    candidate.source_digest().as_hex(),
    content="14652cc6b3db6515846d283719595f355fb345a099870c2f0bdf714346c4ba57",
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content="04dd74298b5af717cab3aebc7f20d36c999c8711f611aa28c31603f30ace3e5e",
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "official Full profile has no KDA decay-control candidate" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is KdaDecayControl(_, _, _)),
  )
}
