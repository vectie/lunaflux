///|
test "official Flash gated RMSNorm joins recurrent gate and o_norm tensors" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_gated_rms_norm_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_gated_norm_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaGatedRmsNorm(64, 128, 0.00001))
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let prefix = "model.language_model.layers.0.self_attn."
  let norm = find_named_tensor(manifest, prefix + "o_norm.weight")
  assert_eq(norm.dtype(), BFloat16)
  assert_eq(norm.shape(), [128L])
  assert_eq(candidate.operands()[3].byte_count(), 256L)
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut norm_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == KdaGatedRmsNorm {
      norm_count += 1
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
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        operation.execution().output(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
    }
  }
  assert_eq(norm_count, 34)
  let gate = @glm_hybrid_cuda_aot.lower_kda_output_gate_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_output_gate_ordinal(requirements),
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
    gate.operands()[5].byte_count(),
    candidate.operands()[2].byte_count(),
  )
  assert_eq(
    recurrent.operands()[8].byte_count(),
    candidate.operands()[1].byte_count(),
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("__fdiv_rn(sum_sq, 128.0f)"))
  assert_true(source.contains("expf(-gate)"))
  assert_true(source.contains("weight[column]), normalized"))
  assert_false(source.contains("o_proj"))
  inspect(
    candidate.source_digest().as_hex(),
    content="6d42f4de233a6ccb01822d4950036f12acadec9a4bc0acbde33138a4651adde8",
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content="c31285da1f675e4f3d0be549ee63f0ca99e720832e5b7a54882f5b569ef37d22",
  )
  assert_false(candidate.manifest_bindable())
}

///|
test "official Full profile has no KDA gated RMSNorm candidate" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is KdaGatedRmsNorm(_, _, _)),
  )
}
