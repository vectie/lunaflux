///|
test "official Flash KDA projection joins exact BF16 tensors and F32 decay gap" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_projection_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaProjection(64, 128))
  assert_eq(candidate.capability(), KdaProjectionAot)
  assert_eq(candidate.hidden_width(), 4096)
  assert_eq(candidate.head_count(), 64)
  assert_eq(candidate.head_dimension(), 128)
  assert_eq(candidate.control_rank(), 128)
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let prefix = "model.language_model.layers.0.self_attn."
  let expected = [
    ("q_proj.weight", [8192L, 4096L]),
    ("k_proj.weight", [8192L, 4096L]),
    ("v_proj.weight", [8192L, 4096L]),
    ("f_a_proj.weight", [128L, 4096L]),
    ("f_b_proj.weight", [8192L, 128L]),
    ("b_proj.weight", [64L, 4096L]),
  ]
  for entry in expected {
    let (suffix, shape) = entry
    let tensor = find_named_tensor(manifest, prefix + suffix)
    assert_eq(tensor.dtype(), BFloat16)
    assert_eq(tensor.shape(), ReadOnlyArray::from_array(shape))
  }
  let decay_rate = find_named_tensor(manifest, prefix + "A_log")
  let decay_bias = find_named_tensor(manifest, prefix + "dt_bias")
  assert_eq(decay_rate.dtype(), Float32)
  assert_eq(decay_rate.shape(), [64L])
  assert_eq(decay_bias.dtype(), Float32)
  assert_eq(decay_bias.shape(), [8192L])
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut projection_count = 0
  let mut decay_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    match operation.kind() {
      KdaProjection => {
        projection_count += 1
        assert_eq(
          operation.execution().tensor_input(),
          @numeric_contract.ComputeDType::bfloat16(),
        )
        assert_eq(
          operation.execution().output(),
          @numeric_contract.ComputeDType::bfloat16(),
        )
      }
      KdaF32DecayControl => {
        decay_count += 1
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
      _ => ()
    }
  }
  assert_eq(projection_count, 34)
  assert_eq(decay_count, 34)
  assert_true(
    requirements
    .requirements()
    .any(r => r.operation() == KdaDecayControl(64, 128, -5.0)),
  )
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("__nv_bfloat16 *forget_low_rank"))
  assert_true(source.contains("__nv_bfloat16 *raw_forget"))
  assert_false(source.contains("dt_bias"))
  assert_false(source.contains("A_log"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("excluded=f32-decay-control"))
  inspect(
    candidate.source_digest().as_hex(),
    content="f900384c10b7cf62e9392cc3b6cd218a7580ae0e8d52f49b9ae2161865e1e2b5",
  )
  inspect(
    candidate.requirements_digest().as_hex(),
    content="d22c4f7cd09a8e932a800899f44dd2e011c13cad076018b9f569f35b4ba4e9d4",
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content="6f3f6306459a9184467cb426b956b0bfd94b5b483a47b62c95063efd0b8d0988",
  )
  ignore(try! @glm_hybrid_candidate_evidence.from_kda_projection(candidate))
  assert_false(candidate.manifest_bindable())
}

///|
test "Full profile has neither KDA projection nor decay-control seam" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements.requirements().any(r => r.operation() is KdaProjection(_, _)),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is KdaDecayControl(_, _, _)),
  )
}
