///|
test "official Flash output gate joins g_a g_b and leaves norm dense typed gaps" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_output_gate_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_output_gate_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaOutputGateProjection(4096, 64, 128))
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let prefix = "model.language_model.layers.0.self_attn."
  let first = find_named_tensor(manifest, prefix + "g_a_proj.weight")
  let second = find_named_tensor(manifest, prefix + "g_b_proj.weight")
  let norm = find_named_tensor(manifest, prefix + "o_norm.weight")
  let output = find_named_tensor(manifest, prefix + "o_proj.weight")
  assert_eq(first.dtype(), BFloat16)
  assert_eq(first.shape(), [128L, 4096L])
  assert_eq(second.dtype(), BFloat16)
  assert_eq(second.shape(), [8192L, 128L])
  assert_eq(norm.shape(), [128L])
  assert_eq(output.shape(), [4096L, 8192L])
  assert_true(
    requirements
    .requirements()
    .any(r => r.operation() == KdaGatedRmsNorm(64, 128, 0.00001)),
  )
  assert_true(
    requirements
    .requirements()
    .any(r => r.operation() == KdaOutputProjection(8192, 4096)),
  )
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut gate_count = 0
  let mut norm_count = 0
  let mut output_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    match operation.kind() {
      KdaOutputGateProjection => gate_count += 1
      KdaGatedRmsNorm => {
        norm_count += 1
        assert_eq(
          operation.execution().activation_compute(),
          @numeric_contract.ComputeDType::f32(),
        )
      }
      KdaOutputProjection => output_count += 1
      _ => ()
    }
  }
  assert_eq(gate_count, 34)
  assert_eq(norm_count, 34)
  assert_eq(output_count, 34)
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("intermediate[intermediate_base + out]"))
  assert_false(source.contains("rsqrt"))
  assert_false(source.contains("sigmoid"))
  inspect(
    candidate.source_digest().as_hex(),
    content="c5fd0742fcbe21ec904080ea70792f974c3ce27d703396c016edcb7d1c2c2655",
  )
  inspect(
    candidate.recipe_digest().as_hex(),
    content="3ed74fe90a12ecee34cacd44f2d66320cf93b1f78f4735066b1829d4c64f4f91",
  )
  assert_false(candidate.manifest_bindable())
}
