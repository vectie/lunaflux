///|
test "official Flash hyper controls join checkpoint numeric and downstream boundaries" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let function = @glm_hybrid_cuda_aot.lower_hyper_connection_function_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_hyper_function_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  let sinkhorn = @glm_hybrid_cuda_aot.lower_hyper_connection_sinkhorn_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_hyper_sinkhorn_ordinal(requirements),
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(
    function.operation(),
    HyperConnectionFunction(4, 4096, 0.000001, 0.00001),
  )
  assert_eq(sinkhorn.operation(), HyperConnectionSinkhorn(4, 20, 0.000001))
  assert_eq(function.family(), FlashFunctionAndCollapse4x4096)
  assert_eq(sinkhorn.family(), FlashPositiveSinkhorn4x20)
  assert_eq(function.operands()[6], sinkhorn.operands()[1])
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  for site in ["hc_attn", "hc_ffn"] {
    let prefix = "model.language_model.layers.0." + site
    let function_weight = find_named_tensor(manifest, prefix + "_fn")
    let base = find_named_tensor(manifest, prefix + "_base")
    let scale = find_named_tensor(manifest, prefix + "_scale")
    assert_eq(function_weight.dtype(), BFloat16)
    assert_eq(function_weight.shape(), [24L, 16384L])
    assert_eq(base.dtype(), Float32)
    assert_eq(base.shape(), [24L])
    assert_eq(scale.dtype(), Float32)
    assert_eq(scale.shape(), [3L])
    assert_eq(function.operands()[2].byte_count(), 786432L)
    assert_eq(function.operands()[3].byte_count(), 96L)
    assert_eq(function.operands()[4].byte_count(), 12L)
  }
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut function_count = 0
  let mut control_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    match operation.kind() {
      HyperConnectionBf16Function => {
        function_count += 1
        assert_eq(
          operation.execution().activation_input(),
          @numeric_contract.ComputeDType::bfloat16(),
        )
        assert_eq(
          operation.execution().activation_compute(),
          @numeric_contract.ComputeDType::f32(),
        )
      }
      HyperConnectionF32Control => {
        control_count += 1
        assert_eq(
          operation.execution().activation_compute(),
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
  assert_eq(function_count, 90)
  assert_eq(control_count, 90)
  let mut mix_ordinal = -1
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is HyperConnectionMix(4) {
      mix_ordinal = ordinal
    }
  }
  assert_true(mix_ordinal >= 0)
  let mix = @glm_hybrid_cuda_aot.lower_hyper_connection_mix_cuda_aot_candidate(
    requirements,
    requirement_ordinal=mix_ordinal,
    maximum_tokens=32,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(sinkhorn.operands()[2].byte_count(), mix.operands()[2].byte_count())
  assert_eq(function.operands()[5].byte_count(), mix.operands()[1].byte_count())
  assert_eq(function.operands()[7].byte_count(), mix.operands()[4].byte_count())
  let function_source = @utf8.decode(function.source_bytes())
  let sinkhorn_source = @utf8.decode(sinkhorn.source_bytes())
  assert_true(function_source.contains("__nv_bfloat16 *function_weights"))
  assert_true(function_source.contains("float *initial_combination"))
  assert_true(sinkhorn_source.contains("iteration = 1; iteration < 20"))
  assert_false(function_source.contains("HyperConnectionMix"))
  ignore(try! @glm_hybrid_candidate_evidence.from_hyper_control(function))
  ignore(try! @glm_hybrid_candidate_evidence.from_hyper_control(sinkhorn))
  assert_false(function.manifest_bindable())
  assert_false(sinkhorn.manifest_bindable())
  inspect(
    function.source_digest().as_hex(),
    content="2ec78e6f9a29acf0a9342c5691902fe7c81b2918765f79d1da0e6e2960aa235e",
  )
  inspect(
    function.recipe_digest().as_hex(),
    content="10fe59185721258dc928d80496fed0c0aefe4bf5914472f1ee026d18b7b56ae6",
  )
  inspect(
    sinkhorn.source_digest().as_hex(),
    content="b32baddf10b4f401b370af603462d2dadb09c89b5acfd7d1de5e1131fd168df2",
  )
  inspect(
    sinkhorn.recipe_digest().as_hex(),
    content="8fe54f2348232edc1831f55eb9987d42209c22b5107ef3e30a0b5c747c401f7a",
  )
}

///|
test "official Full has no hyper-control production requirements" {
  let requirements = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is HyperConnectionFunction(_, _, _, _)),
  )
  assert_false(
    requirements
    .requirements()
    .any(r => r.operation() is HyperConnectionSinkhorn(_, _, _)),
  )
}
