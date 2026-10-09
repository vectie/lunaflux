///|
test "official Flash KDA recurrent transition joins exact mixed dtype ABI" {
  let metadata = flash_metadata()
  let requirements = @glm53_kernel_requirements.build_flash_kernel_requirements(
    metadata,
  )
  let candidate = @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
    requirements,
    requirement_ordinal=find_kda_recurrent_ordinal(requirements),
    maximum_tokens=32,
    maximum_sequences=8,
    target=target(),
    compiler=compiler(),
    limits=limits(),
  )
  assert_eq(candidate.operation(), KdaRecurrentUpdate(64, 128, -5.0))
  assert_eq(candidate.capability(), KdaRecurrentUpdateAot)
  assert_eq(candidate.head_count(), 64)
  assert_eq(candidate.head_dimension(), 128)
  assert_eq(candidate.normalization_epsilon(), 0.000001)
  assert_eq(candidate.dimensions().grid_x(), 1)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.operands().length(), 10)
  let expected_roles : Array[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole,
  ] = [
    StepCountsInput,
    KdaSequenceOffsetsInput,
    KdaRecurrentQueryInput,
    KdaRecurrentKeyInput,
    KdaRecurrentValueInput,
    KdaLogDecayInput,
    KdaInputGateInput,
    KdaRecurrentInitialStateInput,
    KdaRecurrentOutput,
    KdaRecurrentFinalStateOutput,
  ]
  for index, role in expected_roles {
    assert_eq(candidate.operands()[index].role(), role)
  }
  assert_eq(candidate.operands()[1].byte_count(), 36L)
  assert_eq(candidate.operands()[2].byte_count(), 524288L)
  assert_eq(candidate.operands()[5].byte_count(), 1048576L)
  assert_eq(candidate.operands()[6].byte_count(), 4096L)
  assert_eq(candidate.operands()[7].byte_count(), 33554432L)
  assert_eq(candidate.operands()[9].byte_count(), 33554432L)
  let numeric = @glm53_numeric_plan.build_flash_numeric_plan(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let mut recurrent_count = 0
  for ordinal in 0..<numeric.operation_count() {
    guard numeric.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == KdaRecurrentUpdate {
      recurrent_count += 1
      assert_eq(
        operation.execution().activation_input(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
      assert_eq(
        operation.execution().activation_compute(),
        @numeric_contract.ComputeDType::f32(),
      )
      assert_eq(
        operation.execution().output(),
        @numeric_contract.ComputeDType::bfloat16(),
      )
    }
  }
  assert_eq(recurrent_count, 34)
  let manifest = try! @glm53_weights.build_flash_tensor_manifest(
    metadata,
    BFloat16WithFloat32Routing,
  )
  let decay_rate = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.A_log",
  )
  let decay_bias = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.dt_bias",
  )
  let beta_projection = find_named_tensor(
    manifest, "model.language_model.layers.0.self_attn.b_proj.weight",
  )
  assert_eq(decay_rate.dtype(), Float32)
  assert_eq(decay_rate.shape(), [64L])
  assert_eq(decay_bias.dtype(), Float32)
  assert_eq(decay_bias.shape(), [8192L])
  assert_eq(beta_projection.dtype(), BFloat16)
  assert_eq(beta_projection.shape(), [64L, 4096L])
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("expf(log_decay[qbase + k])"))
  assert_true(source.contains("__bfloat162float(beta"))
  assert_true(source.contains("__float2bfloat16_rn(projected)"))
  let recipe = @utf8.decode(candidate.recipe_bytes())
  assert_true(recipe.contains("qualification=serial-correctness-source-only"))
  assert_true(
    recipe.contains("state=sequence-head-key-value-f32-explicit-in-out"),
  )
  ignore(try! @glm_hybrid_candidate_evidence.from_kda_recurrent(candidate))
  assert_false(candidate.manifest_bindable())
}

///|
test "Full has no recurrent KDA and hostile ordinal target compiler fail closed" {
  let full = @glm53_kernel_requirements.build_full_kernel_requirements(
    full_metadata(),
  )
  assert_false(
    full.requirements().any(r => r.operation() is KdaRecurrentUpdate(_, _, _)),
  )
  let exact = @glm53_kernel_requirements.build_flash_kernel_requirements(
    flash_metadata(),
  )
  let ordinal = find_kda_recurrent_ordinal(exact)
  try
    @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      maximum_sequences=8,
      target=target(),
      compiler=compiler(),
      limits=limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("neighboring operation admitted")
  }
  try
    @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      maximum_sequences=8,
      target=try! @catalog.DeviceTarget::new(
        compute_major=9,
        compute_minor=0,
        supports_bf16=false,
        supports_cublas_lt=true,
      ),
      compiler=compiler(),
      limits=limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  let hostile = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=compiler().identity(),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=true,
    max_registers=128,
  )
  try
    @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
      exact,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      maximum_sequences=8,
      target=target(),
      compiler=hostile,
      limits=limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
}
