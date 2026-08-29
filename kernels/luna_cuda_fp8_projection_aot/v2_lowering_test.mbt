///|
fn fp8_aot_plan_v2() -> @plan.ModelPlan {
  let metadata = try! @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=1,
      attention_head_count=4,
      kv_head_count=2,
      context_length=32,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1",
    ),
  )
  try! @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
}

///|
fn fp8_aot_lower_v2(
  operation : Int,
  profile? : @launch_contract.PagedKernelProfile = fp8_aot_profile(),
) -> Fp8ProjectionCudaLoweringV2 {
  let plan = fp8_aot_plan_v2()
  let id = try! @plan.OperationId::new(operation)
  let operands = try! derive_fp8_projection_operands_v2(
    plan,
    operation=id,
    profile,
  )
  try! lower_fp8_projection_cuda_aot_candidate_v2(
    plan,
    operation=id,
    profile,
    fp8_aot_target(),
    fp8_aot_compiler(),
    operands,
    try! Fp8ProjectionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
}

///|
test "FP8 v2 gated source owns two independent scale cells without grid sync" {
  let first = fp8_aot_lower_v2(8)
  let second = fp8_aot_lower_v2(8)
  assert_eq(first, second)
  assert_eq(first.family(), GatedMlp)
  assert_eq(first.workspace_bytes(), 8L)
  assert_eq(
    first.activation_scale_policy(),
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
  )
  assert_eq(
    first.function_symbol(),
    "lunaflux_luna_gated_mlp_fp8_e4m3_w8a8_release_v2",
  )
  let source = try! @utf8.decode(first.source_bytes())
  assert_true(source.contains("CUDA_VERSION < 12000"))
  assert_true(source.contains("__CUDA_ARCH__ != 890"))
  assert_true(source.contains("__CUDA_ARCH__ != 900"))
  assert_true(
    source.contains(
      "__nv_cvt_float_to_fp8(canonical, __NV_SATFINITE, __NV_E4M3)",
    ),
  )
  assert_true(source.contains("bounded == 0.0f ? 0.0f : bounded"))
  assert_eq(count_text(source, "workspace[0] = external_scale"), 1)
  assert_eq(count_text(source, "workspace[1] = post_silu_scale"), 1)
  assert_true(source.contains("workspace[0] = lf_v2_failure()"))
  assert_true(source.contains("workspace[1] = lf_v2_failure()"))
  assert_true(source.contains("__uint_as_float(0x7fc00000U)"))
  assert_true(source.contains("if (index != 0ULL) return"))
  assert_true(
    source.contains("const float silu = __fdiv_rn(gate, denominator);"),
  )
  assert_false(source.contains("__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate)))"))
  assert_true(source.contains("scale_row < token_count"))
  assert_true(source.contains("scale_column < 16"))
  assert_true(source.contains("post_silu_maximum"))
  assert_true(source.contains("if (!isfinite(z)) return"))
  assert_true(source.contains("if (!isfinite(accumulator)) return"))
  assert_true(
    source.contains("lf_v2_e4m3_round(__fdiv_rn(z, post_silu_scale))"),
  )
  assert_false(source.contains("__syncthreads"))
  assert_false(source.contains("cooperative_groups"))
  let operands = first.operands()
  assert_eq(operands.length(), 10)
  assert_eq(operands[9].role(), Workspace)
  assert_eq(operands[9].byte_count(), 8L)
  assert_eq(operands[9].alignment(), 4L)
  let recipe = try! @utf8.decode(first.recipe_bytes())
  assert_true(
    recipe.contains("canonical_domain=lunaflux.luna-cuda-fp8-projection-aot.v2"),
  )
  assert_true(
    recipe.contains("activation_scale_stage_1=post_silu_gate_up_product_v1"),
  )
  assert_true(recipe.contains("workspace_pointer_count=1"))
  assert_true(recipe.contains("workspace_bytes=8"))
  assert_true(
    recipe.contains(
      "workspace_publication=serial_index_zero_after_full_output_validation_v2",
    ),
  )
  assert_true(recipe.contains("nonfinite_failure_sentinel_bits=0x7fc00000"))
  assert_true(
    recipe.contains("gated_exponential=target_compiler_bound_expf_f32_v2"),
  )
}

///|
test "FP8 v2 simple source retains one-cell ABI and cannot replay as gated" {
  let simple = fp8_aot_lower_v2(2)
  assert_eq(simple.family(), QkvProjection)
  assert_eq(simple.workspace_bytes(), 4L)
  assert_eq(
    simple.function_symbol(),
    "lunaflux_luna_qkv_projection_fp8_e4m3_w8a8_release_v2",
  )
  let source = try! @utf8.decode(simple.source_bytes())
  assert_eq(count_text(source, "workspace[0] = external_scale"), 1)
  assert_false(source.contains("workspace[1]"))
  assert_true(simple.source_digest() != fp8_aot_lower(2).source_digest())
  let legacy = fp8_aot_plan()
  try
    derive_fp8_projection_operands_v2(
      legacy,
      operation=try! @plan.OperationId::new(8),
      fp8_aot_profile(),
    )
  catch {
    Unsupported(ActivationScalePolicy) => ()
    _ => fail("unexpected legacy gated-policy rejection")
  } noraise {
    _ => fail("legacy one-scale gated policy replayed as v2")
  }
}

///|
test "FP8 v2 rejects raw operand substitution and binds profile identity" {
  let plan = fp8_aot_plan_v2()
  let operation = try! @plan.OperationId::new(8)
  let profile = fp8_aot_profile()
  let admitted = try! derive_fp8_projection_operands_v2(
    plan,
    operation~,
    profile,
  )
  let substituted : Array[@launch_contract.AotOperand] = []
  for operand in admitted {
    substituted.push(operand)
  }
  let temporary = substituted[3]
  substituted[3] = substituted[4]
  substituted[4] = temporary
  try
    lower_fp8_projection_cuda_aot_candidate_v2(
      plan,
      operation~,
      profile,
      fp8_aot_target(),
      fp8_aot_compiler(),
      substituted,
      try! Fp8ProjectionAotLimits::new(
        max_source_bytes=65536,
        max_recipe_bytes=65536,
      ),
    )
  catch {
    Mismatch(OperandOrder) => ()
    _ => fail("unexpected raw-operand rejection")
  } noraise {
    _ => fail("substituted weight/scale operands were admitted")
  }
  let batch_one = fp8_aot_lower_v2(8, profile=fp8_aot_profile(rows=1, tokens=8))
  let batch_two = fp8_aot_lower_v2(8, profile=fp8_aot_profile(rows=2, tokens=8))
  assert_eq(batch_one.source_digest(), batch_two.source_digest())
  assert_true(batch_one.recipe_digest() != batch_two.recipe_digest())
}

///|
test "FP8 v2 admits exact sm120 without broadening v1" {
  let plan = fp8_aot_plan_v2()
  let operation = try! @plan.OperationId::new(2)
  let profile = fp8_aot_profile()
  let operands = try! derive_fp8_projection_operands_v2(
    plan,
    operation~,
    profile,
  )
  let lowered = try! lower_fp8_projection_cuda_aot_candidate_v2(
    plan,
    operation~,
    profile,
    fp8_aot_target(major=12, minor=0),
    fp8_aot_compiler(),
    operands,
    try! Fp8ProjectionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=65536,
    ),
  )
  let source = try! @utf8.decode(lowered.source_bytes())
  assert_true(source.contains("__CUDA_ARCH__ != 1200"))
  let v1_plan = fp8_aot_plan()
  let v1_operands = try! derive_fp8_projection_operands(
    v1_plan,
    operation~,
    profile,
  )
  try
    lower_fp8_projection_cuda_aot_candidate(
      v1_plan,
      operation~,
      profile,
      fp8_aot_target(major=12, minor=0),
      fp8_aot_compiler(),
      v1_operands,
      try! Fp8ProjectionAotLimits::new(
        max_source_bytes=65536,
        max_recipe_bytes=65536,
      ),
    )
  catch {
    Unsupported(DeviceTarget) => ()
    _ => fail("v1 sm120 rejection reported an unrelated error")
  } noraise {
    _ => fail("sm120 incorrectly entered the v1 FP8 lowerer")
  }
}
