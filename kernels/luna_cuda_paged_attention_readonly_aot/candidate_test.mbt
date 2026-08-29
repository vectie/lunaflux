///|
test "read-only paged attention binds exact fallback layout profile target and ABI" {
  let first = readonly_fixture_candidate()
  let second = readonly_fixture_candidate()
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(first.operation_id(), @plan.OperationId::new(2))
  assert_eq(first.target(), readonly_fixture_target())
  assert_eq(first.profile(), readonly_fixture_profile())
  assert_eq(
    first.function_symbol(),
    "lunaflux_paged_attention_bf16_readonly_v1_ep_103",
  )
  assert_eq(first.kernel_abi(), RotatedQueryPagedKvReadOnlyRawPointerV1)
  assert_false(first.manifest_bindable())
  assert_false(first.promotion_authorized())
  let recipe = @utf8.decode(first.recipe_bytes())
  for
    required in [
      "model_operation_kind=causal_attention",
      "selection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1",
      "fallback_source_sha256=" +
      first.reference_fallback().source_digest().as_hex(),
      "fallback_recipe_sha256=" +
      first.reference_fallback().recipe_digest().as_hex(),
      "target=sm_120",
      "profile=2,3,3",
      "input_row_width=8",
      "tokens_per_page=2",
      "kv_layout=layer_major_split_key_value_v1",
      "kv_layout_stable_version=1",
      "page_stride_bytes=256",
      "rotated_query_only=true",
      "kv_cache_access=read_only",
      "kv_cache_mutation=none",
      "dispatch_canary_per_token=65624",
      "dispatch_canary_cell_count=3",
      "manifest_bindable=false",
      "promotion_authority=none",
    ] {
    assert_true(recipe.contains(required))
  }
}

///|
test "read-only CUDA source has separate symbol/raw ABI and no KV write channel" {
  let candidate = readonly_fixture_candidate()
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("const __nv_bfloat16 *rotated_qkv_rows"))
  assert_true(source.contains("const __nv_bfloat16 *key_cache"))
  assert_true(source.contains("const __nv_bfloat16 *value_cache"))
  assert_true(source.contains("output[output_offset + d] ="))
  assert_true(source.contains("query_offset = token * 8 + query_head * 2"))
  assert_true(source.contains("dispatch_canary[token] = 65624U"))
  assert_false(source.contains("key_cache[cache_index] ="))
  assert_false(source.contains("value_cache[cache_index] ="))
  assert_false(source.contains("*key_cache ="))
  assert_false(source.contains("*value_cache ="))
  assert_false(source.contains("const __nv_bfloat16 *qkv,"))
  assert_false(
    candidate.function_symbol() ==
    candidate.reference_fallback().function_symbol(),
  )
}

///|
test "read-only lowering rejects fallback symbol substitution and bounded output" {
  try readonly_fixture_candidate(entry=3) catch {
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyMismatch(
      Symbol
    ) => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("fallback symbol identity was reused")
  }
  let reference = readonly_fixture_reference()
  try
    @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
      reference,
      @catalog.AotKernelEntryPointId::from_stable_int(103),
      @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
        maximum_source_bytes=32,
        maximum_recipe_bytes=8192,
        maximum_canonical_bytes=8192,
      ),
    )
  catch {
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyLimitExceeded(
      Source
    ) => ()
    _ => fail("unexpected source limit rejection")
  } noraise {
    _ => fail("unbounded source was accepted")
  }
}

///|
test "read-only lowering rejects compact or padded full-QKV row substitution and canary overflow" {
  let operands = readonly_fixture_operands()
  let padded = Array::from_iter(operands.iter())
  padded[6] = readonly_fixture_operand(
    ActivationInput(@plan.ActivationRef::new(1)),
    64L,
    16L,
  )
  let reference = @luna_cuda_paged_attention_aot.lower_paged_attention_cuda_aot_candidate(
    readonly_fixture_operation(),
    readonly_fixture_profile(),
    readonly_fixture_layout(),
    @catalog.AotKernelEntryPointId::from_stable_int(3),
    readonly_fixture_target(),
    readonly_fixture_compiler(),
    padded,
    @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
      max_source_bytes=65536,
      max_recipe_bytes=8192,
    ),
  )
  try
    @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_cuda_aot_candidate(
      reference,
      @catalog.AotKernelEntryPointId::from_stable_int(103),
      @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
        maximum_source_bytes=65536,
        maximum_recipe_bytes=8192,
        maximum_canonical_bytes=8192,
      ),
    )
  catch {
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyMismatch(
      KernelAbi
    ) => ()
    _ => fail("unexpected full-QKV row-width rejection")
  } noraise {
    _ => fail("padded full-QKV row substitution was accepted")
  }
  try readonly_fixture_candidate(entry=1500000000) catch {
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyInvalid(
      KernelAbi
    ) => ()
    _ => fail("unexpected dispatch canary overflow rejection")
  } noraise {
    _ => fail("dispatch canary multiplication overflow was accepted")
  }
}

///|
test "production readonly candidate is exact deterministic and canary free" {
  let qualification = readonly_fixture_candidate()
  let limits = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyAotLimits::new(
    maximum_source_bytes=65536,
    maximum_recipe_bytes=8192,
    maximum_canonical_bytes=8192,
  )
  let first = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_production_cuda_aot_candidate(
    qualification, limits,
  )
  let second = @luna_cuda_paged_attention_readonly_aot.lower_paged_attention_readonly_production_cuda_aot_candidate(
    qualification, limits,
  )
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(
    first.qualification_candidate_digest(),
    qualification.candidate_digest(),
  )
  assert_eq(
    first.kernel_abi(),
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2,
  )
  let source = @utf8.decode(first.source_bytes())
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_false(source.contains("dispatch_canary"))
  assert_true(source.contains("const __nv_bfloat16 *value_cache)"))
  assert_true(source.contains("output[output_offset + d] ="))
  assert_true(recipe.contains("execution_policy=production-fast-path\n"))
  assert_true(recipe.contains("diagnostic_canary=absent\n"))
  assert_false(recipe.contains("dispatch_canary_per_token="))
  assert_false(first.manifest_bindable())
}
