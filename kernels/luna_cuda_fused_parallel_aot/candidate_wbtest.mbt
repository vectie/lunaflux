///|
test "fused QKV RoPE KV lowering is deterministic bounded and fallback distinct" {
  let fixture = fused_test_fixture()
  let first = fused_test_qkv_candidate(fixture)
  let second = fused_test_qkv_candidate(fixture)
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(first.launch_dimensions().block_x(), 128)
  assert_eq(
    first.launch_dimensions().grid_x(),
    fixture.profile.max_query_tokens(),
  )
  assert_eq(first.launch_dimensions().shared_memory_bytes(), 0)
  assert_eq(first.head_dimension(), 32)
  assert_eq(first.tokens_per_page(), 16)
  assert_false(first.manifest_bindable())
  assert_true(first.dispatch_canary() != 0)
  assert_true(
    first.source_digest().as_hex() != fixture.qkv.source_digest().as_hex(),
  )
  assert_true(
    first.source_digest().as_hex() != fixture.rope.source_digest().as_hex(),
  )
  assert_true(
    first.source_digest().as_hex() != fixture.kv_write.source_digest().as_hex(),
  )
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("blockDim.x != LF_BLOCK_THREADS"))
  assert_true(source.contains("atomicAdd(dispatch_canary"))
  assert_true(source.contains("page_table_page_indices"))
  assert_true(source.contains("decode != rows - prefill"))
  assert_true(source.contains("rows > 4"))
  assert_true(source.contains("tokens > 8"))
  assert_true(source.contains("page_cells > 16"))
  assert_true(source.contains("query_row_offsets[rows] != tokens"))
  assert_true(source.contains("page_table_offsets[rows] != page_cells"))
  assert_true(
    source.contains("query_row_offsets[r] >= query_row_offsets[r + 1]"),
  )
  assert_true(
    source.contains("page_table_offsets[r] >= page_table_offsets[r + 1]"),
  )
  assert_true(source.contains("position >= sequence_lengths[row]"))
  assert_true(source.contains("logical_page >= page_end - page_begin"))
  assert_true(source.contains("physical_page >= LF_TOTAL_PAGE_COUNT"))
  assert_true(
    source.contains(
      "projected = __bfloat162float(__float2bfloat16_rn(projected))",
    ),
  )
  assert_true(source.contains("weight[weight_column * LF_INPUT_WIDTH + inner]"))
  assert_true(
    source.contains(
      "paired_weight[paired_weight_column * LF_INPUT_WIDTH + inner]",
    ),
  )
  assert_true(source.contains("const float product = __fmul_rn("))
  assert_true(source.contains("projected = __fadd_rn(projected, product)"))
  assert_true(source.contains("const float paired_product = __fmul_rn("))
  assert_true(source.contains("paired = __fadd_rn(paired, paired_product)"))
  assert_false(source.contains("weight[inner *"))
  assert_true(
    @utf8.decode(fixture.qkv.source_bytes()).contains(
      "weight_base = (unsigned long long)selected_column * 128ULL",
    ),
  )
  assert_true(source.contains("powf(LF_ROPE_THETA"))
  assert_true(source.contains("#define LF_ROPE_THETA 10000.0f\n"))
  assert_false(source.contains("#define LF_ROPE_THETA 10000f\n"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(
    recipe.contains("family=qkv-positioned-rope-paged-kvwrite-block128\n"),
  )
  assert_true(recipe.contains("qkv_fallback_source_sha256="))
  assert_true(recipe.contains("rope_fallback_recipe_sha256="))
  assert_true(recipe.contains("kv_write_fallback_source_sha256="))
  assert_true(recipe.contains("aot_only=true\nruntime_jit=false\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
  assert_true(
    recipe.contains(
      "numerics=fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1\n",
    ),
  )
}

///|
test "fused residual RMSNorm lowering is deterministic and tree reduced" {
  let fixture = fused_test_fixture()
  let first = fused_test_residual_norm_candidate(fixture)
  let second = fused_test_residual_norm_candidate(fixture)
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(first.width(), 128)
  assert_eq(first.launch_dimensions().block_x(), 128)
  assert_eq(first.launch_dimensions().shared_memory_bytes(), 512)
  assert_false(first.manifest_bindable())
  assert_true(first.dispatch_canary() != 0)
  assert_true(
    first.dispatch_canary().to_int64() *
    fixture.profile.max_query_tokens().to_int64() <=
    4294967295L,
  )
  assert_true(
    first.dispatch_canary() !=
    fused_test_qkv_candidate(fixture).dispatch_canary(),
  )
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("#define LF_EPSILON "))
  assert_false(source.contains("#define LF_EPSILON 0f\n"))
  assert_true(source.contains("__shared__ float partial[LF_BLOCK_THREADS]"))
  assert_true(source.contains("stride >>= 1"))
  assert_true(source.contains("rsqrtf"))
  assert_true(source.contains("decode != rows - prefill"))
  assert_true(source.contains("tokens < rows"))
  assert_true(source.contains("token >= tokens"))
  assert_true(
    source.contains("const float rounded = __bfloat162float(encoded)"),
  )
  assert_true(source.contains("atomicAdd(dispatch_canary"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(recipe.contains("family=residual-rmsnorm-block128\n"))
  assert_true(recipe.contains("residual_fallback_source_sha256="))
  assert_true(recipe.contains("norm_fallback_recipe_sha256="))
  assert_true(
    recipe.contains(
      "numerics=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1\n",
    ),
  )
  assert_true(recipe.contains("performance_claim=none") == false)
}

///|
test "fused candidates reject target compiler reference and ordering substitution" {
  let fixture = fused_test_fixture()
  let unsupported_target = @catalog.DeviceTarget::new(
    compute_major=7,
    compute_minor=5,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try
    lower_fused_qkv_rope_paged_kv_write_candidate(
      fixture.plan,
      fixture.layout,
      fixture.qkv,
      fixture.rope,
      fixture.kv_write,
      unsupported_target,
      fixture.compiler,
      fused_test_limits(),
    )
  catch {
    FusedParallelUnsupported(DeviceTarget) => ()
    _ => fail("unexpected target rejection")
  } noraise {
    _ => fail("unsupported target was lowered")
  }
  try
    lower_fused_residual_rmsnorm_candidate(
      fixture.plan,
      fixture.residual,
      fixture.norm,
      fixture.target,
      fused_test_compiler(reassociate=true),
      fused_test_limits(),
    )
  catch {
    FusedParallelUnsupported(CompilerPolicy) => ()
    _ => fail("unexpected compiler rejection")
  } noraise {
    _ => fail("reassociating compiler was admitted")
  }
  try
    lower_fused_residual_rmsnorm_candidate(
      fixture.plan,
      fixture.norm,
      fixture.residual,
      fixture.target,
      fixture.compiler,
      fused_test_limits(),
    )
  catch {
    FusedParallelMismatch(ReferenceFallback) => ()
    _ => fail("unexpected fallback-order rejection")
  } noraise {
    _ => fail("swapped residual and norm fallbacks were admitted")
  }
  let alternate_layout = @device_layout.plan(
    fixture.plan.identity(),
    fixture.plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=32,
      total_page_count=64,
      max_arena_bytes=134217728L,
    ),
  )
  try
    lower_fused_qkv_rope_paged_kv_write_candidate(
      fixture.plan,
      alternate_layout,
      fixture.qkv,
      fixture.rope,
      fixture.kv_write,
      fixture.target,
      fixture.compiler,
      fused_test_limits(),
    )
  catch {
    FusedParallelMismatch(OperandSize) => ()
    _ => fail("unexpected KV-layout replay rejection")
  } noraise {
    _ => fail("KV fallback from another layout was admitted")
  }
}

///|
test "production residual RMSNorm candidate is deterministic and canary free" {
  let fixture = fused_test_fixture()
  let qualification = fused_test_residual_norm_candidate(fixture)
  let first = lower_fused_residual_rmsnorm_production_candidate(
    qualification,
    fused_test_limits(),
  )
  let second = lower_fused_residual_rmsnorm_production_candidate(
    qualification,
    fused_test_limits(),
  )
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(
    first.qualification_candidate_digest(),
    qualification.candidate_digest(),
  )
  let source = @utf8.decode(first.source_bytes())
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_false(source.contains("dispatch_canary"))
  assert_false(source.contains("atomicAdd"))
  assert_true(source.contains("__nv_bfloat16* norm_output)"))
  assert_true(recipe.contains("execution_policy=production-fast-path\n"))
  assert_true(recipe.contains("diagnostic_canary=absent\n"))
  assert_true(
    recipe.contains(
      "abi=step_counts,residual,branch,norm_weight,residual_output,norm_output\n",
    ),
  )
  assert_false(recipe.contains("dispatch_canary="))
  assert_false(first.manifest_bindable())
}

///|
test "production fused QKV RoPE KV candidate is deterministic and canary free" {
  let fixture = fused_test_fixture()
  let qualification = fused_test_qkv_candidate(fixture)
  let first = lower_fused_qkv_rope_paged_kv_write_production_candidate(
    qualification,
    fused_test_limits(),
  )
  let second = lower_fused_qkv_rope_paged_kv_write_production_candidate(
    qualification,
    fused_test_limits(),
  )
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.recipe_bytes(), second.recipe_bytes())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(
    first.qualification_candidate_digest(),
    qualification.candidate_digest(),
  )
  let source = @utf8.decode(first.source_bytes())
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_false(source.contains("dispatch_canary"))
  assert_false(source.contains("atomicAdd"))
  assert_true(source.contains("__nv_bfloat16* value_cache)"))
  assert_true(recipe.contains("execution_policy=production-fast-path\n"))
  assert_true(recipe.contains("diagnostic_canary=absent\n"))
  assert_true(
    recipe.contains(
      "abi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
      "v_weight,rotated_qkv,key_cache,value_cache\n",
    ),
  )
  assert_false(recipe.contains("dispatch_canary="))
  assert_false(first.manifest_bindable())
}

///|
test "candidate promotion remains authority free and evidence bound" {
  let fixture = fused_test_fixture()
  let candidate = fused_test_qkv_candidate(fixture)
  let trials : Array[FusedParallelPromotionTrial] = []
  for shape in 0..<2 {
    for trial in 0..<3 {
      trials.push(
        FusedParallelPromotionTrial::new(
          shape_ordinal=shape,
          trial_ordinal=trial,
          baseline_duration_ns=1000L,
          candidate_duration_ns=900L,
          maximum_absolute_error_ppb=1000L,
          maximum_relative_error_ppb=1000L,
          correctness=FusedParallelDigest::from_sha256("1".repeat(64)),
          dispatch_canary=FusedParallelDigest::from_sha256("2".repeat(64)),
        ),
      )
    }
  }
  let evidence = admit_fused_qkv_rope_kv_promotion_evidence(
    candidate,
    FusedParallelDigest::from_sha256("3".repeat(64)),
    trials,
    FusedParallelPromotionLimits::new(
      maximum_shapes=8,
      trials_per_shape=3,
      maximum_canonical_bytes=1048576,
    ),
  )
  assert_eq(evidence.trial_count(), 6)
  assert_false(evidence.manifest_bindable())
  assert_true(
    @utf8.decode(evidence.canonical_bytes()).contains(
      "measurement_authority=caller-supplied-raw-claim\n",
    ),
  )
  let binding = bind_fused_qkv_rope_kv_promotion_evidence(
    candidate,
    evidence,
    fused_test_limits(),
  )
  assert_eq(binding.candidate_digest(), candidate.candidate_digest())
  assert_eq(binding.source_digest(), candidate.source_digest())
  assert_eq(binding.recipe_digest(), candidate.recipe_digest())
  assert_false(binding.manifest_bindable())
  let record = @utf8.decode(binding.record_bytes())
  assert_true(
    record.contains("evidence_schema=lunaflux-fused-parallel-raw-evidence.v1\n"),
  )
  assert_true(record.contains("physical_performance_authority=external\n"))
  assert_true(record.contains("performance_claim=none\n"))
  assert_true(record.contains("manifest_bindable=false\n"))
  try
    bind_fused_qkv_rope_kv_promotion_evidence(
      candidate,
      evidence,
      FusedParallelAotLimits::new(
        maximum_source_bytes=1048576,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=1,
      ),
    )
  catch {
    FusedParallelLimitExceeded(Evidence) => ()
    _ => fail("unexpected promotion-bound rejection")
  } noraise {
    _ => fail("oversized promotion record was admitted")
  }

  let slow = trials.copy()
  slow[0] = { ..slow[0], candidate_duration_ns: 2000L }
  try
    admit_fused_qkv_rope_kv_promotion_evidence(
      candidate,
      FusedParallelDigest::from_sha256("3".repeat(64)),
      slow,
      FusedParallelPromotionLimits::new(
        maximum_shapes=8,
        trials_per_shape=3,
        maximum_canonical_bytes=1048576,
      ),
    )
  catch {
    FusedParallelMismatch(MicrobenchmarkWin) => ()
    _ => fail("unexpected slow-evidence rejection")
  } noraise {
    _ => fail("slow candidate evidence was admitted")
  }

  let inaccurate = trials.copy()
  inaccurate[0] = {
    ..inaccurate[0],
    maximum_absolute_error_ppb: candidate
    .numerical_policy()
    .maximum_absolute_error_ppb() +
    1L,
  }
  try
    admit_fused_qkv_rope_kv_promotion_evidence(
      candidate,
      FusedParallelDigest::from_sha256("3".repeat(64)),
      inaccurate,
      FusedParallelPromotionLimits::new(
        maximum_shapes=8,
        trials_per_shape=3,
        maximum_canonical_bytes=1048576,
      ),
    )
  catch {
    FusedParallelMismatch(Correctness) => ()
    _ => fail("unexpected inaccurate-evidence rejection")
  } noraise {
    _ => fail("out-of-tolerance evidence was admitted")
  }

  let reordered = trials.copy()
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try
    admit_fused_qkv_rope_kv_promotion_evidence(
      candidate,
      FusedParallelDigest::from_sha256("3".repeat(64)),
      reordered,
      FusedParallelPromotionLimits::new(
        maximum_shapes=8,
        trials_per_shape=3,
        maximum_canonical_bytes=1048576,
      ),
    )
  catch {
    FusedParallelMismatch(ObservationOrder) => ()
    _ => fail("unexpected reordered-evidence rejection")
  } noraise {
    _ => fail("reordered trial evidence was admitted")
  }

  try
    admit_fused_qkv_rope_kv_promotion_evidence(
      candidate,
      candidate.candidate_digest(),
      trials,
      FusedParallelPromotionLimits::new(
        maximum_shapes=8,
        trials_per_shape=3,
        maximum_canonical_bytes=1048576,
      ),
    )
  catch {
    FusedParallelMismatch(Evidence) => ()
    _ => fail("unexpected resource-audit replay rejection")
  } noraise {
    _ => fail("candidate digest was replayed as a resource audit")
  }

  try
    bind_fused_residual_rmsnorm_promotion_evidence(
      fused_test_residual_norm_candidate(fixture),
      evidence,
      fused_test_limits(),
    )
  catch {
    FusedParallelMismatch(Evidence) => ()
    _ => fail("unexpected cross-candidate evidence rejection")
  } noraise {
    _ => fail("QKV evidence was rebound to residual RMSNorm")
  }
}
