///|
test "both fused candidates produce deterministic inert benchmark qualification" {
  assert_eq(
    fused_candidate_benchmark_shape_set_identity().as_hex(),
    "a75940a0836789fa780a994ac55ccbcd1cba4835d6c1101f53b28a3f847a0aa5",
  )
  let set = fused_benchmark_test_set()
  let qkv = set.qkv_candidate()
  let qkv_protocol = fused_benchmark_test_protocol(
    qkv.candidate_digest(),
    fused_candidate_benchmark_shape_set_identity(),
    fused_candidate_benchmark_warmup_iterations(),
    fused_candidate_benchmark_trials_per_shape(),
    fused_candidate_benchmark_repetitions_per_arm(),
  )
  let qkv_observations = fused_benchmark_test_observations(
    qkv.dispatch_canary(),
  )
  let qkv_first = qualify_fused_qkv_candidate_benchmark(
    qkv,
    qkv_protocol,
    qkv_observations,
    maximum_evidence_bytes=1048576,
  )
  let qkv_second = qualify_fused_qkv_candidate_benchmark(
    qkv,
    qkv_protocol,
    qkv_observations,
    maximum_evidence_bytes=1048576,
  )
  assert_eq(qkv_first.family(), QkvPositionedRopePagedKvWrite)
  assert_eq(qkv_first.candidate_digest(), qkv.candidate_digest())
  assert_eq(
    qkv_first.shape_set_identity(),
    fused_candidate_benchmark_shape_set_identity(),
  )
  assert_eq(qkv_first.observation_count(), 32)
  assert_eq(qkv_first.canonical_bytes(), qkv_second.canonical_bytes())
  assert_eq(qkv_first.digest(), qkv_second.digest())
  assert_false(qkv_first.manifest_bindable())
  assert_false(qkv_first.promotion_authority())
  let qkv_record = @utf8.decode(qkv_first.canonical_bytes())
  assert_true(qkv_record.contains("target=sm_120\n"))
  assert_true(qkv_record.contains("toolchain_sha256=" + "b".repeat(64) + "\n"))
  assert_true(qkv_record.contains("compiler_version=13.1.115\n"))
  assert_true(
    qkv_record.contains(
      "compiler_policy=opt3,fmad=0,reassociate=0,max_registers=128\n",
    ),
  )
  assert_true(
    qkv_record.contains(
      "numerical_policy=fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1\n",
    ),
  )
  assert_true(qkv_record.contains("fallback=qkv,operation="))
  assert_true(qkv_record.contains("fallback=positioned-rope,operation="))
  assert_true(qkv_record.contains("fallback=paged-kv-write,operation="))
  assert_true(
    qkv_record.contains(
      "summary=single-origin,baseline_count=8,min_ns=1000,p50_ns=1003,max_ns=1007,total_ns=8028\n",
    ),
  )
  assert_true(
    qkv_record.contains(
      "summary=single-origin,candidate_count=8,min_ns=900,p50_ns=903,max_ns=907,total_ns=7228\n",
    ),
  )
  assert_true(qkv_record.contains("manifest_bindable=false\n"))
  assert_true(qkv_record.contains("promotion_authority=false\n"))

  let residual = set.residual_norm_candidate()
  let residual_protocol = fused_benchmark_test_protocol(
    residual.candidate_digest(),
    fused_candidate_benchmark_shape_set_identity(),
    fused_candidate_benchmark_warmup_iterations(),
    fused_candidate_benchmark_trials_per_shape(),
    fused_candidate_benchmark_repetitions_per_arm(),
  )
  let residual_evidence = qualify_fused_residual_rmsnorm_candidate_benchmark(
    residual,
    residual_protocol,
    fused_benchmark_test_observations(residual.dispatch_canary()),
    maximum_evidence_bytes=1048576,
  )
  assert_eq(residual_evidence.family(), ResidualRmsNorm)
  assert_eq(residual_evidence.candidate_digest(), residual.candidate_digest())
  assert_eq(residual_evidence.observation_count(), 32)
  assert_false(residual_evidence.manifest_bindable())
  assert_false(residual_evidence.promotion_authority())
  let residual_record = @utf8.decode(residual_evidence.canonical_bytes())
  assert_true(
    residual_record.contains(
      "numerical_policy=fused-residual-rmsnorm-block128-f32-tree-tolerance-v1\n",
    ),
  )
  assert_true(residual_record.contains("fallback=residual,operation="))
  assert_true(residual_record.contains("fallback=rmsnorm,operation="))
  assert_true(
    residual_record.contains(
      "win_policy=every-pair-strict-and-each-shape-total-at-least-5-percent.v1\n",
    ),
  )
}
