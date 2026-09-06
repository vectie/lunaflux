///|
test "artifact producer consumes an offline projection storage schedule" {
  let model = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=1,
    attention_head_count=8,
    kv_head_count=4,
    head_dimension=128,
    context_length=8,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let plan = @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("9".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(8),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=8,
      max_arena_bytes=1048576L,
    ),
  )
  let profile = tiny_candidate_profile(rows=8)
  let target = tiny_candidate_target()
  let head = plan.operations()[plan.operations().length() - 1]
  assert_eq(head.kind(), LanguageModelHead)
  let operands = derive_candidate_operands(plan, head, profile, layout)
  let dispatch = @luna_cuda_projection_aot.prepare_projection_cuda_runtime_dispatch(
    head,
    profile,
    target,
    operands,
    [],
  )
  let original = dispatch.dispatch_plan(8)
  let resident = original.prefill().with_selected_row_residency(stable_id=3008)
  // Synthetic timing tests plumbing, not a measured or installed serving policy.
  let record = @luna_projection_strategy.ProjectionAutotuneRecord::new(
    key=@luna_projection_strategy.ProjectionAutotuneKey::new(
      backend=@luna_projection_strategy.ProjectionBackendId::new("cuda-aot-v1"),
      target=@luna_projection_strategy.ProjectionTargetId::new("sm_120"),
      numeric_type=ProjectionBFloat16,
      shape=original.shape(),
      strategy=resident,
    ),
    median_latency_ns=1L,
    sample_count=3,
  )
  let baseline = prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    producer_fixture_compiler(),
    tiny_candidate_limits(),
  )
  let selected = prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    producer_fixture_compiler(),
    tiny_candidate_limits(),
    projection_records=[record],
  )
  let scope = @luna_projection_tuning.ProjectionTuningScope::new(
    backend=record.key().backend(),
    architecture=record.key().target(),
    device_identity="synthetic-test-device",
    toolchain_sha256="a".repeat(64),
  )
  let stored = @luna_projection_tuning.ProjectionTuningSnapshot::new(scope, [
    record,
  ])
  let restored = @luna_projection_tuning.parse_projection_tuning_snapshot(
    stored.canonical_bytes(),
  ).records_for(scope)
  let rebound = prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    producer_fixture_compiler(),
    tiny_candidate_limits(),
    projection_records=restored[:],
  )
  for index in 0..<plan.operations().length() {
    assert_eq(
      selected.candidate_source_bytes(index),
      rebound.candidate_source_bytes(index),
    )
    assert_eq(
      selected.candidate_recipe_bytes(index),
      rebound.candidate_recipe_bytes(index),
    )
  }
  assert_eq(baseline.model_identity(), selected.model_identity())
  for index in 0..<head.id().as_int() {
    assert_eq(
      baseline.candidate_source_bytes(index),
      selected.candidate_source_bytes(index),
    )
  }
  guard selected.candidates()[head.id().as_int()] is Projection(candidate) else {
    fail("head must remain a projection")
  }
  assert_eq(candidate.launch_dimensions().block_x(), 256)
  assert_eq(candidate.launch_dimensions().shared_memory_bytes(), 40960)
  assert_true(
    candidate.source_bytes() !=
    baseline.candidate_source_bytes(head.id().as_int()).unwrap(),
  )
  assert_true(
    candidate.recipe_bytes() !=
    baseline.candidate_recipe_bytes(head.id().as_int()).unwrap(),
  )
}
