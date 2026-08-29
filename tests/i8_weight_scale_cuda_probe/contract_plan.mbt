///|
fn probe_model_plan() -> @plan.ModelPlan raise {
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "621f982b992433d13a900cca317473122b8311c111f936cd6a3b51911621e5be",
    ),
  )
  @llama.build_paged_symmetric_i8_weight_only_v1(
    metadata,
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
}

///|
fn probe_kv_layout(
  plan : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=2,
      max_arena_bytes=4096L,
    ),
  )
}

///|
fn probe_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=1,
    max_query_tokens=3,
    max_page_table_entries=2,
  )
}
