///|
struct FusedProbeCandidateContext {
  qualification : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet
  attention : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering
}

///|
fn fused_probe_candidate_context(
  toolchain_sha256 : String,
) -> FusedProbeCandidateContext raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=256,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=4,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=1,
    max_query_tokens=8,
    max_page_table_entries=4,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=toolchain_sha256,
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let reference = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
  let mut attention = None
  for candidate in reference.candidates() {
    if candidate is PagedAttention(value) {
      attention = Some(value)
      break
    }
  }
  guard attention is Some(attention) else {
    raise FusedCudaProbeRejected("reference attention candidate is absent")
  }
  {
    qualification: @luna_fused_candidate_export.prepare_qualification_set(
      plan, layout, reference,
    ),
    attention,
  }
}

///|
fn fused_probe_candidate_set(
  toolchain_sha256 : String,
) -> @luna_fused_candidate_export.LunaFusedCandidateQualificationSet raise {
  fused_probe_candidate_context(toolchain_sha256).qualification
}
