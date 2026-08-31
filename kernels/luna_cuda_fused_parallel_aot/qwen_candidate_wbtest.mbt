///|
fn qwen_fused_test_candidate() -> FusedQwenQkvQkNormRopeKvWriteCudaCandidate raise {
  let model = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=1024,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=32,
    context_length=256,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let plan = @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("3".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=64,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(19),
    max_query_rows=4,
    max_query_tokens=8,
    max_page_table_entries=16,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let compiler = fused_test_compiler()
  let set = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
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
  let mut qkv : @luna_cuda_projection_aot.ProjectionCudaLowering? = None
  let mut qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering? = None
  let mut rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering? = None
  let mut attention : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering? = None
  for candidate in set.candidates() {
    match candidate {
      Projection(value) =>
        if value.operation_id().as_int() == 2 {
          qkv = Some(value)
        }
      Pointwise(value) => {
        if value.operation_id().as_int() == 3 {
          qk_norm = Some(value)
        }
        if value.operation_id().as_int() == 4 {
          rope = Some(value)
        }
      }
      PagedAttention(value) =>
        if value.operation_id().as_int() == 5 {
          attention = Some(value)
        }
    }
  }
  guard (qkv, qk_norm, rope, attention)
    is (Some(qkv), Some(qk_norm), Some(rope), Some(attention)) else {
    abort("Qwen fused fixture lacked reference fallback")
  }
  let kv_operands : Array[@launch_contract.AotOperand] = []
  for operand in attention.operands() {
    match operand.role() {
      ActivationOutput(_) => ()
      _ => kv_operands.push(operand)
    }
  }
  let kv_write = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
    plan.operations()[5],
    profile,
    layout,
    @catalog.AotKernelEntryPointId::from_stable_int(3901),
    target,
    compiler,
    kv_operands,
    @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  )
  lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
    plan,
    layout,
    qkv,
    qk_norm,
    rope,
    kv_write,
    target,
    compiler,
    fused_test_limits(),
  )
}

///|
test "Qwen fused CUDA lowering consumes the backend-neutral semantic plan" {
  let first = qwen_fused_test_candidate()
  let second = qwen_fused_test_candidate()
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_eq(first.semantic_plan().operation_indices(), (2, 3, 4, 5))
  assert_eq(first.launch_dimensions().block_x(), 128)
  assert_eq(first.launch_dimensions().grid_y(), 8)
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("q_norm_weight"))
  assert_true(source.contains("k_norm_weight"))
  assert_true(source.contains("rsqrtf(reduction[0]"))
  assert_true(source.contains("const int head = (int)blockIdx.y;"))
  assert_false(source.contains("for (int head = 0; head < all_heads; ++head)"))
  assert_false(source.contains("nvcuda::wmma"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(
    recipe.contains(
      "semantic_family=qkv-qknorm-positioned-rope-paged-kvwrite\n",
    ),
  )
  assert_true(
    recipe.contains("reference_fallback=four-distinct-correctness-kernels\n"),
  )
  assert_true(recipe.contains("grid=8,8,1\n"))
  assert_true(recipe.contains("manifest_bindable=false\n"))
}

///|
test "Qwen fused production companion removes the diagnostic canary" {
  let qualification = qwen_fused_test_candidate()
  let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    qualification,
    fused_test_limits(),
  )
  assert_eq(
    production.qualification_candidate_digest(),
    qualification.candidate_digest(),
  )
  assert_eq(production.semantic_plan().operation_indices(), (2, 3, 4, 5))
  let source = @utf8.decode(production.source_bytes())
  assert_false(source.contains("dispatch_canary"))
  assert_true(source.contains("q_norm_weight"))
  let recipe = @utf8.decode(production.recipe_bytes())
  assert_true(recipe.contains("diagnostic_canary=absent\n"))
  assert_true(
    recipe.contains(
      "release_binding=external-approval-and-production-physical-gate-required\n",
    ),
  )
  assert_false(production.manifest_bindable())
}
