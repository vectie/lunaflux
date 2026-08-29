///|
priv struct PagedKvWriteTestFixture {
  plan : @plan.ModelPlan
  layout : @device_layout.DeviceKvLayout
  profile : @launch_contract.PagedKernelProfile
  target : @catalog.DeviceTarget
  compiler : @luna_specialization.LunaCompilerPolicy
  attention : @plan.PlanOperation
  fallback : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering
  input : PositionedQkvActivation
  operands : Array[@launch_contract.AotOperand]
}

///|
fn paged_kv_write_test_limits() -> PagedKvWriteAotLimits raise {
  PagedKvWriteAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
    max_module_bytes=1048576L,
    max_receipt_bytes=65536,
    max_binding_bytes=65536,
  )
}

///|
fn paged_kv_write_test_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=128,
  )
}

///|
fn paged_kv_write_test_fixture() -> PagedKvWriteTestFixture raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=256,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=256,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(4),
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
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=4,
    max_query_tokens=8,
    max_page_table_entries=16,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = paged_kv_write_test_compiler()
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
  let mut fallback : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering? = None
  for candidate in set.candidates() {
    match candidate {
      PagedAttention(value) => if fallback is None { fallback = Some(value) }
      _ => ()
    }
  }
  guard fallback is Some(fallback) else { abort("missing attention fallback") }
  let attention = plan.operations()[fallback.operation_id().as_int()]
  guard attention.inputs()[0] is Activation(positioned_qkv) else {
    abort("attention input is not positioned QKV")
  }
  let input = PositionedQkvActivation::new(qkv=positioned_qkv)
  let base = fallback.operands()
  let operands : Array[@launch_contract.AotOperand] = []
  for index in 0..<6 {
    operands.push(base[index])
  }
  operands.push(base[6])
  operands.push(base[8])
  operands.push(base[9])
  {
    plan,
    layout,
    profile,
    target,
    compiler,
    attention,
    fallback,
    input,
    operands,
  }
}

///|
fn paged_kv_write_test_candidate(
  fixture : PagedKvWriteTestFixture,
  operands? : ArrayView[@launch_contract.AotOperand],
  input? : PositionedQkvActivation,
  layout? : @device_layout.DeviceKvLayout,
  target? : @catalog.DeviceTarget,
  compiler? : @luna_specialization.LunaCompilerPolicy,
) -> PagedKvWriteCudaCandidate raise {
  lower_positioned_paged_kv_write_cuda_aot_candidate(
    fixture.plan,
    fixture.attention,
    input.unwrap_or(fixture.input),
    fixture.profile,
    layout.unwrap_or(fixture.layout),
    fixture.fallback,
    target.unwrap_or(fixture.target),
    compiler.unwrap_or(fixture.compiler),
    operands.unwrap_or(fixture.operands[:]),
    paged_kv_write_test_limits(),
  )
}
