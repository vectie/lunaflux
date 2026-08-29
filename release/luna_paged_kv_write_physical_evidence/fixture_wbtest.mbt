///|
priv struct KvPhysicalFixture {
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate
  compiled : @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact
}

///|
fn kv_fixture_digest(
  value : String,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest {
  @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(value) catch {
    _ => abort("fixture digest is invalid")
  }
}

///|
fn kv_physical_fixture() -> KvPhysicalFixture raise {
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
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="9".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
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
  for value in set.candidates() {
    match value {
      PagedAttention(item) => if fallback is None { fallback = Some(item) }
      _ => ()
    }
  }
  guard fallback is Some(fallback) else { abort("missing fallback") }
  let attention = plan.operations()[fallback.operation_id().as_int()]
  guard attention.inputs()[0] is Activation(positioned_qkv) else {
    abort("missing positioned QKV")
  }
  let base = fallback.operands()
  let operands : Array[@launch_contract.AotOperand] = []
  for index in 0..<7 {
    operands.push(base[index])
  }
  operands.push(base[8])
  operands.push(base[9])
  let limits = @luna_cuda_paged_kv_write_aot.PagedKvWriteAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
    max_module_bytes=1048576L,
    max_receipt_bytes=65536,
    max_binding_bytes=65536,
  )
  let candidate = @luna_cuda_paged_kv_write_aot.lower_positioned_paged_kv_write_cuda_aot_candidate(
    plan,
    attention,
    @luna_cuda_paged_kv_write_aot.PositionedQkvActivation::new(
      qkv=positioned_qkv,
    ),
    profile,
    layout,
    fallback,
    target,
    compiler,
    operands[:],
    limits,
  )
  let module_ = b"fixture-paged-kv-write-cubin"
  let receipt = @luna_cuda_paged_kv_write_aot.positioned_paged_kv_write_compile_receipt(
    candidate,
    "d".repeat(64),
    module_,
  )
  let compiled = @luna_cuda_paged_kv_write_aot.bind_positioned_paged_kv_write_compiled_artifact(
    candidate, module_, module_, receipt, limits,
  )
  { candidate, compiled }
}
