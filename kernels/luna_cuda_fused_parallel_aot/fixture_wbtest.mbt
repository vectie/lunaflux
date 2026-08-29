///|
priv struct FusedParallelTestFixture {
  plan : @plan.ModelPlan
  layout : @device_layout.DeviceKvLayout
  profile : @launch_contract.PagedKernelProfile
  target : @catalog.DeviceTarget
  compiler : @luna_specialization.LunaCompilerPolicy
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  residual : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering
}

///|
fn fused_test_plan() -> @plan.ModelPlan raise {
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
  @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::new(4),
  )
}

///|
fn fused_test_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=128,
  )
}

///|
fn fused_test_limits() -> FusedParallelAotLimits raise {
  FusedParallelAotLimits::new(
    maximum_source_bytes=1048576,
    maximum_recipe_bytes=65536,
    maximum_promotion_bytes=65536,
  )
}

///|
fn fused_test_fixture() -> FusedParallelTestFixture raise {
  let plan = fused_test_plan()
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
  let mut rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering? = None
  let mut attention : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering? = None
  let mut residual : @luna_cuda_pointwise_aot.PointwiseCudaLowering? = None
  let mut norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering? = None
  for candidate in set.candidates() {
    match candidate {
      Projection(value) =>
        if value.family() == QkvProjection && qkv is None {
          qkv = Some(value)
        }
      Pointwise(value) => {
        if value.family() == PositionedRotary && rope is None {
          rope = Some(value)
        }
        if value.family() == ResidualAdd && value.operation_id().as_int() == 6 {
          residual = Some(value)
        }
        if value.family() == RmsNorm && value.operation_id().as_int() == 7 {
          norm = Some(value)
        }
      }
      PagedAttention(value) => if attention is None { attention = Some(value) }
    }
  }
  guard (qkv, rope, attention, residual, norm)
    is (Some(qkv), Some(rope), Some(attention), Some(residual), Some(norm)) else {
    abort("fused candidate fixture lacked reference fallback")
  }
  let kv_operands : Array[@launch_contract.AotOperand] = []
  for operand in attention.operands() {
    match operand.role() {
      ActivationOutput(_) => ()
      _ => kv_operands.push(operand)
    }
  }
  let attention_operation = plan.operations()[attention.operation_id().as_int()]
  let kv_write = @luna_cuda_pointwise_aot.lower_auxiliary_kv_cache_write_cuda_aot(
    attention_operation,
    profile,
    layout,
    @catalog.AotKernelEntryPointId::from_stable_int(2901),
    target,
    compiler,
    kv_operands,
    @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  )
  {
    plan,
    layout,
    profile,
    target,
    compiler,
    qkv,
    rope,
    kv_write,
    residual,
    norm,
  }
}

///|
fn fused_test_qkv_candidate(
  fixture : FusedParallelTestFixture,
) -> FusedQkvRopeKvWriteCudaCandidate raise {
  lower_fused_qkv_rope_paged_kv_write_candidate(
    fixture.plan,
    fixture.layout,
    fixture.qkv,
    fixture.rope,
    fixture.kv_write,
    fixture.target,
    fixture.compiler,
    fused_test_limits(),
  )
}

///|
fn fused_test_residual_norm_candidate(
  fixture : FusedParallelTestFixture,
) -> FusedResidualRmsNormCudaCandidate raise {
  lower_fused_residual_rmsnorm_candidate(
    fixture.plan,
    fixture.residual,
    fixture.norm,
    fixture.target,
    fixture.compiler,
    fused_test_limits(),
  )
}
