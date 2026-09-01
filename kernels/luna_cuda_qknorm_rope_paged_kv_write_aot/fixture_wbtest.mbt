///|
priv struct PostprocessTestFixture {
  plan : @plan.ModelPlan
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan
  profile : @launch_contract.PagedKernelProfile
  layout : @device_layout.DeviceKvLayout
  target : @catalog.DeviceTarget
  compiler : @luna_specialization.LunaCompilerPolicy
  operands : Array[@launch_contract.AotOperand]
}

///|
fn postprocess_test_limits() -> QkNormRopePagedKvWriteAotLimits raise {
  QkNormRopePagedKvWriteAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
    max_module_bytes=1048576L,
    max_receipt_bytes=65536,
    max_binding_bytes=65536,
    maximum_grid_x=4096,
  )
}

///|
fn postprocess_test_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~)
}

///|
fn postprocess_test_compiler() -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="8".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn postprocess_test_fixture(
  head_dimension? : Int = 128,
) -> PostprocessTestFixture raise {
  let model = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=1024,
    hidden_size=head_dimension * 4,
    intermediate_size=head_dimension * 8,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension~,
    context_length=256,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let plan = @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("4".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
  let semantic = @luna_fusion_plan.match_qkv_qknorm_positioned_rope_paged_kv_write(
    plan,
    @plan.OperationId::new(2),
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
    id=@launch_contract.KernelProfileId::from_stable_int(31),
    max_query_rows=4,
    max_query_tokens=5000,
    max_page_table_entries=64,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let compiler = postprocess_test_compiler()
  let (_, _, head_dimension) = semantic.attention_shape()
  let (_, qkv_width) = semantic.projection_shape()
  let (_, norm_index, _, _) = semantic.operation_indices()
  guard plan.operations()[norm_index].inputs()
    is [Activation(_), Tensor(query_norm), Tensor(key_norm)] else {
    abort("QKNorm fixture did not expose norm weights")
  }
  let rows = profile.max_query_rows().to_int64()
  let tokens = profile.max_query_tokens().to_int64()
  let qkv_bytes = tokens * qkv_width.to_int64() * 2L
  let norm_bytes = head_dimension.to_int64() * 2L
  let operands = [
    postprocess_test_operand(RuntimeInput(StepCounts), 20L, 4L),
    postprocess_test_operand(RuntimeInput(QueryPositions), tokens * 4L, 4L),
    postprocess_test_operand(
      RuntimeInput(QueryRowOffsets),
      (rows + 1L) * 4L,
      4L,
    ),
    postprocess_test_operand(RuntimeInput(SequenceLengths), rows * 4L, 4L),
    postprocess_test_operand(
      RuntimeInput(PageTableOffsets),
      (rows + 1L) * 4L,
      4L,
    ),
    postprocess_test_operand(
      RuntimeInput(PageTablePageIndices),
      profile.max_page_table_entries().to_int64() * 4L,
      4L,
    ),
    postprocess_test_operand(
      ActivationInput(semantic.projected_qkv()),
      qkv_bytes,
      16L,
    ),
    postprocess_test_operand(WeightInput(query_norm), norm_bytes, 16L),
    postprocess_test_operand(WeightInput(key_norm), norm_bytes, 16L),
    postprocess_test_operand(
      ActivationOutput(semantic.positioned_qkv()),
      qkv_bytes,
      16L,
    ),
    postprocess_test_operand(
      PersistentKvComponent(semantic.kv_layer(), Key),
      layout.component_stride_bytes(),
      layout.segment_alignment_bytes(),
    ),
    postprocess_test_operand(
      PersistentKvComponent(semantic.kv_layer(), Value),
      layout.component_stride_bytes(),
      layout.segment_alignment_bytes(),
    ),
  ]
  { plan, semantic, profile, layout, target, compiler, operands }
}

///|
fn postprocess_test_candidate(
  fixture : PostprocessTestFixture,
  operands? : ArrayView[@launch_contract.AotOperand],
) -> QkNormRopePagedKvWriteCudaCandidate raise {
  lower_qknorm_rope_paged_kv_write_cuda_aot_candidate(
    fixture.plan,
    fixture.semantic,
    fixture.profile,
    fixture.layout,
    fixture.target,
    fixture.compiler,
    operands.unwrap_or(fixture.operands[:]),
    postprocess_test_limits(),
  )
}
