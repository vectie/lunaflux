///|
fn pointwise_test_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
}

///|
fn pointwise_test_entry(id : Int) -> @catalog.AotKernelEntryPointId raise {
  @catalog.AotKernelEntryPointId::from_stable_int(id)
}

///|
fn pointwise_test_target(
  supports_bf16? : Bool = true,
) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16~,
    supports_cublas_lt=true,
  )
}

///|
fn pointwise_test_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="2222222222222222222222222222222222222222222222222222222222222222",
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
fn pointwise_test_limits() -> @luna_cuda_pointwise_aot.PointwiseAotLimits raise {
  @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
    max_source_bytes=16384,
    max_recipe_bytes=8192,
  )
}

///|
fn pointwise_operand(
  role : @launch_contract.AotOperandRole,
  bytes : Int64,
  alignment? : Int64 = 256L,
) -> @launch_contract.AotOperand raise {
  @launch_contract.AotOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn pointwise_embedding_operation() -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=16, width=8),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
}

///|
fn pointwise_rms_operation() -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(1),
    kind=RmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Tensor(@plan.TensorRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(1)],
    shape=NormalizationShape(width=8, epsilon=0.00001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
}

///|
fn pointwise_qk_rms_operation(
  capability? : @plan.KernelCapabilityId = @plan.KernelCapabilityId::qk_rms_norm(),
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(2),
    kind=QkRmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(1)),
      Tensor(@plan.TensorRef::new(2)),
      Tensor(@plan.TensorRef::new(3)),
    ],
    outputs=[@plan.ActivationRef::new(2)],
    shape=QkNormalizationShape(
      query_heads=2,
      kv_heads=1,
      head_dimension=4,
      epsilon=0.000001,
    ),
    required_capability=capability,
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
}

///|
fn pointwise_residual_operation() -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(2),
    kind=ResidualAdd,
    inputs=[
      Activation(@plan.ActivationRef::new(0)),
      Activation(@plan.ActivationRef::new(1)),
    ],
    outputs=[@plan.ActivationRef::new(2)],
    shape=ElementwiseShape(width=8),
    required_capability=@plan.KernelCapabilityId::residual_add(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
}

///|
fn pointwise_rotary_operation() -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_execution(
    id=@plan.OperationId::new(3),
    kind=RotaryEmbedding,
    inputs=[Activation(@plan.ActivationRef::new(2))],
    outputs=[@plan.ActivationRef::new(3)],
    shape=RotaryShape(
      query_heads=2,
      kv_heads=1,
      head_dimension=4,
      rope_theta=10000.0,
    ),
    required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
    workspace_bytes=0L,
    kv_layer=@plan.KvLayerId::new(0),
    runtime_inputs=[StepCounts, QueryPositions],
    persistent_state=NoPersistentState,
  )
}

///|
fn pointwise_attention_operation() -> @plan.PlanOperation raise {
  let layer = @plan.KvLayerId::new(0)
  @plan.PlanOperation::new_with_execution(
    id=@plan.OperationId::new(4),
    kind=CausalAttention,
    inputs=[Activation(@plan.ActivationRef::new(3))],
    outputs=[@plan.ActivationRef::new(4)],
    shape=AttentionShape(query_heads=2, kv_heads=1, head_dimension=4),
    required_capability=@plan.KernelCapabilityId::paged_causal_attention(),
    workspace_bytes=0L,
    kv_layer=layer,
    runtime_inputs=[
      StepCounts,
      QueryPositions,
      QueryRowOffsets,
      SequenceLengths,
      PageTableOffsets,
      PageTablePageIndices,
    ],
    persistent_state=KvCacheReadWrite(layer),
  )
}

///|
fn pointwise_test_layout() -> @device_layout.DeviceKvLayout raise {
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
    @spec.PlanDigest::from_sha256(
      "4444444444444444444444444444444444444444444444444444444444444444",
    ),
  )
  @device_layout.plan(
    identity,
    @plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=3,
      max_arena_bytes=1536L,
    ),
  )
}
