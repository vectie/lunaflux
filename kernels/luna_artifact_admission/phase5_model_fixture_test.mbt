///|
fn phase5_metadata() -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=10000.0,
      rms_norm_epsilon=0.00001,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn phase5_weight_layout() -> @device_materialize.DeviceWeightLayout raise {
  @device_materialize.plan_numeric_layout(
    phase5_model_plan(),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn phase5_model_plan(
  swap_residual_inputs? : Bool = false,
) -> @plan.ModelPlan raise {
  let layer = @plan.KvLayerId::new(0)
  let residual_inputs : Array[@plan.ValueRef] = if swap_residual_inputs {
    [
      Activation(@plan.ActivationRef::new(0)),
      Activation(@plan.ActivationRef::new(4)),
    ]
  } else {
    [
      Activation(@plan.ActivationRef::new(4)),
      Activation(@plan.ActivationRef::new(0)),
    ]
  }
  let operations = [
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=4),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(1),
      kind=RmsNorm,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=NormalizationShape(width=4, epsilon=0.00001),
      required_capability=@plan.KernelCapabilityId::rms_norm(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(2),
      kind=QkvProjection,
      inputs=[
        Activation(@plan.ActivationRef::new(1)),
        Tensor(@plan.TensorRef::new(2)),
        Tensor(@plan.TensorRef::new(3)),
        Tensor(@plan.TensorRef::new(4)),
      ],
      outputs=[@plan.ActivationRef::new(2)],
      shape=ProjectionShape(input_width=4, output_width=8),
      required_capability=@plan.KernelCapabilityId::qkv_projection(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(3),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(2))],
      outputs=[@plan.ActivationRef::new(3)],
      shape=RotaryShape(
        query_heads=2,
        kv_heads=1,
        head_dimension=2,
        rope_theta=10000.0,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=0L,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(4),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(3))],
      outputs=[@plan.ActivationRef::new(4)],
      shape=AttentionShape(query_heads=2, kv_heads=1, head_dimension=2),
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
    ),
    @plan.PlanOperation::new_with_runtime_inputs(
      id=@plan.OperationId::new(5),
      kind=ResidualAdd,
      inputs=residual_inputs,
      outputs=[@plan.ActivationRef::new(5)],
      shape=ElementwiseShape(width=4),
      required_capability=@plan.KernelCapabilityId::residual_add(),
      workspace_bytes=0L,
      runtime_inputs=[StepCounts],
    ),
  ]
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=phase5_metadata().content_digest(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations~,
    final_output=@plan.ActivationRef::new(5),
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn phase5_kv_layout() -> @device_layout.DeviceKvLayout raise {
  let model_plan = phase5_model_plan()
  @device_layout.plan(
    model_plan.identity(),
    model_plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=8,
      max_arena_bytes=4096L,
    ),
  )
}
