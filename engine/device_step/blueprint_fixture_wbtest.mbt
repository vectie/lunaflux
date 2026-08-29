///|
fn blueprint_metadata(
  digest_head? : Char = '0',
) -> @spec.LlamaModelMetadata raise {
  let digest = digest_head.to_string() +
    "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
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
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn blueprint_weight_layout(
  plan : @plan.ModelPlan,
) -> @device_materialize.DeviceWeightLayout raise {
  @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn blueprint_operation(
  id : Int,
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
  workspace : Int64,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(id),
    kind~,
    inputs~,
    outputs=[@plan.ActivationRef::new(id)],
    shape~,
    required_capability=capability,
    workspace_bytes=workspace,
    runtime_inputs=[StepCounts],
  )
}

///|
fn blueprint_model_plan(
  metadata : @spec.LlamaModelMetadata,
  extra_projection? : Bool = false,
  rotary_workspace? : Int64 = 64L,
  replace_qkv_with_projection? : Bool = false,
  rope_theta? : Double = 10000.0,
  final_language_model_head? : Bool = false,
) -> @plan.ModelPlan raise {
  let operations : Array[@plan.PlanOperation] = []
  operations.push(
    blueprint_operation(
      0,
      Embedding,
      [ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      EmbeddingShape(vocabulary_size=8, width=4),
      @plan.KernelCapabilityId::embedding_lookup(),
      0L,
    ),
  )
  let mut source = @plan.ActivationRef::new(0)
  operations.push(
    blueprint_operation(
      1,
      RmsNorm,
      [Activation(source), Tensor(@plan.TensorRef::new(1))],
      NormalizationShape(width=4, epsilon=0.000001),
      @plan.KernelCapabilityId::rms_norm(),
      0L,
    ),
  )
  source = @plan.ActivationRef::new(1)
  if extra_projection {
    let projection_id = operations.length()
    operations.push(
      blueprint_operation(
        projection_id,
        OutputProjection,
        [Activation(source), Tensor(@plan.TensorRef::new(5))],
        ProjectionShape(input_width=4, output_width=4),
        @plan.KernelCapabilityId::dense_projection(),
        0L,
      ),
    )
    source = @plan.ActivationRef::new(projection_id)
  }
  let qkv_id = operations.length()
  if replace_qkv_with_projection {
    operations.push(
      blueprint_operation(
        qkv_id,
        OutputProjection,
        [Activation(source), Tensor(@plan.TensorRef::new(2))],
        ProjectionShape(input_width=4, output_width=4),
        @plan.KernelCapabilityId::dense_projection(),
        0L,
      ),
    )
  } else {
    operations.push(
      blueprint_operation(
        qkv_id,
        QkvProjection,
        [
          Activation(source),
          Tensor(@plan.TensorRef::new(2)),
          Tensor(@plan.TensorRef::new(3)),
          Tensor(@plan.TensorRef::new(4)),
        ],
        ProjectionShape(input_width=4, output_width=8),
        @plan.KernelCapabilityId::qkv_projection(),
        0L,
      ),
    )
  }
  let layer = @plan.KvLayerId::new(0)
  let rotary_id = operations.length()
  operations.push(
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(rotary_id),
      kind=RotaryEmbedding,
      inputs=[Activation(@plan.ActivationRef::new(qkv_id))],
      outputs=[@plan.ActivationRef::new(rotary_id)],
      shape=RotaryShape(
        query_heads=2,
        kv_heads=1,
        head_dimension=2,
        rope_theta~,
      ),
      required_capability=@plan.KernelCapabilityId::positioned_rotary_embedding(),
      workspace_bytes=rotary_workspace,
      kv_layer=layer,
      runtime_inputs=[StepCounts, QueryPositions],
      persistent_state=NoPersistentState,
    ),
  )
  let attention_id = operations.length()
  operations.push(
    @plan.PlanOperation::new_with_execution(
      id=@plan.OperationId::new(attention_id),
      kind=CausalAttention,
      inputs=[Activation(@plan.ActivationRef::new(rotary_id))],
      outputs=[@plan.ActivationRef::new(attention_id)],
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
  )
  let final_output = if final_language_model_head {
    let head_id = operations.length()
    operations.push(
      blueprint_operation(
        head_id,
        LanguageModelHead,
        [
          Activation(@plan.ActivationRef::new(attention_id)),
          Tensor(@plan.TensorRef::new(5)),
        ],
        ProjectionShape(input_width=4, output_width=8),
        @plan.KernelCapabilityId::dense_projection(),
        0L,
      ),
    )
    @plan.ActivationRef::new(head_id)
  } else {
    @plan.ActivationRef::new(attention_id)
  }
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=metadata.content_digest(),
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
    final_output~,
    kv_execution=PagedKeyValue,
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn blueprint_layout(
  plan : @plan.ModelPlan,
  pages? : Int = 8,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=pages,
      max_arena_bytes=8192L,
    ),
  )
}

///|
fn blueprint_target(minor? : Int = 0) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}
