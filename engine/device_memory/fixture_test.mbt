///|
fn memory_test_model() -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn memory_append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let end = shape.fold(init=2L, (size, dimension) => size * dimension) + start
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=start,
      data_end=end,
    ),
  )
  offset.val = end
}

///|
fn memory_test_layout() -> @device_materialize.DeviceWeightLayout raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  memory_append_tensor(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  memory_append_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [6L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.mlp.up_proj.weight",
    [6L, 4L],
    offset,
  )
  memory_append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    offset,
  )
  memory_append_tensor(entries, "model.norm.weight", [4L], offset)
  memory_append_tensor(entries, "lm_head.weight", [8L, 4L], offset)
  let metadata = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=16,
    payload_bytes=offset.val,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=4096,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=4096L,
      max_materialized_tensor_bytes=4096L,
    ),
  )
  @device_materialize.plan_layout(
    @llama_weights.bind(memory_test_model(), metadata),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn memory_operation(
  id : Int,
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new(
    id=@plan.OperationId::new(id),
    kind~,
    inputs~,
    outputs=[@plan.ActivationRef::new(id)],
    shape~,
    required_capability=capability,
    workspace_bytes=0L,
  )
}

///|
fn memory_semantic_plan(
  max_batch_rows? : Int = 2,
  max_sequence_tokens? : Int = 3,
) -> @plan.ModelPlan raise {
  let operations = [
    memory_operation(
      0,
      Embedding,
      [ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      EmbeddingShape(vocabulary_size=8, width=4),
      @plan.KernelCapabilityId::embedding_lookup(),
    ),
    memory_operation(
      1,
      RmsNorm,
      [Activation(@plan.ActivationRef::new(0)), Tensor(@plan.TensorRef::new(1))],
      NormalizationShape(width=4, epsilon=0.000001),
      @plan.KernelCapabilityId::rms_norm(),
    ),
    memory_operation(
      2,
      OutputProjection,
      [Activation(@plan.ActivationRef::new(1)), Tensor(@plan.TensorRef::new(5))],
      ProjectionShape(input_width=4, output_width=4),
      @plan.KernelCapabilityId::dense_projection(),
    ),
    memory_operation(
      3,
      ResidualAdd,
      [
        Activation(@plan.ActivationRef::new(0)),
        Activation(@plan.ActivationRef::new(2)),
      ],
      ElementwiseShape(width=4),
      @plan.KernelCapabilityId::residual_add(),
    ),
    memory_operation(
      4,
      RmsNorm,
      [Activation(@plan.ActivationRef::new(3)), Tensor(@plan.TensorRef::new(6))],
      NormalizationShape(width=4, epsilon=0.000001),
      @plan.KernelCapabilityId::rms_norm(),
    ),
  ]
  @plan.ModelPlan::new(
    identity=memory_test_model().identity(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows~,
      max_sequence_tokens~,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=512L),
    operations~,
    final_output=@plan.ActivationRef::new(4),
  )
}

///|
fn memory_catalog_for(
  semantic : @plan.ModelPlan,
  workspace_bytes? : Int64 = 32L,
  workspace_alignment? : Int64 = 16L,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  for operation in semantic.operations() {
    let mut found = false
    for entry in entries {
      if entry.operation_kind() == operation.kind() &&
        entry.operation_shape() == operation.shape() &&
        entry.capability() == operation.required_capability() {
        found = true
      }
    }
    if !found {
      entries.push(
        @catalog.CatalogEntry::new(
          semantic_version=@catalog.KernelSemanticVersion::v1(),
          operation_kind=operation.kind(),
          operation_shape=operation.shape(),
          capability=operation.required_capability(),
          compute_major=8,
          compute_minor=0,
          implementation=ContentAddressedAot(digest),
          workspace_bytes~,
          workspace_alignment~,
        ),
      )
    }
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries~).resolve(
    semantic, target,
  )
}

///|
fn memory_static_plan(
  max_batch_rows? : Int = 2,
  max_sequence_tokens? : Int = 3,
  workspace_bytes? : Int64 = 32L,
  workspace_alignment? : Int64 = 16L,
) -> @device_plan.StaticDevicePlan raise {
  let semantic = memory_semantic_plan(max_batch_rows~, max_sequence_tokens~)
  @device_plan.build(
    semantic,
    memory_test_layout(),
    memory_catalog_for(semantic, workspace_bytes~, workspace_alignment~),
  )
}

///|
fn memory_limits(max_arena_bytes? : Int64 = 4096L) -> DeviceMemoryLimits raise {
  @device_memory.DeviceMemoryLimits::new(
    activation_alignment=64L,
    max_arena_bytes~,
  )
}
