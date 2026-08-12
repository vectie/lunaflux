///|
fn profile_test_model(
  digest? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> @spec.LlamaModelMetadata raise {
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
    @spec.ContentDigest::from_sha256(digest),
  )
}

///|
fn profile_append_tensor(
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
fn profile_test_layout(
  model? : @spec.LlamaModelMetadata = profile_test_model(),
) -> @device_materialize.DeviceWeightLayout raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  profile_append_tensor(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  profile_append_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [6L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.mlp.up_proj.weight",
    [6L, 4L],
    offset,
  )
  profile_append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    offset,
  )
  profile_append_tensor(entries, "model.norm.weight", [4L], offset)
  profile_append_tensor(entries, "lm_head.weight", [8L, 4L], offset)
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
    @llama_weights.bind(model, metadata),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn profile_test_target(compute_major? : Int = 8) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major~,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn profile_family_id(kind : @plan.OperationKind) -> Int {
  match kind {
    Embedding => 1
    RmsNorm => 2
    QkvProjection => 3
    RotaryEmbedding => 4
    CausalAttention => 5
    OutputProjection => 6
    ResidualAdd => 7
    GatedMlp => 8
    LanguageModelHead => 9
  }
}

///|
fn profile_catalog_for(
  semantic : @plan.ModelPlan,
  target? : @catalog.DeviceTarget = profile_test_target(),
  workspace_bytes? : Int64 = 32L,
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
      let family = @catalog.AotKernelFamily::new(
        module_digest=digest,
        family_id=@catalog.AotKernelFamilyId::from_stable_int(
          profile_family_id(operation.kind()),
        ),
      )
      entries.push(
        @catalog.CatalogEntry::new(
          semantic_version=@catalog.KernelSemanticVersion::v1(),
          operation_kind=operation.kind(),
          operation_shape=operation.shape(),
          capability=operation.required_capability(),
          compute_major=target.compute_major(),
          compute_minor=target.compute_minor(),
          implementation=ContentAddressedAot(family),
          workspace_bytes~,
          workspace_alignment=16L,
        ),
      )
    }
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries~).resolve(
    semantic, target,
  )
}

///|
fn profile_static_plan(
  semantic? : @plan.ModelPlan = @llama.build(profile_test_model()),
  layout? : @device_materialize.DeviceWeightLayout = profile_test_layout(),
  target? : @catalog.DeviceTarget = profile_test_target(),
  workspace_bytes? : Int64 = 32L,
) -> @device_plan.StaticDevicePlan raise {
  @device_plan.build(
    semantic,
    layout,
    profile_catalog_for(semantic, target~, workspace_bytes~),
  )
}

///|
fn profile_memory_plan(
  static_plan? : @device_plan.StaticDevicePlan = profile_static_plan(),
) -> @device_memory.DeviceMemoryPlan raise {
  @device_memory.plan(
    static_plan,
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
  )
}

///|
fn profile_embedding_plan(
  max_batch_rows : Int,
  max_sequence_tokens : Int,
) -> @plan.ModelPlan raise {
  let model = profile_test_model()
  let operation = @plan.PlanOperation::new(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  @plan.ModelPlan::new(
    identity=model.identity(),
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
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=64L),
    operations=[operation],
    final_output=@plan.ActivationRef::new(0),
  )
}
