///|
fn device_plan_test_model(
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
fn device_plan_tensor_bytes(shape : ArrayView[Int64]) -> Int64 {
  shape.fold(init=2L, (size, dimension) => size * dimension)
}

///|
fn device_plan_append_raw(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  offset : Ref[Int64],
) -> Unit {
  let start = offset.val
  let end = start + device_plan_tensor_bytes(shape)
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
fn device_plan_test_bindings() -> @llama_weights.LlamaWeightBindings raise {
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let offset = Ref(0L)
  device_plan_append_raw(entries, "model.embed_tokens.weight", [8L, 4L], offset)
  device_plan_append_raw(
    entries,
    "model.layers.0.input_layernorm.weight",
    [4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [4L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [2L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [2L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [4L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [6L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.mlp.up_proj.weight",
    [6L, 4L],
    offset,
  )
  device_plan_append_raw(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    offset,
  )
  device_plan_append_raw(entries, "model.norm.weight", [4L], offset)
  device_plan_append_raw(entries, "lm_head.weight", [8L, 4L], offset)
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
  let model = device_plan_test_model()
  let plan = @llama.build(model)
  @llama_weights.bind(model, plan, metadata)
}

///|
fn device_plan_test_layout() -> @device_materialize.DeviceWeightLayout raise {
  @device_materialize.plan_layout(
    device_plan_test_bindings(),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn device_plan_numeric_layout(
  model_plan : @plan.ModelPlan,
) -> @device_materialize.DeviceWeightLayout raise {
  @device_materialize.plan_numeric_layout(
    model_plan,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn device_plan_test_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn device_plan_catalog_for(
  model_plan : @plan.ModelPlan,
  workspace_bytes? : Int64 = 16L,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  for operation in model_plan.operations() {
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
          operation.id().as_int() + 1,
        ),
      )
      entries.push(
        @catalog.CatalogEntry::new(
          semantic_version=@catalog.KernelSemanticVersion::v1(),
          operation_kind=operation.kind(),
          operation_shape=operation.shape(),
          capability=operation.required_capability(),
          compute_major=8,
          compute_minor=0,
          implementation=ContentAddressedAot(family),
          workspace_bytes~,
          workspace_alignment=16L,
        ),
      )
    }
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries~).resolve(
    model_plan,
    device_plan_test_target(),
  )
}

///|
fn device_plan_base_constraints() -> @plan.ShapeConstraints raise {
  @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=16,
    vocabulary_size=8,
    hidden_size=4,
    intermediate_size=6,
    attention_head_count=2,
    kv_head_count=1,
    head_dimension=2,
  )
}

///|
fn device_plan_embedding(
  tensor_reference : Int,
  workspace_bytes? : Int64 = 0L,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[
      ModelInput(TokenIds),
      Tensor(@plan.TensorRef::new(tensor_reference)),
    ],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=8, width=4),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes~,
  )
}
