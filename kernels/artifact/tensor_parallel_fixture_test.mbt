///|
fn tp_append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  position : Ref[Int64],
) -> Unit {
  let mut elements = 1L
  for dimension in shape {
    elements *= dimension
  }
  let bytes = elements * 2L
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=position.val,
      data_end=position.val + bytes,
    ),
  )
  position.val += bytes
}

///|
fn tp_model_and_weights() -> (
  @plan.ModelPlan,
  @llama_weights.LlamaWeightBindings,
) raise {
  let spec = @spec.LlamaModelSpec::new(
    vocabulary_size=16,
    hidden_size=8,
    intermediate_size=12,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=32,
    rope_theta=10000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let position = Ref(0L)
  tp_append_tensor(entries, "model.embed_tokens.weight", [16L, 8L], position)
  tp_append_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [8L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [4L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [4L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [8L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [12L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.mlp.up_proj.weight",
    [12L, 8L],
    position,
  )
  tp_append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [8L, 12L],
    position,
  )
  tp_append_tensor(entries, "model.norm.weight", [8L], position)
  tp_append_tensor(entries, "lm_head.weight", [16L, 8L], position)
  let tensors = @safetensors.validate_metadata(
    entries,
    encoded_header_bytes=1024,
    payload_bytes=position.val,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=4096,
      max_tensor_count=entries.length(),
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=position.val,
      max_materialized_tensor_bytes=position.val,
    ),
  )
  let plan = @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
  (plan, @llama_weights.bind(metadata, plan, tensors))
}

///|
fn tp_family_index(operation : @plan.PlanOperation) -> Int {
  match operation.kind() {
    Embedding => 20
    RmsNorm => 21
    QkRmsNorm => 27
    QkvProjection => 22
    RotaryEmbedding => 23
    CausalAttention => 24
    OutputProjection => 25
    ResidualAdd => 26
    GatedMlp => 28
    LanguageModelHead => 31
  }
}

///|
fn tp_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=64,
        ),
      )
  }
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  )
}

///|
fn tp_layouts(
  model : @plan.ModelPlan,
) -> (@device_layout.DeviceKvLayout, @device_layout.DeviceKvLayout) raise {
  let full_geometry = model.kv_cache_geometry()
  let limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page=64,
    total_page_count=8,
    max_arena_bytes=1048576L,
  )
  let full = @device_layout.plan(model.identity(), full_geometry, limits)
  let local_layout = @device_layout.plan(
    model.identity(),
    @plan.KvCacheGeometry::new(
      layer_count=full_geometry.layer_count(),
      kv_head_count=1,
      head_dimension=full_geometry.head_dimension(),
      dtype=full_geometry.dtype(),
      layout=full_geometry.layout(),
    ),
    limits,
  )
  (full, local_layout)
}

///|
fn tp_profiles() -> Array[@launch_contract.PagedKernelProfile] raise {
  [
    @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=2,
      max_query_tokens=4,
      max_page_table_entries=2,
    ),
    @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(2),
      max_query_rows=1,
      max_query_tokens=1,
      max_page_table_entries=1,
    ),
  ]
}

///|
fn tp_contracts(
  digest : @catalog.AotArtifactDigest,
) -> @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet raise {
  let (model, weights) = tp_model_and_weights()
  let rank_plan = @llama_tensor_parallel.build_rank_plan(
    model,
    weights,
    rank=1,
    world_size=2,
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
  let (full_layout, local_layout) = tp_layouts(model)
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model.operations() {
    let family_index = tp_family_index(operation)
    if family_index != index + 20 {
      continue
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=tp_semantics(operation),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(family(digest, family_index)),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 0L {
          1L
        } else {
          16L
        },
      ),
    )
  }
  let resolved = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v3(),
    entries~,
  ).resolve_paged(model, full_layout, target)
  let profiles = tp_profiles()
  let inputs : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
  ] = []
  for profile in profiles {
    for operation in model.operations() {
      let family_index = tp_family_index(operation)
      let geometry = rank_plan.operation(operation.id()).unwrap()
      inputs.push(
        @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput::new(
          profile_id=profile.id(),
          operation_id=operation.id(),
          entry_point=entry(digest, family_index, family_index + 100),
          dimensions=dimensions(profile.max_query_tokens()),
          rank=1,
          world_size=2,
          device_target=target,
          max_query_rows=profile.max_query_rows(),
          max_query_tokens=profile.max_query_tokens(),
          max_page_table_entries=profile.max_page_table_entries(),
          input_width=geometry.input_width(),
          output_width=geometry.output_width(),
          inner_width=geometry.inner_width(),
          query_head_count=geometry.query_head_count(),
          kv_head_count=geometry.kv_head_count(),
          head_dimension=geometry.head_dimension(),
          kv_layout_version=1,
          tokens_per_page=64,
        ),
      )
    }
  }
  @tensor_parallel_launch_contract.admit_paged_rank(
    model,
    rank_plan,
    local_layout,
    resolved,
    expected_rank=1,
    expected_world_size=2,
    expected_target=target,
    profiles,
    inputs,
    @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
      max_profiles=2,
      max_contracts=24,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn tp_export_inputs(
  contracts : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
) -> Array[@artifact.KernelEntryPointInput] {
  let output : Array[@artifact.KernelEntryPointInput] = []
  for contract in contracts.contracts() {
    let identity = contract.entry_point()
    if !output.any(input => input.identity() == identity) {
      output.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=identity,
          function_symbol="shared_tp_symbol",
        ),
      )
    }
  }
  output
}
