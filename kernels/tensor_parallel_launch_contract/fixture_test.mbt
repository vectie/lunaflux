///|
struct LaunchFixture {
  model : @plan.ModelPlan
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan
  local_layout : @device_layout.DeviceKvLayout
  full_layout : @device_layout.DeviceKvLayout
  resolved : @catalog.ResolvedKernelCatalog
  target : @catalog.DeviceTarget
}

///|
fn append_raw_tensor(
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
fn fixture_model_and_weights() -> (
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
  append_raw_tensor(entries, "model.embed_tokens.weight", [16L, 8L], position)
  append_raw_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.self_attn.q_proj.weight",
    [8L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.self_attn.k_proj.weight",
    [4L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.self_attn.v_proj.weight",
    [4L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.self_attn.o_proj.weight",
    [8L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.mlp.gate_proj.weight",
    [12L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.mlp.up_proj.weight",
    [12L, 8L],
    position,
  )
  append_raw_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [8L, 12L],
    position,
  )
  append_raw_tensor(entries, "model.norm.weight", [8L], position)
  append_raw_tensor(entries, "lm_head.weight", [16L, 8L], position)
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
fn fixture_family(index : Int) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=@catalog.AotArtifactDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
    family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
  )
}

///|
fn fixture_family_index(operation : @plan.PlanOperation) -> Int {
  match operation.kind() {
    Embedding => 0
    RmsNorm => 1
    QkRmsNorm => 7
    QkvProjection => 2
    RotaryEmbedding => 3
    CausalAttention => 4
    OutputProjection => 5
    ResidualAdd => 6
    GatedMlp => 8
    LanguageModelHead => 11
  }
}

///|
fn fixture_semantics(
  operation : @plan.PlanOperation,
  tokens_per_page : Int,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page~,
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
fn fixture() -> LaunchFixture raise {
  let (model, weights) = fixture_model_and_weights()
  let rank_plan = @llama_tensor_parallel.build_rank_plan(
    model,
    weights,
    rank=1,
    world_size=2,
    alignment_bytes=16L,
    max_arena_bytes=1048576L,
  )
  let full_layout = @device_layout.plan(
    model.identity(),
    model.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=64,
      total_page_count=8,
      max_arena_bytes=1048576L,
    ),
  )
  let full_geometry = model.kv_cache_geometry()
  let local_layout = @device_layout.plan(
    model.identity(),
    @plan.KvCacheGeometry::new(
      layer_count=full_geometry.layer_count(),
      kv_head_count=1,
      head_dimension=full_geometry.head_dimension(),
      dtype=full_geometry.dtype(),
      layout=full_geometry.layout(),
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=64,
      total_page_count=8,
      max_arena_bytes=1048576L,
    ),
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model.operations() {
    let family_index = fixture_family_index(operation)
    if index != family_index {
      continue
    }
    let implementation : @catalog.KernelImplementationId = ContentAddressedAot(
      fixture_family(family_index),
    )
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=fixture_semantics(operation, 64),
        compute_major=8,
        compute_minor=0,
        implementation~,
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
  { model, rank_plan, local_layout, full_layout, resolved, target }
}

///|
fn fixture_profile(
  rows? : Int = 2,
  tokens? : Int = 4,
  pages? : Int = 2,
) -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=rows,
    max_query_tokens=tokens,
    max_page_table_entries=pages,
  )
}

///|
fn fixture_dimensions() -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn fixture_inputs(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  changed_index? : Int = -1,
  changed_family? : Bool = false,
  full_profile? : Bool = false,
) -> Array[TensorParallelAotLaunchContractInput] raise {
  let output : Array[TensorParallelAotLaunchContractInput] = []
  for index, operation in model.operations() {
    let geometry = rank_plan.operation(operation.id()).unwrap()
    let family_index = fixture_family_index(operation)
    let family = if index == changed_index && changed_family {
      fixture_family(family_index + 100)
    } else {
      fixture_family(family_index)
    }
    output.push(
      TensorParallelAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=@catalog.AotKernelEntryPoint::new(
          family~,
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            index + 1,
          ),
        ),
        dimensions=fixture_dimensions(),
        rank=1,
        world_size=2,
        device_target=target,
        max_query_rows=profile.max_query_rows(),
        max_query_tokens=profile.max_query_tokens(),
        max_page_table_entries=profile.max_page_table_entries(),
        input_width=if index == changed_index && full_profile {
          geometry.input_width() + 1
        } else {
          geometry.input_width()
        },
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
  output
}

///|
fn fixture_limits() -> TensorParallelLaunchContractLimits raise TensorParallelLaunchContractError {
  TensorParallelLaunchContractLimits::new(
    max_profiles=4,
    max_contracts=64,
    max_operands_per_contract=16,
  )
}

///|
fn admit_fixture(
  source : LaunchFixture,
  profile : @launch_contract.PagedKernelProfile,
  inputs : ArrayView[TensorParallelAotLaunchContractInput],
  rank? : Int = 1,
  world? : Int = 2,
  target? : @catalog.DeviceTarget? = None,
  layout? : @device_layout.DeviceKvLayout? = None,
) -> TensorParallelAotLaunchContractSet raise TensorParallelLaunchContractError {
  admit_paged_rank(
    source.model,
    source.rank_plan,
    layout.unwrap_or(source.local_layout),
    source.resolved,
    expected_rank=rank,
    expected_world_size=world,
    expected_target=target.unwrap_or(source.target),
    [profile],
    inputs,
    fixture_limits(),
  )
}
