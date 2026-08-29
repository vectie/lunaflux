///|
struct ExecutionFixture {
  model : @plan.ModelPlan
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan
  artifacts : @artifact.KernelArtifactBundle
  profiles : Array[@launch_contract.PagedKernelProfile]
}

///|
fn exec_weight_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn exec_weight_file() -> Bytes {
  let entries : Array[String] = [
    exec_weight_entry("model.embed_tokens.weight", "[16,8]", 0, 256),
    exec_weight_entry("model.layers.0.input_layernorm.weight", "[8]", 256, 272),
    exec_weight_entry(
      "model.layers.0.self_attn.q_proj.weight", "[8,8]", 272, 400,
    ),
    exec_weight_entry(
      "model.layers.0.self_attn.k_proj.weight", "[4,8]", 400, 464,
    ),
    exec_weight_entry(
      "model.layers.0.self_attn.v_proj.weight", "[4,8]", 464, 528,
    ),
    exec_weight_entry(
      "model.layers.0.self_attn.o_proj.weight", "[8,8]", 528, 656,
    ),
    exec_weight_entry(
      "model.layers.0.post_attention_layernorm.weight", "[8]", 656, 672,
    ),
    exec_weight_entry("model.layers.0.mlp.gate_proj.weight", "[12,8]", 672, 864),
    exec_weight_entry("model.layers.0.mlp.up_proj.weight", "[12,8]", 864, 1056),
    exec_weight_entry(
      "model.layers.0.mlp.down_proj.weight", "[8,12]", 1056, 1248,
    ),
    exec_weight_entry("model.norm.weight", "[8]", 1248, 1264),
    exec_weight_entry("lm_head.weight", "[16,8]", 1264, 1520),
  ]
  let header = @utf8.encode("{" + entries.join(",") + "}")
  let bytes : Array[Byte] = []
  for shift in 0..<8 {
    bytes.push((header.length().to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    bytes.push(byte)
  }
  for index in 0..<1520 {
    bytes.push((index % 251).to_byte())
  }
  Bytes::from_array(bytes)
}

///|
fn exec_metadata(file : BytesView) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
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
    ),
    @spec.ContentDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(file)),
    ),
  )
}

///|
fn exec_file_limits() -> @device_materialize.DeviceWeightFileLimits raise {
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=31,
    safetensors=@safetensors.HeaderLimits::new(
      max_header_bytes=8192,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=4096L,
      max_materialized_tensor_bytes=4096L,
    ),
  )
}

///|
fn exec_target(minor? : Int = 0) -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn exec_topology(
  target : @catalog.DeviceTarget,
) -> @device_topology.AdmittedLocalTopology raise {
  let ranks = [
    @device_topology.LocalTopologyRank::new(
      ordinal=2,
      name="rank-two",
      total_memory_bytes=17179869184L,
      target~,
    ),
    @device_topology.LocalTopologyRank::new(
      ordinal=5,
      name="rank-five",
      total_memory_bytes=17179869184L,
      target~,
    ),
  ]
  @device_topology.admit_local_topology(
    @device_topology.LocalTopologyDeclaration::new(ranks, [
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
    ]),
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      [
        @device_topology.LocalProbeDevice::new(
          ordinal=2,
          name="rank-two",
          total_memory_bytes=17179869184L,
          target~,
        ),
        @device_topology.LocalProbeDevice::new(
          ordinal=5,
          name="rank-five",
          total_memory_bytes=17179869184L,
          target~,
        ),
      ],
      [
        @device_topology.LocalPeerAccess::new(
          source_ordinal=2,
          destination_ordinal=5,
        ),
        @device_topology.LocalPeerAccess::new(
          source_ordinal=5,
          destination_ordinal=2,
        ),
      ],
    ),
  )
}

///|
fn exec_family_index(operation : @plan.PlanOperation) -> Int {
  match operation.kind() {
    Embedding => 0
    RmsNorm => 1
    QkvProjection => 2
    RotaryEmbedding => 3
    CausalAttention => 4
    OutputProjection => 5
    ResidualAdd => 6
    GatedMlp => 8
    LanguageModelHead => 11
    QkRmsNorm => abort("tensor-parallel fixture rejects QK RMS normalization")
  }
}

///|
fn exec_semantics(
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
fn exec_profile(
  id : Int,
  tokens : Int,
) -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(id),
    max_query_rows=2,
    max_query_tokens=tokens,
    max_page_table_entries=2,
  )
}

///|
fn exec_dimensions() -> @launch_contract.AotLaunchDimensions raise {
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
fn exec_limits() -> TensorParallelExecutionPlanLimits raise {
  TensorParallelExecutionPlanLimits::new(
    max_operations=64,
    max_operands=1024,
    max_descriptors=16,
    max_collectives=32,
    max_descriptor_bytes=1048576L,
    max_activation_workspace_bytes=1048576L,
    artifact_limits=@artifact.KernelArtifactLimits::new(
      max_modules=4,
      max_entry_points=64,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=64,
    ),
  )
}
