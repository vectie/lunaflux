///|
fn rank_local_stream_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn rank_local_weight_file() -> Bytes {
  let entries : Array[String] = [
    rank_local_stream_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    rank_local_stream_entry(
      "model.layers.0.input_layernorm.weight", "[4]", 64, 72,
    ),
    rank_local_stream_entry(
      "model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104,
    ),
    rank_local_stream_entry(
      "model.layers.0.self_attn.k_proj.weight", "[4,4]", 104, 136,
    ),
    rank_local_stream_entry(
      "model.layers.0.self_attn.v_proj.weight", "[4,4]", 136, 168,
    ),
    rank_local_stream_entry(
      "model.layers.0.self_attn.o_proj.weight", "[4,4]", 168, 200,
    ),
    rank_local_stream_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 200, 208,
    ),
    rank_local_stream_entry(
      "model.layers.0.mlp.gate_proj.weight", "[6,4]", 208, 256,
    ),
    rank_local_stream_entry(
      "model.layers.0.mlp.up_proj.weight", "[6,4]", 256, 304,
    ),
    rank_local_stream_entry(
      "model.layers.0.mlp.down_proj.weight", "[4,6]", 304, 352,
    ),
    rank_local_stream_entry("model.norm.weight", "[4]", 352, 360),
    rank_local_stream_entry("lm_head.weight", "[8,4]", 360, 424),
  ]
  let header = @utf8.encode("{" + entries.join(",") + "}")
  let output : Array[Byte] = []
  for shift in 0..<8 {
    output.push((header.length().to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    output.push(byte)
  }
  for index in 0..<424 {
    output.push((index % 251).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn rank_local_metadata(file : BytesView) -> @spec.LlamaModelMetadata raise {
  @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=2,
      context_length=16,
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
fn rank_local_file_limits() -> @device_materialize.DeviceWeightFileLimits raise {
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=17,
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
fn rank_local_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn rank_local_topology() -> @device_topology.AdmittedLocalTopology raise {
  let target = rank_local_target()
  let ranks = [
    @device_topology.LocalTopologyRank::new(
      ordinal=2,
      name="rank-two",
      total_memory_bytes=2147483648L,
      target~,
    ),
    @device_topology.LocalTopologyRank::new(
      ordinal=5,
      name="rank-five",
      total_memory_bytes=2147483648L,
      target~,
    ),
  ]
  let declaration = @device_topology.LocalTopologyDeclaration::new(ranks, [
    @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
  ])
  let devices = [
    @device_topology.LocalProbeDevice::new(
      ordinal=2,
      name="rank-two",
      total_memory_bytes=2147483648L,
      target~,
    ),
    @device_topology.LocalProbeDevice::new(
      ordinal=5,
      name="rank-five",
      total_memory_bytes=2147483648L,
      target~,
    ),
  ]
  let peers = [
    @device_topology.LocalPeerAccess::new(
      source_ordinal=2,
      destination_ordinal=5,
    ),
    @device_topology.LocalPeerAccess::new(
      source_ordinal=5,
      destination_ordinal=2,
    ),
  ]
  @device_topology.admit_local_topology(
    declaration,
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      devices,
      peers,
    ),
  )
}

///|
fn rank_local_runtime(
  topology : @device_topology.AdmittedLocalTopology,
) -> @tensor_parallel_admission.TensorParallelRuntimeAdmission raise {
  let declarations : Array[
    @tensor_parallel_admission.TensorParallelRankDeclaration,
  ] = []
  for rank in 0..<topology.world_size() {
    let device = topology.rank(rank).unwrap()
    declarations.push(
      @tensor_parallel_admission.TensorParallelRankDeclaration::new(
        rank~,
        world_size=2,
        process_ordinal=device.ordinal(),
        device_name=device.name(),
        target=device.target(),
        peer_rank_count=1,
        memory_ceiling_bytes=1073741824L,
        weight_reserved_bytes=4096L,
        activation_workspace_reserved_bytes=0L,
        kv_reserved_bytes=1048576L,
        collective_reserved_bytes=1048576L,
      ),
    )
  }
  @tensor_parallel_admission.admit(
    topology,
    @tensor_parallel_admission.TensorParallelRuntimeDeclaration::new(
      host_count=1,
      pipeline_stage_count=1,
      declarations,
    ),
  )
}

///|
fn rank_local_kv_plan(
  root : @approved_fs.ApprovedRoot,
  metadata : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
) -> @tensor_parallel_kv_plan.TensorParallelKvPlan raise {
  let topology = rank_local_topology()
  let devices : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan] = []
  for rank in 0..<2 {
    let inspection = @device_materialize.inspect_sharded_file(
      root,
      metadata,
      model_plan,
      @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
      rank~,
      world_size=2,
      alignment_bytes=64L,
      max_arena_bytes=4096L,
      limits=rank_local_file_limits(),
    )
    devices.push(
      @tensor_parallel_device_plan.build(
        model_plan,
        inspection.rank_plan(),
        inspection,
        @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
          topology,
          rank~,
        ),
        topology.target(),
      ),
    )
  }
  @tensor_parallel_kv_plan.admit(
    model_plan,
    @worker_protocol.ModelPlanGeneration::from_wire(7UL),
    rank_local_runtime(topology),
    devices,
    tokens_per_page=128,
    total_page_count=4,
  )
}
