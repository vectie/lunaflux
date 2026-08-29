///|
fn tpdp_append_tensor(
  entries : Array[@safetensors.RawTensorMetadata],
  name : String,
  shape : ArrayView[Int64],
  position : Ref[Int64],
) -> Unit {
  let mut elements = 1L
  for dimension in shape {
    elements *= dimension
  }
  let byte_length = elements * 2L
  entries.push(
    @safetensors.RawTensorMetadata::new(
      name~,
      dtype="BF16",
      shape~,
      data_start=position.val,
      data_end=position.val + byte_length,
    ),
  )
  position.val += byte_length
}

///|
fn tpdp_fixture(
  content? : String = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> (@plan.ModelPlan, @llama_weights.LlamaWeightBindings) raise {
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
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
    @spec.ContentDigest::from_sha256(content),
  )
  let entries : Array[@safetensors.RawTensorMetadata] = []
  let position = Ref(0L)
  tpdp_append_tensor(entries, "model.embed_tokens.weight", [8L, 4L], position)
  tpdp_append_tensor(
    entries,
    "model.layers.0.input_layernorm.weight",
    [4L],
    position,
  )
  for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
    tpdp_append_tensor(
      entries,
      "model.layers.0.self_attn.\{name}.weight",
      [4L, 4L],
      position,
    )
  }
  tpdp_append_tensor(
    entries,
    "model.layers.0.post_attention_layernorm.weight",
    [4L],
    position,
  )
  for name in ["gate_proj", "up_proj"] {
    tpdp_append_tensor(
      entries,
      "model.layers.0.mlp.\{name}.weight",
      [6L, 4L],
      position,
    )
  }
  tpdp_append_tensor(
    entries,
    "model.layers.0.mlp.down_proj.weight",
    [4L, 6L],
    position,
  )
  tpdp_append_tensor(entries, "model.norm.weight", [4L], position)
  tpdp_append_tensor(entries, "lm_head.weight", [8L, 4L], position)
  let bindings = @llama_weights.bind(
    metadata,
    @llama.build(metadata),
    @safetensors.validate_metadata(
      entries,
      encoded_header_bytes=1024,
      payload_bytes=position.val,
      limits=@safetensors.HeaderLimits::new(
        max_header_bytes=8192,
        max_tensor_count=32,
        max_rank=2,
        max_name_length=128,
        max_payload_bytes=4096L,
        max_materialized_tensor_bytes=4096L,
      ),
    ),
  )
  (@llama.build(metadata), bindings)
}

///|
fn tpdp_rank_plan(
  model_plan : @plan.ModelPlan,
  bindings : @llama_weights.LlamaWeightBindings,
  rank : Int,
  world_size : Int,
  alignment? : Int64 = 64L,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise {
  @llama_tensor_parallel.build_rank_plan(
    model_plan,
    bindings,
    rank~,
    world_size~,
    alignment_bytes=alignment,
    max_arena_bytes=4096L,
  )
}

///|
fn tpdp_target(
  minor? : Int = 0,
) -> @catalog.DeviceTarget raise @catalog.CatalogError {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn tpdp_two_rank_topology(
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
  let declaration = @device_topology.LocalTopologyDeclaration::new(ranks, [
    @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
  ])
  let devices = [
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
fn tpdp_stream_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn tpdp_weight_file() -> Bytes {
  let entries : Array[String] = [
    tpdp_stream_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    tpdp_stream_entry("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    tpdp_stream_entry(
      "model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104,
    ),
    tpdp_stream_entry(
      "model.layers.0.self_attn.k_proj.weight", "[4,4]", 104, 136,
    ),
    tpdp_stream_entry(
      "model.layers.0.self_attn.v_proj.weight", "[4,4]", 136, 168,
    ),
    tpdp_stream_entry(
      "model.layers.0.self_attn.o_proj.weight", "[4,4]", 168, 200,
    ),
    tpdp_stream_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 200, 208,
    ),
    tpdp_stream_entry("model.layers.0.mlp.gate_proj.weight", "[6,4]", 208, 256),
    tpdp_stream_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 256, 304),
    tpdp_stream_entry("model.layers.0.mlp.down_proj.weight", "[4,6]", 304, 352),
    tpdp_stream_entry("model.norm.weight", "[4]", 352, 360),
    tpdp_stream_entry("lm_head.weight", "[8,4]", 360, 424),
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
fn tpdp_file_model(file : BytesView) -> @spec.LlamaModelMetadata raise {
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
fn tpdp_file_limits() -> @device_materialize.DeviceWeightFileLimits raise {
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
