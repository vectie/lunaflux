///|
fn group_fixture_tensor(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn group_fixture_weight_file() -> Bytes {
  let entries : Array[String] = [
    group_fixture_tensor("model.embed_tokens.weight", "[8,4]", 0, 64),
    group_fixture_tensor("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    group_fixture_tensor(
      "model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104,
    ),
    group_fixture_tensor(
      "model.layers.0.self_attn.k_proj.weight", "[4,4]", 104, 136,
    ),
    group_fixture_tensor(
      "model.layers.0.self_attn.v_proj.weight", "[4,4]", 136, 168,
    ),
    group_fixture_tensor(
      "model.layers.0.self_attn.o_proj.weight", "[4,4]", 168, 200,
    ),
    group_fixture_tensor(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 200, 208,
    ),
    group_fixture_tensor(
      "model.layers.0.mlp.gate_proj.weight", "[6,4]", 208, 256,
    ),
    group_fixture_tensor("model.layers.0.mlp.up_proj.weight", "[6,4]", 256, 304),
    group_fixture_tensor(
      "model.layers.0.mlp.down_proj.weight", "[4,6]", 304, 352,
    ),
    group_fixture_tensor("model.norm.weight", "[4]", 352, 360),
    group_fixture_tensor("lm_head.weight", "[8,4]", 360, 424),
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
fn group_fixture_metadata(
  weight_file : BytesView,
) -> @spec.LlamaModelMetadata raise {
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
      @crypto.bytes_to_hex_string(@crypto.sha256(weight_file)),
    ),
  )
}

///|
fn group_fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn group_fixture_topology(
  target : @catalog.DeviceTarget,
) -> (
  @device_topology.LocalTopologyDeclaration,
  @device_topology.AdmittedLocalTopology,
) raise {
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
  let projection = @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
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
  )
  (declaration, @device_topology.admit_local_topology(declaration, projection))
}
