///|
fn fixture_weight_entry(
  name : String,
  shape : String,
  start : Int,
  end : Int,
) -> String {
  "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}"
}

///|
fn fixture_weight_file() -> Bytes {
  let entries : Array[String] = [
    fixture_weight_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    fixture_weight_entry("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    fixture_weight_entry(
      "model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104,
    ),
    fixture_weight_entry(
      "model.layers.0.self_attn.k_proj.weight", "[2,4]", 104, 120,
    ),
    fixture_weight_entry(
      "model.layers.0.self_attn.v_proj.weight", "[2,4]", 120, 136,
    ),
    fixture_weight_entry(
      "model.layers.0.self_attn.o_proj.weight", "[4,4]", 136, 168,
    ),
    fixture_weight_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 168, 176,
    ),
    fixture_weight_entry(
      "model.layers.0.mlp.gate_proj.weight", "[6,4]", 176, 224,
    ),
    fixture_weight_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 224, 272),
    fixture_weight_entry(
      "model.layers.0.mlp.down_proj.weight", "[4,6]", 272, 320,
    ),
    fixture_weight_entry("model.norm.weight", "[4]", 320, 328),
    fixture_weight_entry("lm_head.weight", "[8,4]", 328, 392),
  ]
  let header = @utf8.encode("{" + entries.join(",") + "}")
  let output : Array[Byte] = []
  for shift in 0..<8 {
    output.push((header.length().to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    output.push(byte)
  }
  for index in 0..<392 {
    output.push((index % 251).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn fixture_model(file : BytesView) -> @spec.LlamaModelMetadata raise {
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
      @crypto.bytes_to_hex_string(@crypto.sha256(file)),
    ),
  )
}

///|
fn fixture_manifest(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  module_bytes : BytesView,
) -> String {
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes))
  let kinds : Array[@plan.OperationKind] = []
  let mut entries = ""
  let mut operations = ""
  for index, operation in plan.operations() {
    if index > 0 {
      operations += ","
    }
    let mut kind_index = 0
    for candidate_index, candidate in kinds {
      if candidate == operation.kind() {
        kind_index = candidate_index + 1
      }
    }
    if kind_index == 0 {
      kinds.push(operation.kind())
      kind_index = kinds.length()
      if kind_index > 1 {
        entries += ","
      }
      entries += "{\"family_id\":" +
        kind_index.to_string() +
        ",\"entry_point_id\":" +
        kind_index.to_string() +
        ",\"symbol\":\"kernel_" +
        kind_index.to_string() +
        "\"}"
    }
    let workspace = operation.workspace_bytes()
    operations += "{\"operation_id\":" +
      operation.id().as_int().to_string() +
      ",\"module_index\":0,\"family_id\":" +
      kind_index.to_string() +
      ",\"entry_point_id\":" +
      kind_index.to_string() +
      ",\"workspace_bytes\":" +
      workspace.to_string() +
      ",\"workspace_alignment\":" +
      (if workspace > 0L { "16" } else { "1" }) +
      ",\"dimensions\":{\"grid_x\":1,\"grid_y\":1,\"grid_z\":1," +
      "\"block_x\":32,\"block_y\":1,\"block_z\":1," +
      "\"shared_memory_bytes\":0}}"
  }
  "{\"schema_version\":2,\"model_identity\":{\"content_sha256\":\"" +
  plan.identity().content().as_hex() +
  "\",\"plan_sha256\":\"" +
  plan.identity().plan().as_hex() +
  "\"},\"device_target\":{\"compute_major\":" +
  target.compute_major().to_string() +
  ",\"compute_minor\":" +
  target.compute_minor().to_string() +
  ",\"supports_bf16\":true,\"supports_cublas_lt\":false}," +
  "\"catalog_version\":3,\"profile\":{\"id\":1},\"modules\":[{" +
  "\"sha256\":\"" +
  digest +
  "\",\"path\":\"kernel.cubin\",\"entry_points\":[" +
  entries +
  "]}],\"operations\":[" +
  operations +
  "]}"
}
