///|
const TP_WORLD_SIZE : Int = 2

///|
const TP_MODEL_GENERATION : UInt64 = 7UL

///|
const TP_GROUP_GENERATION : UInt64 = 9UL

///|
const TP_RUNTIME_VERSION : Int = 21403

///|
const TP_TOKENS_PER_PAGE : Int = 64

///|
const TP_TOTAL_PAGE_COUNT : Int = 8

///|
const TP_MAX_ROWS : Int = 2

///|
const TP_MAX_TOKENS : Int = 16

///|
const TP_MAX_PAGES : Int = 2

///|
fn weight_entry(
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
    weight_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    weight_entry("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    weight_entry("model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104),
    weight_entry("model.layers.0.self_attn.k_proj.weight", "[4,4]", 104, 136),
    weight_entry("model.layers.0.self_attn.v_proj.weight", "[4,4]", 136, 168),
    weight_entry("model.layers.0.self_attn.o_proj.weight", "[4,4]", 168, 200),
    weight_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 200, 208,
    ),
    weight_entry("model.layers.0.mlp.gate_proj.weight", "[6,4]", 208, 256),
    weight_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 256, 304),
    weight_entry("model.layers.0.mlp.down_proj.weight", "[4,6]", 304, 352),
    weight_entry("model.norm.weight", "[4]", 352, 360),
    weight_entry("lm_head.weight", "[8,4]", 360, 424),
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
fn fixture_metadata(file : BytesView) -> @spec.LlamaModelMetadata raise {
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
fn fixture_weight_limits() -> @device_materialize.DeviceWeightFileLimits raise {
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
fn fixture_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn fixture_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=TP_MAX_ROWS,
    max_decode_rows=TP_MAX_ROWS,
    max_plan_rows=TP_MAX_ROWS,
    max_plan_tokens=TP_MAX_TOKENS,
    max_plan_pages=TP_MAX_PAGES,
    max_capabilities=32,
    max_completion_slots=TP_MAX_ROWS,
    max_sequence_tokens=16,
    max_token_id=7,
  )
}

///|
fn fixture_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=7,
    max_stop_token_ids=2,
    max_stop_strings=1,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=100UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}
