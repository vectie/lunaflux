///|
fn wb_sharded_file() -> Bytes {
  let entries : Array[String] = [
    wb_stream_entry("model.embed_tokens.weight", "[8,4]", 0, 64),
    wb_stream_entry("model.layers.0.input_layernorm.weight", "[4]", 64, 72),
    wb_stream_entry("model.layers.0.self_attn.q_proj.weight", "[4,4]", 72, 104),
    wb_stream_entry("model.layers.0.self_attn.k_proj.weight", "[4,4]", 104, 136),
    wb_stream_entry("model.layers.0.self_attn.v_proj.weight", "[4,4]", 136, 168),
    wb_stream_entry("model.layers.0.self_attn.o_proj.weight", "[4,4]", 168, 200),
    wb_stream_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 200, 208,
    ),
    wb_stream_entry("model.layers.0.mlp.gate_proj.weight", "[6,4]", 208, 256),
    wb_stream_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 256, 304),
    wb_stream_entry("model.layers.0.mlp.down_proj.weight", "[4,6]", 304, 352),
    wb_stream_entry("model.norm.weight", "[4]", 352, 360),
    wb_stream_entry("lm_head.weight", "[8,4]", 360, 424),
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
fn wb_sharded_reordered_file() -> Bytes {
  let entries : Array[String] = [
    wb_stream_entry("model.embed_tokens.weight", "[8,4]", 360, 424),
    wb_stream_entry("model.layers.0.input_layernorm.weight", "[4]", 352, 360),
    wb_stream_entry("model.layers.0.self_attn.q_proj.weight", "[4,4]", 320, 352),
    wb_stream_entry("model.layers.0.self_attn.k_proj.weight", "[4,4]", 288, 320),
    wb_stream_entry("model.layers.0.self_attn.v_proj.weight", "[4,4]", 256, 288),
    wb_stream_entry("model.layers.0.self_attn.o_proj.weight", "[4,4]", 224, 256),
    wb_stream_entry(
      "model.layers.0.post_attention_layernorm.weight", "[4]", 216, 224,
    ),
    wb_stream_entry("model.layers.0.mlp.gate_proj.weight", "[6,4]", 168, 216),
    wb_stream_entry("model.layers.0.mlp.up_proj.weight", "[6,4]", 120, 168),
    wb_stream_entry("model.layers.0.mlp.down_proj.weight", "[4,6]", 72, 120),
    wb_stream_entry("model.norm.weight", "[4]", 64, 72),
    wb_stream_entry("lm_head.weight", "[8,4]", 0, 64),
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
    output.push(((index * 7 + 3) % 251).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn wb_sharded_model(file : BytesView) -> @spec.LlamaModelMetadata raise {
  let spec = @spec.LlamaModelSpec::new(
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
  )
  @spec.LlamaModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(file)),
    ),
  )
}

///|
fn wb_sharded_bindings(
  file : Bytes,
  model : @spec.LlamaModelMetadata,
) -> @llama_weights.LlamaWeightBindings raise {
  let header_bytes = @safetensors_reader.declared_header_bytes(
    file[:8],
    limits=wb_stream_limits().safetensors(),
  )
  let metadata = @safetensors_reader.read_header(
    file[:8 + header_bytes],
    payload_bytes=file.length().to_int64() - 8L - header_bytes.to_int64(),
    limits=wb_stream_limits().safetensors(),
  )
  let plan = @llama.build(model)
  @llama_weights.bind(model, plan, metadata)
}

///|
fn wb_sharded_rank_plan(
  file : Bytes,
  rank : Int,
) -> @tensor_parallel_plan.TensorParallelRankPlan raise {
  let model = wb_sharded_model(file)
  @llama_tensor_parallel.build_rank_plan(
    @llama.build(model),
    wb_sharded_bindings(file, model),
    rank~,
    world_size=2,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
}

///|
fn wb_prepare_sharded(
  file : Bytes,
  rank : Int,
) -> PreparedShardedStreamingLoad[SourceStamp] raise {
  let model = wb_sharded_model(file)
  prepare_sharded_streaming_load(
    model,
    @llama.build(model),
    rank,
    2,
    64L,
    4096L,
    wb_stamp(file),
    file.length().to_int64(),
    wb_stream_limits(transfer_chunk_bytes=17),
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
  )
}

///|
fn wb_copy_sharded(
  arena : FixedArray[Byte],
  source : FixedArray[Byte],
  source_offset : Int,
  destination_offset : Int64,
  byte_count : Int64,
) -> Unit {
  source.blit_to(
    arena,
    len=byte_count.to_int(),
    src_offset=source_offset,
    dst_offset=destination_offset.to_int(),
  )
}
