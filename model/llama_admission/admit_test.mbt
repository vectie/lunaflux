///|
fn config_bytes() -> Bytes {
  let config =
    #|{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":3,"hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":4,"rope_theta":10000.0,"torch_dtype":"bfloat16","tie_word_embeddings":false,"rope_scaling":null,"sliding_window":null,"hidden_act":"silu","rms_norm_eps":0.000001,"attention_bias":false,"attention_dropout":0.0,"mlp_bias":false,"head_dim":2,"pretraining_tp":1}
  @utf8.encode(config)
}

///|
fn append_entry(
  text : StringBuilder,
  first : Bool,
  name : String,
  shape : String,
  bytes : Int,
  offset : Ref[Int],
) -> Unit {
  if !first {
    text.write_string(",")
  }
  let start = offset.val
  let end = start + bytes
  text.write_string(
    "\"\{name}\":{\"dtype\":\"BF16\",\"shape\":\{shape},\"data_offsets\":[\{start},\{end}]}",
  )
  offset.val = end
}

///|
fn safetensors_bytes() -> Bytes {
  let header = StringBuilder()
  header.write_string("{")
  let offset = Ref(0)
  let names = [
    ("model.embed_tokens.weight", "[3,2]", 12),
    ("model.layers.0.input_layernorm.weight", "[2]", 4),
    ("model.layers.0.self_attn.q_proj.weight", "[2,2]", 8),
    ("model.layers.0.self_attn.k_proj.weight", "[2,2]", 8),
    ("model.layers.0.self_attn.v_proj.weight", "[2,2]", 8),
    ("model.layers.0.self_attn.o_proj.weight", "[2,2]", 8),
    ("model.layers.0.post_attention_layernorm.weight", "[2]", 4),
    ("model.layers.0.mlp.gate_proj.weight", "[2,2]", 8),
    ("model.layers.0.mlp.up_proj.weight", "[2,2]", 8),
    ("model.layers.0.mlp.down_proj.weight", "[2,2]", 8),
    ("model.norm.weight", "[2]", 4),
    ("lm_head.weight", "[3,2]", 12),
  ]
  for index, entry in names {
    append_entry(header, index == 0, entry.0, entry.1, entry.2, offset)
  }
  header.write_string("}")
  let header_bytes = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  let encoded_length = header_bytes.length().to_uint64().to_le_bytes()
  for byte in encoded_length {
    output.push(byte)
  }
  for byte in header_bytes {
    output.push(byte)
  }
  for _ in 0..<offset.val {
    output.push(0)
  }
  Bytes::from_array(output)
}

///|
fn content_digest() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn header_limits() -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=12,
    max_rank=2,
    max_name_length=96,
    max_payload_bytes=256L,
    max_materialized_tensor_bytes=256L,
  )
}

///|
test "selected model admission validates all pre-allocation stages" {
  let admitted = @llama_admission.admit(
    config_bytes(),
    safetensors_bytes(),
    content_digest(),
    config_max_bytes=2048,
    config_max_depth=8,
    header_limits=header_limits(),
  )
  assert_eq(admitted.safetensors().tensor_count(), 12)
  assert_eq(admitted.bindings().length(), 12)
  assert_eq(admitted.plan().operations().length(), 12)
  assert_eq(admitted.metadata().content_digest(), content_digest())
  assert_eq(admitted.plan().identity(), admitted.bindings().identity())
  assert_eq(
    @llama_admission.complete(admitted.metadata(), admitted.safetensors()),
    admitted,
  )
}

///|
test "configuration and weight failures retain their admission stage" {
  try
    @llama_admission.admit(
      b"{}",
      safetensors_bytes(),
      content_digest(),
      config_max_bytes=2048,
      config_max_depth=8,
      header_limits=header_limits(),
    )
  catch {
    @llama_admission.LlamaAdmissionError::Configuration(_) => ()
    _ => fail("expected configuration-stage rejection")
  } noraise {
    _ => fail("expected invalid configuration")
  }
  try
    @llama_admission.admit(
      config_bytes(),
      b"short",
      content_digest(),
      config_max_bytes=2048,
      config_max_depth=8,
      header_limits=header_limits(),
    )
  catch {
    @llama_admission.LlamaAdmissionError::Safetensors(MissingHeaderLength) => ()
    _ => fail("expected safetensors-stage rejection")
  } noraise {
    _ => fail("expected invalid safetensors")
  }
}
