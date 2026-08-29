///|
fn selected_qwen3_config() -> String {
  let document =
    #|{"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":151643,"eos_token_id":151645,"head_dim":128,"hidden_act":"silu","hidden_size":1024,"initializer_range":0.02,"intermediate_size":3072,"max_position_embeddings":40960,"max_window_layers":28,"model_type":"qwen3","num_attention_heads":16,"num_hidden_layers":28,"num_key_value_heads":8,"rms_norm_eps":0.000001,"rope_scaling":null,"rope_theta":1000000,"sliding_window":null,"tie_word_embeddings":true,"torch_dtype":"bfloat16","transformers_version":"4.51.0","use_cache":true,"use_sliding_window":false,"vocab_size":151936}
  document
}

///|
fn parse_selected_qwen3(
  document : StringView,
) -> @qwen3_spec.Qwen3ModelMetadata raise @config_reader.ModelConfigReaderError {
  @config_reader.parse_qwen3_string(
    document,
    content_digest(),
    max_bytes=4096,
    max_depth=8,
  )
}

///|
test "pinned Qwen3 config retains explicit head geometry" {
  let text = selected_qwen3_config()
  let from_text = parse_selected_qwen3(text)
  let from_bytes = @config_reader.parse_qwen3_bytes(
    @utf8.encode(text),
    content_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(from_text, from_bytes)
  let spec = from_text.spec()
  assert_eq(spec.hidden_size(), 1024)
  assert_eq(spec.head_dimension(), 128)
  assert_eq(spec.query_projection_width(), 2048)
  assert_eq(spec.kv_projection_width(), 1024)
}

///|
test "Qwen3 parser rejects family substitution and unsafe semantic drift" {
  let source = selected_qwen3_config()
  let mutations : Array[(String, String, @config_reader.ModelConfigReaderError)] = [
    (
      "\"model_type\":\"qwen3\"",
      "\"model_type\":\"llama\"",
      InvalidValue(field=ModelType, issue=UnsupportedValue, index=0),
    ),
    (
      "\"head_dim\":128",
      "\"head_dim\":64",
      InvalidValue(field=HeadDimension, issue=UnsupportedValue, index=0),
    ),
    (
      "\"tie_word_embeddings\":true",
      "\"tie_word_embeddings\":false",
      InvalidValue(field=TieWordEmbeddings, issue=UnsupportedValue, index=0),
    ),
    (
      "\"use_sliding_window\":false",
      "\"use_sliding_window\":true",
      InvalidValue(field=UseSlidingWindow, issue=UnsupportedValue, index=0),
    ),
  ]
  for mutation in mutations {
    let (old, replacement, expected) = mutation
    try parse_selected_qwen3(source.replace(old~, new=replacement)) catch {
      error => assert_eq(error, expected)
    } noraise {
      _ => fail("hostile Qwen3 semantic mutation was admitted")
    }
  }
}

///|
test "Qwen3-only fields do not widen Llama admission" {
  let foreign = selected_config().replace(
    old="\"use_cache\":true",
    new="\"use_cache\":true,\"use_sliding_window\":false",
  )
  try parse_selected(foreign) catch {
    UnknownField(_) => ()
    _ => fail("unexpected Llama rejection for Qwen3-only field")
  } noraise {
    _ => fail("Llama parser admitted a Qwen3-only semantic field")
  }
}
