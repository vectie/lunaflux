///|
fn content_digest() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
}

///|
fn plan_digest() -> @spec.PlanDigest {
  try! @spec.PlanDigest::from_sha256(
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  )
}

///|
fn selected_config() -> String {
  let document =
    #|{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":32000,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":128,"rope_theta":10000.0,"torch_dtype":"bfloat16","tie_word_embeddings":false,"rope_scaling":null,"sliding_window":null,"hidden_act":"silu","rms_norm_eps":0.000001,"attention_bias":false,"attention_dropout":0.0,"mlp_bias":false,"head_dim":4,"pretraining_tp":1,"_name_or_path":"selected","transformers_version":"4.43.0","bos_token_id":1,"eos_token_id":2,"pad_token_id":null,"initializer_range":0.02,"use_cache":true}
  document
}

///|
fn parse_selected(
  document : StringView,
  max_bytes? : Int = 4096,
  max_depth? : Int = 8,
) -> @spec.LlamaModelMetadata raise @config_reader.ModelConfigReaderError {
  @config_reader.parse_string(
    document,
    content_digest(),
    plan_digest(),
    max_bytes~,
    max_depth~,
  )
}

///|
fn assert_rejected(
  document : StringView,
  expected : @config_reader.ModelConfigReaderError,
) -> Unit raise {
  try parse_selected(document) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("expected selected-model configuration rejection")
  }
}
