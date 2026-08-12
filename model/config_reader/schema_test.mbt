///|
test "unknown and executable metadata fail closed without retaining payloads" {
  let secret = "payload_secret_remote_type"
  let unknown = selected_config().replace(
    old="\"use_cache\":true",
    new="\"use_cache\":true,\"payload_secret_remote_type\":1",
  )
  try parse_selected(unknown) catch {
    UnknownField(index=28) as error =>
      assert_false(@debug.to_string(error).contains(secret))
    _ => fail("unexpected unknown-field error")
  } noraise {
    _ => fail("expected unknown field")
  }
  let executable : Array[(String, @config_reader.ExecutableField)] = [
    ("\"auto_map\":{}", AutoMap),
    ("\"trust_remote_code\":true", TrustRemoteCode),
    ("\"custom_pipelines\":{}", CustomPipelines),
  ]
  for entry in executable {
    let (field, expected) = entry
    let document = selected_config().replace(
      old="\"use_cache\":true",
      new="\"use_cache\":true," + field,
    )
    assert_rejected(document, ExecutableMetadata(expected))
  }
}

///|
test "missing mistyped fractional and overflowing dimensions are bounded" {
  assert_rejected(
    selected_config().replace(old="\"vocab_size\":32000,", new=""),
    MissingField(VocabularySize),
  )
  assert_rejected(
    selected_config().replace(
      old="\"vocab_size\":32000",
      new="\"vocab_size\":\"32000\"",
    ),
    InvalidValue(field=VocabularySize, issue=ExpectedInteger, index=0),
  )
  assert_rejected(
    selected_config().replace(
      old="\"vocab_size\":32000",
      new="\"vocab_size\":32000.5",
    ),
    InvalidValue(field=VocabularySize, issue=ExpectedInteger, index=0),
  )
  assert_rejected(
    selected_config().replace(
      old="\"vocab_size\":32000",
      new="\"vocab_size\":9223372036854775808",
    ),
    InvalidValue(field=VocabularySize, issue=ExpectedInteger, index=0),
  )
  assert_rejected(
    selected_config().replace(
      old="\"architectures\":[\"LlamaForCausalLM\"]",
      new="\"architectures\":\"LlamaForCausalLM\"",
    ),
    InvalidValue(field=Architectures, issue=ExpectedArray, index=0),
  )
}

///|
test "invalid head relationships reach the model specification validator" {
  let hidden_mismatch = selected_config().replace(
    old="\"hidden_size\":16",
    new="\"hidden_size\":15",
  )
  assert_rejected(
    hidden_mismatch,
    Validation(InvalidHeadRelationship(HiddenNotDivisibleByAttentionHeads)),
  )
  let kv_exceeds = selected_config().replace(
    old="\"num_key_value_heads\":2",
    new="\"num_key_value_heads\":5",
  )
  assert_rejected(
    kv_exceeds,
    Validation(InvalidHeadRelationship(KvHeadsExceedAttentionHeads)),
  )
  let non_grouped = selected_config()
    .replace(old="\"hidden_size\":16", new="\"hidden_size\":24")
    .replace(old="\"num_attention_heads\":4", new="\"num_attention_heads\":6")
    .replace(old="\"num_key_value_heads\":2", new="\"num_key_value_heads\":4")
  assert_rejected(
    non_grouped,
    Validation(InvalidHeadRelationship(AttentionHeadsNotDivisibleByKvHeads)),
  )
}

///|
test "dtype RoPE architecture and unsupported semantic variants fail closed" {
  assert_rejected(
    selected_config().replace(
      old="\"torch_dtype\":\"bfloat16\"",
      new="\"torch_dtype\":\"float16\"",
    ),
    InvalidValue(field=TorchDType, issue=UnsupportedValue, index=0),
  )
  for rope in ["0", "10000000000000"] {
    assert_rejected(
      selected_config().replace(
        old="\"rope_theta\":10000.0",
        new="\"rope_theta\":" + rope,
      ),
      Validation(InvalidRopeTheta),
    )
  }
  assert_rejected(
    selected_config().replace(
      old="\"architectures\":[\"LlamaForCausalLM\"]",
      new="\"architectures\":[\"MistralForCausalLM\"]",
    ),
    InvalidValue(field=Architectures, issue=UnsupportedValue, index=0),
  )
  assert_rejected(
    selected_config().replace(
      old="\"model_type\":\"llama\"",
      new="\"model_type\":\"mistral\"",
    ),
    InvalidValue(field=ModelType, issue=UnsupportedValue, index=0),
  )
  let unsupported : Array[
    (String, String, @config_reader.ModelConfigReaderError),
  ] = [
    (
      "\"tie_word_embeddings\":false",
      "\"tie_word_embeddings\":true",
      InvalidValue(field=TieWordEmbeddings, issue=UnsupportedValue, index=0),
    ),
    (
      "\"rope_scaling\":null",
      "\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":2.0}",
      InvalidValue(field=RopeScaling, issue=ExpectedNull, index=0),
    ),
    (
      "\"sliding_window\":null",
      "\"sliding_window\":4096",
      InvalidValue(field=SlidingWindow, issue=ExpectedNull, index=0),
    ),
    (
      "\"hidden_act\":\"silu\"",
      "\"hidden_act\":\"gelu\"",
      InvalidValue(field=HiddenActivation, issue=UnsupportedValue, index=0),
    ),
  ]
  for entry in unsupported {
    let (old, replacement, expected) = entry
    assert_rejected(selected_config().replace(old~, new=replacement), expected)
  }
}
