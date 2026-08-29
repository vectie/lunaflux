///|
test "selected dense Llama metadata is deterministic across string and bytes" {
  let document = selected_config()
  let from_string = parse_selected(document)
  let from_bytes = @config_reader.parse_bytes(
    @utf8.encode(document),
    content_digest(),
    max_bytes=4096,
    max_depth=8,
  )
  assert_eq(from_string, from_bytes)
  let model = from_string.spec()
  assert_eq(model.vocabulary_size(), 32000)
  assert_eq(model.hidden_size(), 16)
  assert_eq(model.intermediate_size(), 32)
  assert_eq(model.layer_count(), 2)
  assert_eq(model.attention_head_count(), 4)
  assert_eq(model.kv_head_count(), 2)
  assert_eq(model.head_dimension(), 4)
  assert_eq(model.context_length(), 128)
  assert_eq(model.rope_theta(), 10000.0)
  assert_eq(model.weight_dtype(), BFloat16)
  assert_eq(model.kv_dtype(), BFloat16)
  assert_eq(
    from_string.content_digest().as_hex(),
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
}

///|
test "missing KV head count uses the declared multi-head attention default" {
  let document = selected_config().replace(
    old="\"num_key_value_heads\":2,",
    new="",
  )
  let metadata = parse_selected(document)
  assert_eq(metadata.spec().attention_head_count(), 4)
  assert_eq(metadata.spec().kv_head_count(), 4)
}

///|
test "BF16 canonical artifact spelling is accepted" {
  let document = selected_config().replace(
    old="\"torch_dtype\":\"bfloat16\"",
    new="\"torch_dtype\":\"BF16\"",
  )
  assert_eq(parse_selected(document).spec().weight_dtype(), BFloat16)
}

///|
test "config admission rejects Llama layers above the plan ceiling" {
  let maximum = selected_config().replace(
    old="\"num_hidden_layers\":2",
    new="\"num_hidden_layers\":454",
  )
  assert_eq(parse_selected(maximum).spec().layer_count(), 454)
  let excessive = maximum.replace(
    old="\"num_hidden_layers\":454",
    new="\"num_hidden_layers\":455",
  )
  try parse_selected(excessive) catch {
    Validation(InvalidDimension(LayerCount)) => ()
    _ => fail("unexpected config layer-ceiling failure")
  } noraise {
    _ => fail("config admitted layers above the plan operation ceiling")
  }
}

///|
test "byte depth UTF-8 JSON and duplicate bounds fail before admission" {
  try
    @config_reader.parse_string(
      selected_config(),
      content_digest(),
      max_bytes=0,
      max_depth=8,
    )
  catch {
    InvalidByteLimit => ()
    _ => fail("unexpected byte limit error")
  } noraise {
    _ => fail("expected invalid byte limit")
  }
  try
    @config_reader.parse_string(
      selected_config(),
      content_digest(),
      max_bytes=4096,
      max_depth=0,
    )
  catch {
    InvalidDepthLimit => ()
    _ => fail("unexpected depth limit error")
  } noraise {
    _ => fail("expected invalid depth limit")
  }
  try parse_selected(selected_config(), max_bytes=32) catch {
    ByteLimitExceeded(limit=32) => ()
    _ => fail("unexpected byte bound error")
  } noraise {
    _ => fail("expected byte bound")
  }
  try
    @config_reader.parse_bytes(
      ([0xff] : Bytes),
      content_digest(),
      max_bytes=1,
      max_depth=8,
    )
  catch {
    InvalidUtf8 => ()
    _ => fail("unexpected UTF-8 error")
  } noraise {
    _ => fail("expected invalid UTF-8")
  }
  try
    parse_selected("{\"_name_or_path\":{\"a\":{\"b\":1}}}", max_depth=2)
  catch {
    NestingLimitExceeded(limit=2) => ()
    _ => fail("unexpected nesting error")
  } noraise {
    _ => fail("expected nesting bound")
  }
  assert_rejected(
    selected_config().replace(
      old="\"vocab_size\":32000",
      new="\"vocab_size\":32000,\"voc\\u0061b_size\":32000",
    ),
    DuplicateField(depth=1, index=3),
  )
  assert_rejected("{", MalformedJson)
  assert_rejected("[]", NonObjectRoot)
}
