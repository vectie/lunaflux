///|
test "omitted BPE ignore_merges uses false rather than exact-token lookup" {
  let document = tokenizer_document(empty_merges=true)
  guard @json.parse(document) is Object(root) && root["model"] is Object(model) else {
    fail("fixture")
  }
  model.remove("ignore_merges")
  let omitted = @json_reader.parse_string(
    Json::object(root).stringify(),
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
  let encoded = omitted.encode_text(
    "hi",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(encoded.token_ids, content="[104, 105]")
  let explicit = @json_reader.parse_string(
    document,
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
  assert_eq(
    encoded.token_ids,
    explicit.encode_text(
      "hi",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
  )
}

///|
test "present BPE ignore_merges rejects invalid field types" {
  for value in [Json::null(), Json::string("false"), Json::number(0.0)] {
    guard @json.parse(tokenizer_document()) is Object(root) &&
      root["model"] is Object(model) else {
      fail("fixture")
    }
    model["ignore_merges"] = value
    try
      @json_reader.parse_string(
        Json::object(root).stringify(),
        digest=test_digest(),
        limits=test_limits(),
        max_bytes=65536,
        max_depth=16,
      )
    catch {
      @json_reader.InvalidField(component=Model, field=IgnoreMerges, index=0) =>
        ()
      _ => fail("unexpected optional-field error")
    } noraise {
      _ => fail("invalid ignore_merges type was coerced")
    }
  }
}
