///|
fn qwen3_test_root(
  regex? : String = QWEN3_SPLIT_REGEX,
  normalizer_extra? : Bool = false,
) -> Map[String, Json] {
  let normalizer : Map[String, Json] = { "type": Json::string("NFC") }
  if normalizer_extra {
    normalizer["form"] = Json::string("NFC")
  }
  {
    "version": Json::string("1.0"),
    "truncation": Json::null(),
    "padding": Json::null(),
    "added_tokens": Json::array([]),
    "normalizer": Json::object(normalizer),
    "pre_tokenizer": Json::object({
      "type": Json::string("Sequence"),
      "pretokenizers": Json::array([
        Json::object({
          "type": Json::string("Split"),
          "pattern": Json::object({ "Regex": Json::string(regex) }),
          "behavior": Json::string("Isolated"),
          "invert": Json::boolean(false),
        }),
        Json::object({
          "type": Json::string("ByteLevel"),
          "add_prefix_space": Json::boolean(false),
          "trim_offsets": Json::boolean(false),
          "use_regex": Json::boolean(false),
        }),
      ]),
    }),
    "post_processor": Json::object({
      "type": Json::string("ByteLevel"),
      "add_prefix_space": Json::boolean(false),
      "trim_offsets": Json::boolean(false),
      "use_regex": Json::boolean(false),
    }),
    "decoder": Json::object({
      "type": Json::string("ByteLevel"),
      "add_prefix_space": Json::boolean(false),
      "trim_offsets": Json::boolean(false),
      "use_regex": Json::boolean(false),
    }),
    "model": Json::null(),
  }
}

///|
fn qwen3_added_tokens() -> Array[Json] {
  let entries : Array[Json] = []
  for id in QWEN3_FIRST_ADDED_TOKEN_ID..<=QWEN3_LAST_ADDED_TOKEN_ID {
    entries.push(
      Json::object({
        "id": Json::number(id.to_double()),
        "content": Json::string(qwen3_added_token_content(id).unwrap()),
        "single_word": Json::boolean(false),
        "lstrip": Json::boolean(false),
        "rstrip": Json::boolean(false),
        "normalized": Json::boolean(false),
        "special": Json::boolean(id <= 151656),
      }),
    )
  }
  entries
}

///|
fn qwen3_test_limits() -> @tokenizer.TokenizerLimits {
  {
    max_input_bytes: 1024,
    max_output_tokens: 1024,
    max_decoded_bytes: 4096,
    max_vocab_entries: 200000,
    max_merge_rules: 200000,
    max_token_bytes: 256,
    max_special_tokens: 64,
  }
}

///|
test "exact Qwen3 NFC Split ByteLevel pipeline and added tokens are admitted" {
  let root = qwen3_test_root()
  guard validate_root_semantics(root) is Qwen3Json else {
    fail("expected Qwen3 profile")
  }
  root["added_tokens"] = Json::array(qwen3_added_tokens())
  let added = parse_added_tokens(root, qwen3_test_limits(), Qwen3Json)
  assert_eq(added.length(), 26)
  assert_true(added[151643].special)
  assert_false(added[151668].special)
}

///|
test "nearby Qwen3 regex normalizer and added-token schemas fail closed" {
  try validate_root_semantics(qwen3_test_root(regex="\\s+")) catch {
    UnsupportedSemantics(component=PreTokenizer, feature=Qwen3PreTokenization) =>
      ()
    _ => fail("unexpected regex error")
  } noraise {
    _ => fail("changed regex was admitted")
  }
  try validate_root_semantics(qwen3_test_root(normalizer_extra=true)) catch {
    UnknownField(component=Normalizer, index=1) => ()
    _ => fail("unexpected normalizer error")
  } noraise {
    _ => fail("extended normalizer was admitted")
  }
  let root = qwen3_test_root()
  let entries = qwen3_added_tokens()
  guard entries[25] is Object(last) else { fail("fixture invariant") }
  last["special"] = Json::boolean(true)
  root["added_tokens"] = Json::array(entries)
  try parse_added_tokens(root, qwen3_test_limits(), Qwen3Json) catch {
    UnsupportedSemantics(component=AddedToken, feature=Qwen3AddedTokens) => ()
    _ => fail("unexpected added-token error")
  } noraise {
    _ => fail("changed added-token schema was admitted")
  }
}
