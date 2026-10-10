///|
fn ordered_root() -> Map[String, Json] {
  let root = qwen3_test_root()
  root["normalizer"] = Json::object({
    "type": Json::string("Sequence"),
    "normalizers": Json::array([]),
  })
  let splits = [ORDERED_NUMBER_REGEX, ORDERED_SCRIPT_REGEX, ORDERED_WORD_REGEX].map(regex => {
      Json::object({
        "type": Json::string("Split"),
        "pattern": Json::object({ "Regex": Json::string(regex) }),
        "behavior": Json::string("Isolated"),
        "invert": Json::boolean(false),
      })
    },
  )
  splits.push(
    Json::object({
      "type": Json::string("ByteLevel"),
      "add_prefix_space": Json::boolean(false),
      "trim_offsets": Json::boolean(true),
      "use_regex": Json::boolean(false),
    }),
  )
  root["pre_tokenizer"] = Json::object({
    "type": Json::string("Sequence"),
    "pretokenizers": Json::array(splits),
  })
  root
}

///|
test "identity normalization preserves normalized added-token matching" {
  let root = ordered_root()
  let entry : Map[String, Json] = {
    "id": Json::number(256),
    "content": Json::string("<｜User｜>"),
    "single_word": Json::boolean(false),
    "lstrip": Json::boolean(false),
    "rstrip": Json::boolean(false),
    "normalized": Json::boolean(true),
    "special": Json::boolean(false),
  }
  root["added_tokens"] = Json::array([Json::object(entry)])
  let added = parse_added_tokens(root, qwen3_test_limits(), OrderedScriptJson)
  assert_eq(added[256].content, "<｜User｜>")
  assert_false(added[256].special)
  for field in ["single_word", "lstrip", "rstrip"] {
    entry[field] = Json::boolean(true)
    try parse_added_tokens(root, qwen3_test_limits(), OrderedScriptJson) catch {
      UnsupportedSemantics(component=AddedToken, feature=AddedTokenMatching) =>
        ()
      _ => fail("wrong added-token semantics error")
    } noraise {
      _ => fail("unsupported matching silently skipped")
    }
    entry[field] = Json::boolean(false)
  }
}

///|
test "ordered split pipeline requires identity normalization and semantic split order" {
  guard validate_root_semantics(ordered_root()) is OrderedScriptJson else {
    fail("wrong pipeline")
  }
  let root = ordered_root()
  guard root["pre_tokenizer"] is { "pretokenizers": Array(splits), .. } else {
    fail("fixture")
  }
  let first = splits[0]
  splits[0] = splits[1]
  splits[1] = first
  try validate_root_semantics(root) catch {
    UnsupportedSemantics(component=PreTokenizer, feature=RegexPreTokenization) =>
      ()
    _ => fail("wrong ordered split error")
  } noraise {
    _ => fail("changed split order accepted")
  }
  let root = ordered_root()
  root["normalizer"] = Json::object({
    "type": Json::string("Sequence"),
    "normalizers": Json::array([Json::object({ "type": Json::string("NFC") })]),
  })
  try validate_root_semantics(root) catch {
    UnsupportedSemantics(component=Normalizer, feature=NormalizationBehavior) =>
      ()
    _ => fail("wrong normalizer error")
  } noraise {
    _ => fail("unsupported normalization silently skipped")
  }
}
