///|
test "no-normalizer triplet pipeline carries offsets options without changing IDs" {
  let root = qwen3_test_root(regex=DIGIT_TRIPLET_SPLIT_REGEX)
  root["normalizer"] = Json::null()
  guard root["pre_tokenizer"] is Object(sequence) &&
    sequence["pretokenizers"] is Array([_, Object(bytes)]) else {
    fail("fixture")
  }
  bytes["trim_offsets"] = Json::boolean(true)
  for name in ["post_processor", "decoder"] {
    guard root[name] is Object(value) else { fail("fixture") }
    value["add_prefix_space"] = Json::boolean(true)
    value["use_regex"] = Json::boolean(true)
  }
  guard validate_root_semantics(root) is DigitTripletJson else {
    fail("profile")
  }
  guard root["pre_tokenizer"] is Object(sequence) &&
    sequence["pretokenizers"] is Array([Object(split), _]) else {
    fail("fixture")
  }
  split["pattern"] = Json::object({ "Regex": Json::string("\\p{N}{1,4}") })
  try validate_root_semantics(root) catch {
    UnsupportedSemantics(component=PreTokenizer, feature=RegexPreTokenization) =>
      ()
    _ => fail("unexpected regex error")
  } noraise {
    _ => fail("different regex silently admitted")
  }
}

///|
test "external added vocabulary retains Added versus Special and sorts by ID" {
  let root = qwen3_test_root(regex=DIGIT_TRIPLET_SPLIT_REGEX)
  root["normalizer"] = Json::null()
  let entries : Array[Json] = []
  for (id, content, special) in [(257, "<think>", false), (256, "<s>", true)] {
    entries.push(
      Json::object({
        "id": Json::number(id.to_double()),
        "content": Json::string(content),
        "single_word": Json::boolean(false),
        "lstrip": Json::boolean(false),
        "rstrip": Json::boolean(false),
        "normalized": Json::boolean(false),
        "special": Json::boolean(special),
      }),
    )
  }
  root["added_tokens"] = Json::array(entries)
  let limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 1024,
    max_output_tokens: 1024,
    max_decoded_bytes: 4096,
    max_vocab_entries: 512,
    max_merge_rules: 128,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
  let specials = parse_added_tokens(root, limits, DigitTripletJson)
  let vocab : Map[String, Json] = Map([])
  // The fixture's ByteLevel alphabet maps raw printable bytes identically.
  for value in 0..<256 {
    let scalar = if (value >= 0x21 && value <= 0x7e) ||
      (value >= 0xa1 && value <= 0xac) ||
      value >= 0xae {
      value
    } else if value <= 0x20 {
      0x100 + value
    } else if value <= 0xa0 {
      0x121 + value - 0x7f
    } else {
      0x143
    }
    vocab[scalar.to_char().unwrap().to_string()] = Json::number(
      value.to_double(),
    )
  }
  let parsed = parse_vocabulary(
    Json::object(vocab),
    specials,
    limits,
    DigitTripletJson,
  )
  assert_eq(parsed.entries.length(), 258)
  assert_true(parsed.entries[256].kind is Special)
  assert_true(parsed.entries[257].kind is Added)
  root["model"] = Json::object({
    "type": Json::string("BPE"),
    "dropout": Json::null(),
    "unk_token": Json::null(),
    "continuing_subword_prefix": Json::null(),
    "end_of_word_suffix": Json::null(),
    "fuse_unk": Json::boolean(false),
    "byte_fallback": Json::boolean(false),
    "ignore_merges": Json::boolean(true),
    "vocab": Json::object(vocab),
    "merges": Json::array([]),
  })
  let spec = parse_string(
    Json::object(root).stringify(),
    digest=@tokenizer.TokenizerDigest::parse("0".repeat(64)),
    limits~,
    max_bytes=65536,
    max_depth=16,
  )
  let encoded = spec.encode_text(
    "<s><think>123",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(encoded.token_ids, content="[256, 257, 49, 50, 51]")
  assert_eq(
    spec.decode_text(encoded.token_ids, special_policy=Skip),
    "<think>123",
  )
}
