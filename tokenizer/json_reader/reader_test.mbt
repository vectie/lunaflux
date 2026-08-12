///|
fn test_limits() -> @tokenizer.TokenizerLimits {
  {
    max_input_bytes: 1024,
    max_output_tokens: 1024,
    max_decoded_bytes: 4096,
    max_vocab_entries: 512,
    max_merge_rules: 128,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
}

///|
fn test_digest() -> @tokenizer.TokenizerDigest {
  try! @tokenizer.TokenizerDigest::parse(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn bytelevel_scalar_for_byte(value : Int) -> Int {
  if (value >= 0x21 && value <= 0x7e) ||
    (value >= 0xa1 && value <= 0xac) ||
    (value >= 0xae && value <= 0xff) {
    value
  } else if value <= 0x20 {
    0x100 + value
  } else if value <= 0xa0 {
    0x121 + value - 0x7f
  } else {
    0x143
  }
}

///|
fn bytelevel_token(value : Int) -> String {
  bytelevel_scalar_for_byte(value).to_char().unwrap().to_string()
}

///|
fn tokenizer_document(
  pair_merge? : Bool = false,
  special? : Bool = true,
) -> String {
  let vocab : Map[String, Json] = Map([])
  for value in 0..<256 {
    vocab[bytelevel_token(value)] = Json::number(value.to_double())
  }
  vocab["hi"] = Json::number(256.0)
  let added : Array[Json] = []
  if special {
    vocab["<s>"] = Json::number(257.0)
    added.push(
      Json::object({
        "id": Json::number(257.0),
        "content": Json::string("<s>"),
        "single_word": Json::boolean(false),
        "lstrip": Json::boolean(false),
        "rstrip": Json::boolean(false),
        "normalized": Json::boolean(false),
        "special": Json::boolean(true),
      }),
    )
  }
  let merge = if pair_merge {
    Json::array([Json::string("h"), Json::string("i")])
  } else {
    Json::string("h i")
  }
  Json::object({
    "version": Json::string("1.0"),
    "truncation": Json::null(),
    "padding": Json::null(),
    "added_tokens": Json::array(added),
    "normalizer": Json::null(),
    "pre_tokenizer": Json::object({
      "type": Json::string("ByteLevel"),
      "add_prefix_space": Json::boolean(false),
      "trim_offsets": Json::boolean(true),
      "use_regex": Json::boolean(false),
    }),
    "post_processor": Json::null(),
    "decoder": Json::object({
      "type": Json::string("ByteLevel"),
      "add_prefix_space": Json::boolean(false),
      "trim_offsets": Json::boolean(true),
      "use_regex": Json::boolean(false),
    }),
    "model": Json::object({
      "type": Json::string("BPE"),
      "dropout": Json::null(),
      "unk_token": Json::null(),
      "continuing_subword_prefix": Json::null(),
      "end_of_word_suffix": Json::null(),
      "fuse_unk": Json::boolean(false),
      "byte_fallback": Json::boolean(false),
      "ignore_merges": Json::boolean(false),
      "vocab": Json::object(vocab),
      "merges": Json::array([merge]),
    }),
  }).stringify()
}

///|
fn parse_fixture(
  pair_merge? : Bool = false,
) -> @tokenizer.TokenizerSpec raise @json_reader.TokenizerJsonError {
  @json_reader.parse_string(
    tokenizer_document(pair_merge~),
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
}

///|
test "byte alphabet, Unicode byte mapping, merge, and special are exact" {
  let spec = parse_fixture()
  inspect(spec.vocab_size(), content="258")
  inspect(spec.merge_count(), content="1")
  let merged = spec.encode_text(
    "hi",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(merged.token_ids, content="[256]")
  let unicode = spec.encode_text(
    "é",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(unicode.token_ids, content="[195, 169]")
  inspect(
    spec.decode_text(unicode.token_ids, special_policy=Preserve),
    content="é",
  )
  let special = spec.encode_text(
    "a<s>b",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(special.token_ids, content="[97, 257, 98]")
}

///|
test "pair merge rows and byte input are deterministic" {
  let spec = parse_fixture(pair_merge=true)
  let input : Bytes = [0x00, 0x20, 0x7f, 0xad, 0xff]
  let encoded = spec.encode_bytes(
    input,
    special_policy=AsOrdinary,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(encoded.token_ids, content="[0, 32, 127, 173, 255]")
  assert_eq(
    spec.decode_bytes(encoded.token_ids, special_policy=Preserve),
    input,
  )
}

///|
test "byte, UTF-8, depth, duplicate, and schema errors stay typed" {
  try
    @json_reader.parse_string(
      "{}",
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=1,
      max_depth=4,
    )
  catch {
    ByteLimitExceeded(limit=1) => ()
    _ => fail("expected byte limit")
  } noraise {
    _ => fail("expected failure")
  }
  try
    @json_reader.parse_bytes(
      [0xff],
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=8,
      max_depth=4,
    )
  catch {
    InvalidUtf8 => ()
    _ => fail("expected invalid UTF-8")
  } noraise {
    _ => fail("expected failure")
  }
  try
    @json_reader.parse_string(
      "{\"a\":{\"b\":{}}}",
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=64,
      max_depth=2,
    )
  catch {
    NestingLimitExceeded(limit=2) => ()
    _ => fail("expected depth limit")
  } noraise {
    _ => fail("expected failure")
  }
  try
    @json_reader.parse_string(
      "{\"version\":\"1.0\",\"version\":\"1.0\"}",
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=128,
      max_depth=4,
    )
  catch {
    DuplicateField(depth=1, index=1) => ()
    _ => fail("expected duplicate")
  } noraise {
    _ => fail("expected failure")
  }
}

///|
fn replace_once(document : String, before : String, after : String) -> String {
  document.replace(old=before, new=after)
}

///|
test "unsupported tokenizer behavior is never silently admitted" {
  let base = tokenizer_document()
  let regex = replace_once(base, "\"use_regex\":false", "\"use_regex\":true")
  try
    @json_reader.parse_string(
      regex,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnsupportedSemantics(component=PreTokenizer, feature=RegexPreTokenization) =>
      ()
    _ => fail("expected regex incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
  let fallback = replace_once(
    base, "\"byte_fallback\":false", "\"byte_fallback\":true",
  )
  try
    @json_reader.parse_string(
      fallback,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnsupportedSemantics(component=Model, feature=ByteFallbackBehavior) => ()
    _ => fail("expected fallback incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
  let normalized = replace_once(
    base, "\"normalizer\":null", "\"normalizer\":{\"type\":\"NFC\"}",
  )
  try
    @json_reader.parse_string(
      normalized,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnsupportedSemantics(component=Normalizer, feature=NormalizationBehavior) =>
      ()
    _ => fail("expected normalization incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
  let decoder = replace_once(
    base, "\"decoder\":{\"type\":\"ByteLevel\"", "\"decoder\":{\"type\":\"WordPiece\"",
  )
  try
    @json_reader.parse_string(
      decoder,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnsupportedSemantics(component=Decoder, feature=DecoderKind) => ()
    _ => fail("expected decoder incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
  let model = replace_once(base, "\"type\":\"BPE\"", "\"type\":\"WordPiece\"")
  try
    @json_reader.parse_string(
      model,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnsupportedSemantics(component=Model, feature=ModelKind) => ()
    _ => fail("expected model incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
  let malformed_merge = replace_once(base, "\"h i\"", "\"missing i\"")
  try
    @json_reader.parse_string(
      malformed_merge,
      digest=test_digest(),
      limits=test_limits(),
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    UnknownMergePiece(index=0, part=Left) => ()
    _ => fail("expected merge incompatibility")
  } noraise {
    _ => fail("expected failure")
  }
}

///|
test "vocabulary, merge, and special limits apply before core construction" {
  let tiny_vocab = { ..test_limits(), max_vocab_entries: 255 }
  try
    @json_reader.parse_string(
      tokenizer_document(),
      digest=test_digest(),
      limits=tiny_vocab,
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    VocabularyLimitExceeded(limit=255) => ()
    _ => fail("expected vocabulary limit")
  } noraise {
    _ => fail("expected failure")
  }
  let no_merges = { ..test_limits(), max_merge_rules: 0 }
  try
    @json_reader.parse_string(
      tokenizer_document(),
      digest=test_digest(),
      limits=no_merges,
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    MergeLimitExceeded(limit=0) => ()
    _ => fail("expected merge limit")
  } noraise {
    _ => fail("expected failure")
  }
  let no_specials = { ..test_limits(), max_special_tokens: 0 }
  try
    @json_reader.parse_string(
      tokenizer_document(),
      digest=test_digest(),
      limits=no_specials,
      max_bytes=65536,
      max_depth=16,
    )
  catch {
    AddedTokenLimitExceeded(limit=0) => ()
    _ => fail("expected special limit")
  } noraise {
    _ => fail("expected failure")
  }
}
