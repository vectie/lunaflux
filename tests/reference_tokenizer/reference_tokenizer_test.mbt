///|
let fixture_directory = "tests/reference_tokenizer"

///|
async fn read_fixture(name : String) -> Bytes {
  @fs.read_file("\{fixture_directory}/\{name}").binary()
}

///|
fn tokenizer_limits() -> @tokenizer.TokenizerLimits {
  {
    max_input_bytes: 4096,
    max_output_tokens: 4096,
    max_decoded_bytes: 4096,
    max_vocab_entries: 512,
    max_merge_rules: 64,
    max_token_bytes: 64,
    max_special_tokens: 8,
  }
}

///|
fn corpus_int(value : Json) -> Int raise {
  guard value is Number(number, repr=_) else {
    fail("corpus token ID must be a number")
  }
  let integer = number.to_int()
  if integer.to_double() != number {
    fail("corpus token ID must be exact")
  }
  integer
}

///|
fn corpus_ids(value : Json) -> Array[Int] raise {
  guard value is Array(values) else {
    fail("corpus token IDs must be an array")
  }
  values.map(corpus_int)
}

///|
async test "pinned upstream ByteLevel BPE observations match LunaFlux" {
  let tokenizer_document = read_fixture("tokenizer.json")
  assert_eq(tokenizer_document.length(), 2783)
  let digest_hex = @crypto.bytes_to_hex_string(
    @crypto.sha256(tokenizer_document),
  )
  assert_eq(
    digest_hex, "72e19c379865733c37776d8386a87707ac50703aed1633fabb5c708b1479e142",
  )
  let digest = @tokenizer.TokenizerDigest::parse(digest_hex)
  let spec = @json_reader.parse_bytes(
    tokenizer_document,
    digest~,
    limits=tokenizer_limits(),
    max_bytes=4096,
    max_depth=16,
  )
  assert_eq(spec.vocab_size(), 264)
  assert_eq(spec.merge_count(), 8)
  guard @json.parse(@utf8.decode(read_fixture("corpus.json"))) is Object(root) &&
    root["schema"] is String("lunaflux.reference-tokenizer.v1") &&
    root["upstream_revision"]
    is String("f383101a26663708484cac0727792aad74f78234") &&
    root["tokenizers_version"] is String("0.22.2") &&
    root["tokenizer_sha256"] is String(expected_digest) &&
    root["cases"] is Array(cases) else {
    fail("reference tokenizer corpus identity changed")
  }
  assert_eq(expected_digest, digest_hex)
  assert_eq(cases.length(), 5)
  for case in cases {
    guard case is Object(fields) &&
      fields["text"] is String(text) &&
      fields["decoded"] is String(decoded) else {
      fail("reference tokenizer case must contain text and decoded output")
    }
    let expected_ids = corpus_ids(fields["ids"])
    let encoded = spec.encode_text(
      text,
      special_policy=AsOrdinary,
      output_limit=4096,
      truncation=RejectOverflow,
    )
    assert_eq(encoded.token_ids, expected_ids)
    assert_eq(spec.decode_text(expected_ids, special_policy=Preserve), decoded)
    let expected_hex = match fields["utf8_hex"] {
      String(hex) => hex
      _ => fail("reference tokenizer case must contain UTF-8 bytes")
    }
    assert_eq(@crypto.bytes_to_hex_string(@utf8.encode(text)), expected_hex)
  }
}
