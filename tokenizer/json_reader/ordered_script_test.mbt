///|
test "ordered JSON executes identity-normalized added tokens without normalizing text" {
  guard @json.parse(tokenizer_document()) is Object(root) else {
    fail("fixture")
  }
  root["normalizer"] = Json::object({
    "type": Json::string("Sequence"),
    "normalizers": Json::array([]),
  })
  let patterns = [
    "\\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+", "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
  ]
  let splits = patterns.map(regex => {
    Json::object({
      "type": Json::string("Split"),
      "pattern": Json::object({ "Regex": Json::string(regex) }),
      "behavior": Json::string("Isolated"),
      "invert": Json::boolean(false),
    })
  })
  splits.push(root["pre_tokenizer"])
  root["pre_tokenizer"] = Json::object({
    "type": Json::string("Sequence"),
    "pretokenizers": Json::array(splits),
  })
  guard root["added_tokens"] is Array(added) else { fail("fixture") }
  added.push(
    Json::object({
      "id": Json::number(258),
      "content": Json::string("<｜User｜>"),
      "single_word": Json::boolean(false),
      "lstrip": Json::boolean(false),
      "rstrip": Json::boolean(false),
      "normalized": Json::boolean(true),
      "special": Json::boolean(false),
    }),
  )
  let spec = @json_reader.parse_string(
    Json::object(root).stringify(),
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
  let text = "<｜User｜>e\u0301hi"
  let encoded = spec.encode_text(
    text,
    special_policy=Allow,
    output_limit=32,
    truncation=RejectOverflow,
  )
  assert_eq(encoded.token_ids, [258, 101, 204, 129, 256])
  assert_eq(spec.decode_text(encoded.token_ids, special_policy=Skip), text)
}
