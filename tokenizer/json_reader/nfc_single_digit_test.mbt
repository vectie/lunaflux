///|
test "NFC byte BPE accepts model-owned ASCII marker IDs and offset-only options" {
  guard @json.parse(tokenizer_document()) is Object(root) else {
    fail("fixture")
  }
  root["normalizer"] = @json.parse("{\"type\":\"NFC\"}")
  let pre : Json = {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        },
        "behavior": "Isolated",
        "invert": false,
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false,
      },
    ],
  }
  root["pre_tokenizer"] = pre
  root["decoder"] = @json.parse(
    "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}",
  )
  root["post_processor"] = @json.parse(
    "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}",
  )
  guard root["added_tokens"] is Array(tokens) else { fail("fixture") }
  let marker : Json = {
    "id": 258,
    "content": "<image>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": true,
  }
  tokens.push(marker)
  let spec = @json_reader.parse_nfc_single_digit_bytes(
    @utf8.encode(Json::object(root).stringify())[:],
    digest=test_digest(),
    limits=test_limits(),
    max_bytes=65536,
    max_depth=16,
  )
  let encoded = spec.encode_text(
    "<image>hie\u0301",
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  assert_eq(encoded.token_ids[0], 258)
  assert_eq(
    spec.decode_text(encoded.token_ids, special_policy=Preserve),
    "<image>hié",
  )
}
