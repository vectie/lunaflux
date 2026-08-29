///|
test "tokenizer file source evidence rejects every binding substitution" {
  let digest = try! @tokenizer.TokenizerDigest::parse(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let other = try! @tokenizer.TokenizerDigest::parse(
    "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let file_limits = try! TokenizerFileLimits::new(
    max_file_bytes=4096,
    max_json_depth=16,
  )
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 64,
    max_output_tokens: 64,
    max_decoded_bytes: 64,
    max_vocab_entries: 512,
    max_merge_rules: 512,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
  assert_true(
    tokenizer_file_source_matches(
      "/approved/model", "tokenizer.json", digest, file_limits, tokenizer_limits,
      "/approved/model", "tokenizer.json", digest, file_limits, tokenizer_limits,
    ),
  )
  assert_false(
    tokenizer_file_source_matches(
      "/approved/model", "tokenizer.json", digest, file_limits, tokenizer_limits,
      "/approved/other", "tokenizer.json", digest, file_limits, tokenizer_limits,
    ),
  )
  assert_false(
    tokenizer_file_source_matches(
      "/approved/model", "tokenizer.json", digest, file_limits, tokenizer_limits,
      "/approved/model", "other.json", digest, file_limits, tokenizer_limits,
    ),
  )
  assert_false(
    tokenizer_file_source_matches(
      "/approved/model", "tokenizer.json", digest, file_limits, tokenizer_limits,
      "/approved/model", "tokenizer.json", other, file_limits, tokenizer_limits,
    ),
  )
}
