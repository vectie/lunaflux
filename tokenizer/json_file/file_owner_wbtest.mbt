///|
test "tokenizer source comparison never accepts a file-limit substitution" {
  let digest = try! @tokenizer.TokenizerDigest::parse(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
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
  let exact = try! TokenizerFileLimits::new(
    max_file_bytes=4096,
    max_json_depth=16,
  )
  let wider = try! TokenizerFileLimits::new(
    max_file_bytes=8192,
    max_json_depth=16,
  )
  assert_false(
    tokenizer_file_source_matches(
      "/approved/model", "tokenizer.json", digest, exact, tokenizer_limits, "/approved/model",
      "tokenizer.json", digest, wider, tokenizer_limits,
    ),
  )
}

///|
test "tokenizer terminal close failure prevents snapshot publication" {
  try
    finish_tokenizer_file(Ok(b"exact"), () => raise FileOperationFailed(Close))
  catch {
    FileOperationFailed(Close) => ()
    _ => fail("unexpected tokenizer terminal-close failure")
  } noraise {
    _ => fail("tokenizer bytes escaped a failed terminal close")
  }
}
