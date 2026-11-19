///|
test "MiniCPM5 inserts BOS once and cannot merge across a numeric triplet" {
  let entries : Array[@tokenizer.VocabularyEntry] = [
    { token_id: 0, piece: b"<s>", kind: Special, },
  ]
  for value in 0..<256 {
    entries.push({
      token_id: value + 1,
      piece: Bytes::make(1, value.to_byte()),
      kind: Normal,
    })
  }
  for (id, piece) in [(257, b"12"), (258, b"123"), (259, b"1234")] {
    entries.push({ token_id: id, piece, kind: Normal, })
  }
  let spec = @tokenizer.TokenizerSpec::new_minicpm5_bpe(
    entries,
    [
      { left_token_id: 50, right_token_id: 51, result_token_id: 257, },
      { left_token_id: 257, right_token_id: 52, result_token_id: 258, },
      { left_token_id: 258, right_token_id: 53, result_token_id: 259, },
    ],
    @tokenizer.TokenizerDigest::parse("a".repeat(64)),
    {
      max_input_bytes: 1024,
      max_output_tokens: 1024,
      max_decoded_bytes: 4096,
      max_vocab_entries: 300,
      max_merge_rules: 16,
      max_token_bytes: 64,
      max_special_tokens: 8,
    },
  )
  let encoded = spec.encode_text(
    "1234",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  assert_eq(encoded.token_ids, [0, 258, 53])
  assert_eq(spec.decode_text(encoded.token_ids, special_policy=Skip), "1234")
}
