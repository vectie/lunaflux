///|
fn qwen35_mark_test_spec(
  mark_aware : Bool,
) -> TokenizerSpec raise TokenizerError {
  let vocabulary : Array[VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  vocabulary.push({ token_id: 256, piece: b"\xe0\xa4", kind: Normal })
  vocabulary.push({ token_id: 257, piece: b"\xe0\xa4\xbe", kind: Normal })
  vocabulary.push({ token_id: 258, piece: b"a\xe0\xa4\xbe", kind: Normal })
  let merges : Array[MergeRule] = [
    { left_token_id: 224, right_token_id: 164, result_token_id: 256 },
    { left_token_id: 256, right_token_id: 190, result_token_id: 257 },
    { left_token_id: 97, right_token_id: 257, result_token_id: 258 },
  ]
  let digest = TokenizerDigest::parse(
    if mark_aware {
      "13579bdf".repeat(8)
    } else {
      "02468ace".repeat(8)
    },
  )
  let limits : TokenizerLimits = {
    max_input_bytes: 64,
    max_output_tokens: 64,
    max_decoded_bytes: 256,
    max_vocab_entries: 512,
    max_merge_rules: 64,
    max_token_bytes: 32,
    max_special_tokens: 32,
  }
  if mark_aware {
    TokenizerSpec::new_qwen35_bpe(vocabulary, merges, digest, limits)
  } else {
    TokenizerSpec::new_qwen3_bpe(vocabulary, merges, digest, limits)
  }
}

///|
test "Qwen3.5 mark-aware split does not alter dense Qwen3 behavior" {
  assert_eq(qwen35_unicode_mark_ranges.length(), 327 * 6)
  assert_true(qwen35_is_mark(0x093e))
  assert_true(qwen35_is_mark(0x0301))
  assert_true(qwen35_is_mark(0xe0100))
  assert_false(qwen35_is_mark('a'.to_int()))
  assert_false(qwen35_is_mark('中'.to_int()))
  let dense = qwen35_mark_test_spec(false).encode_text(
    "aा",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  let mark_aware = qwen35_mark_test_spec(true).encode_text(
    "aा",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(dense.token_ids, content="[97, 257]")
  debug_inspect(mark_aware.token_ids, content="[258]")
  assert_eq(
    qwen35_mark_test_spec(false).decode_text(
      dense.token_ids,
      special_policy=Preserve,
    ),
    "aा",
  )
  assert_eq(
    qwen35_mark_test_spec(true).decode_text(
      mark_aware.token_ids,
      special_policy=Preserve,
    ),
    "aा",
  )
}
