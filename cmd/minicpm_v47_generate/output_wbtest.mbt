///|
test "result decoding preserves complete UTF8 and reports a truncated byte prefix" {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal, })
  }
  let tokenizer = @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0".repeat(64)),
    {
      max_input_bytes: 64,
      max_output_tokens: 64,
      max_decoded_bytes: 1024,
      max_vocab_entries: 256,
      max_merge_rules: 1,
      max_token_bytes: 1,
      max_special_tokens: 1,
    },
  )
  assert_eq(decode_output(tokenizer, [0xe2, 0x82, 0xac]), ("€", true))
  assert_eq(decode_output(tokenizer, [0xe2, 0x82]), ("�", false))
  assert_eq(decode_output(tokenizer, []), ("", true))
  assert_true(
    try decode_output(tokenizer, [256]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
