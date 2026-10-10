///|
fn text_spec() -> @tokenizer.TokenizerSpec raise {
  let entries : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    entries.push({
      token_id: value,
      piece: Bytes::make(1, value.to_byte()),
      kind: Normal,
    })
  }
  entries.push({ token_id: 256, piece: b"<s>", kind: Special, })
  @tokenizer.TokenizerSpec::new(
    entries,
    [],
    @tokenizer.TokenizerDigest::parse("a".repeat(64)),
    {
      max_input_bytes: 1024,
      max_output_tokens: 128,
      max_decoded_bytes: 1024,
      max_vocab_entries: 257,
      max_merge_rules: 1,
      max_token_bytes: 32,
      max_special_tokens: 1,
    },
  )
}

///|
test "shared text input preserves literals and produces exact little endian IDs" {
  let result = @checkpoint_text_input.encode_text_input(
    text_spec(),
    b"a<s>b",
    maximum_tokens=3,
  )
  debug_inspect(result.token_ids().to_array(), content="[97, 256, 98]")
  assert_eq(result.decoded_bytes(), b"a<s>b")
  assert_eq(
    result.i32_le_bytes(),
    b"\x61\x00\x00\x00\x00\x01\x00\x00\x62\x00\x00\x00",
  )
  let empty = @checkpoint_text_input.encode_text_input(
    text_spec(),
    b"",
    maximum_tokens=1,
  )
  assert_eq(empty.token_ids().length(), 0)
  assert_eq(empty.i32_le_bytes(), b"")
}

///|
test "shared text input rejects context overflow instead of silently truncating" {
  try
    @checkpoint_text_input.encode_text_input(
      text_spec(),
      b"abc",
      maximum_tokens=2,
    )
  catch {
    @tokenizer.OutputTooLong(..) => ()
    _ => fail("unexpected overflow error")
  } noraise {
    _ => fail("overflow was accepted")
  }
}
