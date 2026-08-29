///|
fn sentencepiece_output_spec() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = [
    { token_id: 0, piece: b"<unk>", kind: Special },
    { token_id: 1, piece: b"<s>", kind: Special },
    { token_id: 2, piece: b"</s>", kind: Special },
  ]
  for value in 0..<256 {
    vocabulary.push({
      token_id: value + 3,
      piece: [value.to_byte()],
      kind: ByteFallback,
    })
  }
  vocabulary.push({ token_id: 259, piece: @utf8.encode("▁L"), kind: Normal })
  @tokenizer.TokenizerSpec::new_sentencepiece_bpe(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("fedcba9876543210".repeat(4)),
    {
      max_input_bytes: 64,
      max_output_tokens: 64,
      max_decoded_bytes: 256,
      max_vocab_entries: 260,
      max_merge_rules: 8,
      max_token_bytes: 16,
      max_special_tokens: 3,
    },
    unknown_token_id=0,
    bos_token_id=1,
  )
}

///|
test "incremental SentencePiece decode strips only the stream-leading model space" {
  let output = IncrementalOutput::new(
    sentencepiece_output_spec(),
    stops([]),
    fixture_inference(),
    special_policy=Skip,
  )
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    output.push_token_into(1, destination, destination_offset=0).copied_bytes(),
    0,
  )
  assert_eq(
    output
    .push_token_into(259, destination, destination_offset=0)
    .copied_bytes(),
    1,
  )
  assert_eq(destination[0], b'L')
  assert_eq(
    output
    .push_token_into(100, destination, destination_offset=1)
    .copied_bytes(),
    1,
  )
  assert_eq(destination[1], b'a')
}

///|
test "incremental SentencePiece byte fallback fuses UTF-8 without decoder reordering" {
  let output = IncrementalOutput::new(
    sentencepiece_output_spec(),
    stops([]),
    fixture_inference(),
    special_policy=Skip,
  )
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    output
    .push_token_into(229, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  assert_eq(
    output
    .push_token_into(153, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  assert_eq(
    output
    .push_token_into(132, destination, destination_offset=0)
    .copied_bytes(),
    3,
  )
  assert_eq(Bytes::makei(3, index => destination[index]), @utf8.encode("▁"))
}
