///|
fn triplet_spec() -> @tokenizer.TokenizerSpec raise {
  let vocabulary = base_vocabulary()
  for
    piece in [
      b"123", b"456", b"1234", b"abc", b"\xc3\xa9", b"e\xcc\x81", b"\xd9\xa1\xd9\xa2\xd9\xa3",
    ] {
    vocabulary.push({ token_id: vocabulary.length(), piece, kind: Normal, })
  }
  @tokenizer.TokenizerSpec::new_digit_triplet_bpe(
    vocabulary,
    [],
    test_digest(),
    test_limits(),
    ignore_merges=true,
  )
}

///|
test "triplet regex groups numbers by scalar and does not normalize Unicode" {
  let spec = triplet_spec()
  debug_inspect(
    spec.encode_text(
      "1234567",
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    ).token_ids,
    content="[256, 257, 55]",
  )
  debug_inspect(
    spec.encode_text(
      "abc",
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    ).token_ids,
    content="[259]",
  )
  debug_inspect(
    spec.encode_text(
      "é",
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    ).token_ids,
    content="[260]",
  )
  let decomposed = spec.encode_text(
    "e\u0301",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  // The combining mark is not a letter in this regex, unlike mark-aware Qwen.
  debug_inspect(decomposed.token_ids, content="[101, 204, 129]")
  assert_eq(
    spec.decode_text(decomposed.token_ids, special_policy=Preserve),
    "e\u0301",
  )
  debug_inspect(
    spec.encode_text(
      "١٢٣٤",
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    ).token_ids,
    content="[262, 217, 164]",
  )
  try
    spec.encode_bytes(
      [0xff],
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    )
  catch {
    InvalidUtf8Input(byte_offset=0) => ()
    _ => fail("expected invalid UTF8")
  } noraise {
    _ => fail("invalid UTF8 accepted")
  }
}
