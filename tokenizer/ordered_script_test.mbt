///|
fn ordered_spec() -> @tokenizer.TokenizerSpec raise {
  let vocabulary = base_vocabulary()
  for
    text in [
      "123", "456", "abc", "!abc", "é", "e\u0301", "中文", "カ", "・", "ナ",
      "abc中文", "　", " abc", "  ", "!!\n", "١٢٣", "カ・ナ",
    ] {
    vocabulary.push({
      token_id: vocabulary.length(),
      piece: @utf8.encode(text),
      kind: Normal,
    })
  }
  @tokenizer.TokenizerSpec::new_ordered_script_bpe(
    vocabulary,
    [],
    test_digest(),
    test_limits(),
    ignore_merges=true,
  )
}

///|
test "ordered numeric script and word splits preserve distinct BPE domains" {
  let spec = ordered_spec()
  for
    (text, expected) in [
      ("1234567", [256, 257, 55]),
      ("abc中文abc", [258, 262, 258]),
      ("!abc", [259]),
      ("é", [260]),
      ("e\u0301", [261]),
      ("カ・ナ", [263, 264, 265]),
      ("　 abc", [267, 268]),
      ("!!\n", [270]),
      ("١٢٣٤", [271, 217, 164]),
    ] {
    let encoded = spec.encode_text(
      text,
      special_policy=AsOrdinary,
      output_limit=64,
      truncation=RejectOverflow,
    )
    assert_eq(encoded.token_ids, expected)
    assert_eq(
      spec.decode_text(encoded.token_ids, special_policy=Preserve),
      text,
    )
  }
  try
    spec.encode_bytes(
      [0xff],
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    )
  catch {
    InvalidUtf8Input(byte_offset=0) => ()
    _ => fail("wrong UTF8 error")
  } noraise {
    _ => fail("invalid UTF8 accepted")
  }
}
