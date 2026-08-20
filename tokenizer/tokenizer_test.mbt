///|
fn test_limits(
  max_input_bytes? : Int = 1_024,
  max_output_tokens? : Int = 1_024,
  max_decoded_bytes? : Int = 2_048,
) -> @tokenizer.TokenizerLimits {
  {
    max_input_bytes,
    max_output_tokens,
    max_decoded_bytes,
    max_vocab_entries: 512,
    max_merge_rules: 256,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
}

///|
fn base_vocabulary() -> Array[@tokenizer.VocabularyEntry] {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    let piece : Bytes = [value.to_byte()]
    vocabulary.push({ token_id: value, piece, kind: Normal })
  }
  vocabulary
}

///|
fn test_digest() -> @tokenizer.TokenizerDigest raise @tokenizer.TokenizerError {
  @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4))
}

///|
fn corpus_spec() -> @tokenizer.TokenizerSpec raise @tokenizer.TokenizerError {
  let vocabulary = base_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"he", kind: Normal })
  vocabulary.push({ token_id: 257, piece: b"hel", kind: Normal })
  vocabulary.push({ token_id: 258, piece: b"hell", kind: Normal })
  vocabulary.push({ token_id: 259, piece: b"hello", kind: Normal })
  vocabulary.push({ token_id: 260, piece: b"ab", kind: Normal })
  vocabulary.push({ token_id: 261, piece: b"bc", kind: Normal })
  vocabulary.push({ token_id: 262, piece: b"<e", kind: Special })
  vocabulary.push({ token_id: 263, piece: b"<eos>", kind: Special })
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [
      { left_token_id: 104, right_token_id: 101, result_token_id: 256 },
      { left_token_id: 256, right_token_id: 108, result_token_id: 257 },
      { left_token_id: 257, right_token_id: 108, result_token_id: 258 },
      { left_token_id: 258, right_token_id: 111, result_token_id: 259 },
      { left_token_id: 97, right_token_id: 98, result_token_id: 260 },
      { left_token_id: 98, right_token_id: 99, result_token_id: 261 },
    ],
    test_digest(),
    test_limits(),
  )
}

///|
test "validated identity and immutable metadata" {
  let vocabulary = base_vocabulary()
  let spec = @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    test_digest(),
    test_limits(),
  )
  vocabulary[0] = { token_id: 999, piece: b"changed", kind: Normal }
  inspect(spec.vocab_size(), content="256")
  inspect(spec.merge_count(), content="0")
  inspect(
    spec.digest().as_string(),
    content="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  debug_inspect(
    spec.encode_bytes(
      b"\x00",
      special_policy=AsOrdinary,
      output_limit=4,
      truncation=RejectOverflow,
    ).token_ids,
    content="[0]",
  )
}

///|
test "pinned byte BPE corpus is deterministic" {
  let spec = corpus_spec()
  let first = spec.encode_text(
    "hello hello",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  let second = spec.encode_text(
    "hello hello",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  debug_inspect(first.token_ids, content="[259, 32, 259]")
  assert_eq(first, second)
  debug_inspect(
    spec.encode_text(
      "abc",
      special_policy=AsOrdinary,
      output_limit=8,
      truncation=RejectOverflow,
    ).token_ids,
    content="[260, 99]",
  )
  inspect(
    spec.decode_text(first.token_ids, special_policy=Preserve),
    content="hello hello",
  )
}

///|
test "Unicode uses exact UTF-8 bytes and arbitrary bytes round trip" {
  let spec = corpus_spec()
  let encoded = spec.encode_text(
    "你好🌙",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  debug_inspect(
    encoded.token_ids,
    content="[228, 189, 160, 229, 165, 189, 240, 159, 140, 153]",
  )
  inspect(
    spec.decode_text(encoded.token_ids, special_policy=Preserve),
    content="你好🌙",
  )
  let arbitrary : Bytes = [0xff, 0x00, 0xfe]
  let raw = spec.encode_bytes(
    arbitrary,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  assert_eq(
    spec.decode_bytes(raw.token_ids, special_policy=Preserve),
    arbitrary,
  )
  try spec.decode_text(raw.token_ids, special_policy=Preserve) catch {
    InvalidUtf8Output => ()
    _ => fail("expected invalid UTF-8 output")
  } noraise {
    _ => fail("expected strict text decode to fail")
  }
}

///|
test "special token policy is explicit and longest-match" {
  let spec = corpus_spec()
  let allowed = spec.encode_text(
    "hello<eos><e",
    special_policy=Allow,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(allowed.token_ids, content="[259, 263, 262]")
  inspect(
    spec.decode_text(allowed.token_ids, special_policy=Skip),
    content="hello",
  )
  let ordinary = spec.encode_text(
    "<e>",
    special_policy=AsOrdinary,
    output_limit=16,
    truncation=RejectOverflow,
  )
  debug_inspect(ordinary.token_ids, content="[60, 101, 62]")
  try
    spec.encode_text(
      "prefix<eos>",
      special_policy=Reject,
      output_limit=32,
      truncation=RejectOverflow,
    )
  catch {
    SpecialTokenRejected(token_id=263) => ()
    _ => fail("unexpected special-token rejection")
  } noraise {
    _ => fail("expected special token rejection")
  }
}

///|
test "single-piece decode copies into fixed caller storage transactionally" {
  let spec = corpus_spec()
  let destination = FixedArray::make(8, b'!')
  assert_eq(spec.decoded_piece_bytes(259, special_policy=Preserve), 5)
  assert_eq(
    spec.copy_decoded_piece(
      259,
      destination,
      destination_offset=2,
      special_policy=Preserve,
    ),
    5,
  )
  assert_eq(destination[0], b'!')
  assert_eq(destination[1], b'!')
  assert_eq(destination[2], b'h')
  assert_eq(destination[3], b'e')
  assert_eq(destination[4], b'l')
  assert_eq(destination[5], b'l')
  assert_eq(destination[6], b'o')
  assert_eq(destination[7], b'!')

  let unchanged = FixedArray::make(4, b'?')
  try
    spec.copy_decoded_piece(
      259,
      unchanged,
      destination_offset=0,
      special_policy=Preserve,
    )
  catch {
    DecodeDestinationTooSmall(required=5, available=4) => ()
    _ => fail("unexpected fixed destination error")
  } noraise {
    _ => fail("expected fixed destination rejection")
  }
  for byte in unchanged {
    assert_eq(byte, b'?')
  }
}

///|
test "single-piece decode handles special policy and invalid inputs" {
  let spec = corpus_spec()
  let destination = FixedArray::make(6, b'.')
  assert_eq(spec.decoded_piece_bytes(263, special_policy=Skip), 0)
  assert_eq(
    spec.copy_decoded_piece(
      263,
      destination,
      destination_offset=6,
      special_policy=Skip,
    ),
    0,
  )
  try
    spec.copy_decoded_piece(
      263,
      destination,
      destination_offset=-1,
      special_policy=Skip,
    )
  catch {
    InvalidDecodeDestination => ()
    _ => fail("unexpected destination-offset error")
  } noraise {
    _ => fail("expected destination-offset rejection")
  }
  try spec.decoded_piece_bytes(9_999, special_policy=Preserve) catch {
    UnknownTokenId(token_id=9_999) => ()
    _ => fail("unexpected unknown-token error")
  } noraise {
    _ => fail("expected unknown-token rejection")
  }
}

///|
test "output and input bounds reject or truncate deterministically" {
  let spec = corpus_spec()
  let truncated = spec.encode_text(
    "hello!",
    special_policy=AsOrdinary,
    output_limit=1,
    truncation=Truncate,
  )
  debug_inspect(
    truncated,
    content="{ token_ids: [259], input_bytes: 6, truncated: true }",
  )
  try
    spec.encode_text(
      "hello!",
      special_policy=AsOrdinary,
      output_limit=1,
      truncation=RejectOverflow,
    )
  catch {
    OutputTooLong(actual=2, maximum=1) => ()
    _ => fail("unexpected output bound error")
  } noraise {
    _ => fail("expected output overflow")
  }
  let bounded = @tokenizer.TokenizerSpec::new(
    base_vocabulary(),
    [],
    test_digest(),
    test_limits(max_input_bytes=2),
  )
  try
    bounded.encode_bytes(
      b"abc",
      special_policy=AsOrdinary,
      output_limit=8,
      truncation=RejectOverflow,
    )
  catch {
    InputTooLong(actual=3, maximum=2) => ()
    _ => fail("unexpected input bound error")
  } noraise {
    _ => fail("expected input overflow")
  }
}

///|
test "invalid digest and incomplete byte vocabulary fail closed" {
  try @tokenizer.TokenizerDigest::parse("ABC") catch {
    InvalidDigest => ()
    _ => fail("unexpected digest error")
  } noraise {
    _ => fail("expected invalid digest")
  }
  let incomplete = base_vocabulary()
  ignore(incomplete.pop())
  try
    @tokenizer.TokenizerSpec::new(incomplete, [], test_digest(), test_limits())
  catch {
    MissingByteToken(byte_value=255) => ()
    _ => fail("unexpected incomplete-vocabulary error")
  } noraise {
    _ => fail("expected incomplete byte vocabulary rejection")
  }
}

///|
test "token IDs must be dense within the vocabulary range" {
  let sparse = base_vocabulary()
  sparse.push({ token_id: 300, piece: b"ab", kind: Normal })
  try
    @tokenizer.TokenizerSpec::new(sparse, [], test_digest(), test_limits())
  catch {
    InvalidTokenId(entry_index=256) => ()
    _ => fail("unexpected sparse token-ID error")
  } noraise {
    _ => fail("sparse token IDs were admitted")
  }
}

///|
test "duplicate vocabulary and malformed merge rules fail closed" {
  let duplicate = base_vocabulary()
  duplicate.push({ token_id: 256, piece: b"a", kind: Normal })
  try
    @tokenizer.TokenizerSpec::new(duplicate, [], test_digest(), test_limits())
  catch {
    DuplicateTokenPiece(first_index=97, duplicate_index=256) => ()
    _ => fail("unexpected duplicate piece error")
  } noraise {
    _ => fail("expected duplicate piece rejection")
  }
  let vocabulary = base_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"ba", kind: Normal })
  try
    @tokenizer.TokenizerSpec::new(
      vocabulary,
      [{ left_token_id: 97, right_token_id: 98, result_token_id: 256 }],
      test_digest(),
      test_limits(),
    )
  catch {
    InvalidMergeResult(rule_index=0) => ()
    _ => fail("unexpected merge validation error")
  } noraise {
    _ => fail("expected malformed merge rejection")
  }
}

///|
test "decode rejects unknown IDs and decoded byte overflow" {
  let spec = corpus_spec()
  try spec.decode_bytes([9_999], special_policy=Preserve) catch {
    UnknownTokenId(token_id=9_999) => ()
    _ => fail("unexpected unknown-token error")
  } noraise {
    _ => fail("expected unknown token rejection")
  }
  let bounded = @tokenizer.TokenizerSpec::new(
    base_vocabulary(),
    [],
    test_digest(),
    test_limits(max_decoded_bytes=2),
  )
  try bounded.decode_bytes([97, 98, 99], special_policy=Preserve) catch {
    DecodedOutputTooLong(maximum=2) => ()
    _ => fail("unexpected decoded output error")
  } noraise {
    _ => fail("expected decoded byte overflow")
  }
}
