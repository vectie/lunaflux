///|
fn sentencepiece_fixture_digest() -> TokenizerDigest {
  try! TokenizerDigest::parse(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
}

///|
fn sentencepiece_fixture_limits() -> TokenizerLimits {
  {
    max_input_bytes: 128,
    max_output_tokens: 128,
    max_decoded_bytes: 512,
    max_vocab_entries: 260,
    max_merge_rules: 8,
    max_token_bytes: 16,
    max_special_tokens: 3,
  }
}

///|
fn sentencepiece_fixture_spec() -> TokenizerSpec raise {
  let vocabulary : Array[VocabularyEntry] = [
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
  TokenizerSpec::new_sentencepiece_bpe(
    vocabulary,
    [],
    sentencepiece_fixture_digest(),
    sentencepiece_fixture_limits(),
    unknown_token_id=0,
    bos_token_id=1,
  )
}

///|
test "SentencePiece normalization template and byte fallback match selected corpus" {
  let spec = sentencepiece_fixture_spec()
  assert_eq(
    spec.encode_text(
      "",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    [1],
  )
  assert_eq(
    spec.encode_text(
      "Hello world",
      special_policy=AsOrdinary,
      output_limit=32,
      truncation=RejectOverflow,
    ).token_ids,
    [
      1, 229, 153, 132, 75, 104, 111, 111, 114, 229, 153, 132, 122, 114, 117, 111,
      103,
    ],
  )
  assert_eq(
    spec.encode_text(
      "💩",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    [1, 229, 153, 132, 243, 162, 149, 172],
  )
}

///|
test "SentencePiece decoder order distinguishes model pieces from byte fallback" {
  let spec = sentencepiece_fixture_spec()
  assert_eq(spec.decode_text([259], special_policy=Skip), "L")
  assert_eq(spec.decode_text([229, 153, 132, 75], special_policy=Skip), "▁H")
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    spec.copy_decoded_piece(
      259,
      destination,
      destination_offset=0,
      special_policy=Skip,
    ),
    2,
  )
  assert_eq(destination[0], b' ')
  assert_eq(destination[1], b'L')
}

///|
test "SentencePiece worker is bounded reusable and rejects malformed UTF-8" {
  let spec = sentencepiece_fixture_spec()
  assert_eq(spec.luna_worker_required_int_cells(16), 300UL)
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=16,
    output_capacity=32,
    LunaTokenizerStepBudget::new(1),
  )
  let work = worker.begin_bytes(
    @utf8.encode("é"),
    special_policy=AsOrdinary,
    output_limit=16,
    truncation=RejectOverflow,
  )
  while work.progress() is LunaTokenizerPending {
    assert_eq(work.last_work_units(), 1)
  }
  let output = FixedArray::make(work.token_count(), -1)
  for index in 0..<output.length() {
    work.copy_tokens_to(
      output,
      source_offset=index,
      destination_offset=index,
      length=1,
    )
  }
  assert_eq(output, FixedArray::from_array([1, 229, 153, 132, 198, 172]))
  work.abort()
  try
    ignore(
      spec.encode_bytes(
        [0xff],
        special_policy=AsOrdinary,
        output_limit=16,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InvalidUtf8Input(byte_offset=0) => ()
    _ => fail("unexpected malformed UTF-8 failure")
  } noraise {
    _ => fail("malformed SentencePiece input was accepted")
  }
}

///|
test "SentencePiece template participates in output bounds" {
  let spec = sentencepiece_fixture_spec()
  assert_eq(
    spec.encode_text(
      "",
      special_policy=AsOrdinary,
      output_limit=1,
      truncation=RejectOverflow,
    ).token_ids,
    [1],
  )
  try
    ignore(
      spec.encode_text(
        "A",
        special_policy=AsOrdinary,
        output_limit=1,
        truncation=RejectOverflow,
      ),
    )
  catch {
    OutputTooLong(actual=5, maximum=1) => ()
    _ => fail("unexpected SentencePiece output-bound failure")
  } noraise {
    _ => fail("SentencePiece template overflow was accepted")
  }
}
