///|
fn fixture_tokenizer() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  vocabulary.push({ token_id: 300, piece: b"helloE", kind: Normal })
  vocabulary.push({ token_id: 301, piece: b"NDx", kind: Normal })
  vocabulary.push({ token_id: 302, piece: b"abcE", kind: Normal })
  vocabulary.push({ token_id: 1000, piece: b"<eos>", kind: Special })
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 64,
      max_output_tokens: 64,
      max_decoded_bytes: 256,
      max_vocab_entries: 512,
      max_merge_rules: 16,
      max_token_bytes: 8,
      max_special_tokens: 4,
    },
  )
}

///|
fn fixture_inference() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=32,
    max_context_tokens=64,
    max_token_id=1000,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=1000UL,
    max_top_k=1001,
    max_temperature=4.0,
  )
}

///|
fn stops(values : Array[Bytes]) -> @inference.StopConditions raise {
  @inference.StopConditions::new([], values, fixture_inference())
}

///|
test "UTF-8 split across byte tokens publishes only a complete scalar" {
  let output = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    fixture_inference(),
    special_policy=Preserve,
  )
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    output
    .push_token_into(0xe4, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  assert_eq(
    output
    .push_token_into(0xbd, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  let completed = output.push_token_into(
    0xa0,
    destination,
    destination_offset=1,
  )
  assert_eq(completed.copied_bytes(), 3)
  assert_false(completed.stop_matched())
  assert_eq(destination[1], b'\xe4')
  assert_eq(destination[2], b'\xbd')
  assert_eq(destination[3], b'\xa0')
}

///|
test "stop text spanning token pieces is withheld before terminal" {
  let output = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([b"END"]),
    fixture_inference(),
    special_policy=Preserve,
  )
  let destination = FixedArray::make(16, b'!')
  let first = output.push_token_into(300, destination, destination_offset=0)
  assert_eq(first.copied_bytes(), 5)
  assert_false(first.stop_matched())
  assert_eq(Bytes::makei(5, index => destination[index]), b"hello")
  let second = output.push_token_into(301, destination, destination_offset=5)
  assert_eq(second.copied_bytes(), 0)
  assert_true(second.stop_matched())
  assert_true(output.is_stopped())
  try output.push_token_into(1, destination, destination_offset=0) catch {
    Invalid(rule=Lifecycle, issue=Terminal) => ()
    _ => fail("unexpected stopped-stream error")
  } noraise {
    _ => fail("stopped stream accepted another token")
  }
}

///|
test "destination rejection leaves stream state retryable" {
  let output = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([b"END"]),
    fixture_inference(),
    special_policy=Preserve,
  )
  let small = FixedArray::make(4, b'?')
  try output.push_token_into(300, small, destination_offset=0) catch {
    Invalid(rule=Destination, issue=TooSmall) => ()
    _ => fail("unexpected destination error")
  } noraise {
    _ => fail("small destination accepted output")
  }
  for byte in small {
    assert_eq(byte, b'?')
  }
  let exact = FixedArray::make(5, b'!')
  assert_eq(
    output.push_token_into(300, exact, destination_offset=0).copied_bytes(),
    5,
  )
  assert_eq(Bytes::makei(5, index => exact[index]), b"hello")
}

///|
test "finish publishes an unmatched stop prefix and rejects truncation" {
  let output = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([b"END"]),
    fixture_inference(),
    special_policy=Preserve,
  )
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    output
    .push_token_into(302, destination, destination_offset=0)
    .copied_bytes(),
    3,
  )
  assert_eq(output.finish_into(destination, destination_offset=3), 1)
  assert_eq(Bytes::makei(4, index => destination[index]), b"abcE")
  assert_true(output.is_finished())

  let truncated = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    fixture_inference(),
    special_policy=Preserve,
  )
  ignore(truncated.push_token_into(0xe4, destination, destination_offset=0))
  try truncated.finish_into(destination, destination_offset=0) catch {
    Invalid(rule=Utf8, issue=Invalid) => ()
    _ => fail("unexpected truncated UTF-8 error")
  } noraise {
    _ => fail("truncated UTF-8 finished successfully")
  }
}

///|
test "invalid UTF-8 unknown tokens and skipped specials are payload safe" {
  let destination = FixedArray::make(8, b'!')
  let invalid_utf8 = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    fixture_inference(),
    special_policy=Preserve,
  )
  try
    invalid_utf8.push_token_into(0xc0, destination, destination_offset=0)
  catch {
    Invalid(rule=Utf8, issue=Invalid) => ()
    _ => fail("unexpected invalid UTF-8 error")
  } noraise {
    _ => fail("invalid UTF-8 token was published")
  }
  let unknown = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    fixture_inference(),
    special_policy=Preserve,
  )
  try unknown.push_token_into(9999, destination, destination_offset=0) catch {
    Invalid(rule=Token, issue=Invalid) => ()
    _ => fail("unexpected unknown-token error")
  } noraise {
    _ => fail("unknown token was published")
  }
  let skipped = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([]),
    fixture_inference(),
    special_policy=Skip,
  )
  assert_eq(
    skipped
    .push_token_into(1000, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
}

///|
test "stop matcher spans UTF-8 bytes and rejects an undersized envelope" {
  let utf8_stop = IncrementalOutput::new(
    fixture_tokenizer(),
    stops([b"\xe4\xbd\xa0"]),
    fixture_inference(),
    special_policy=Preserve,
  )
  let destination = FixedArray::make(8, b'!')
  ignore(utf8_stop.push_token_into(0xe4, destination, destination_offset=0))
  ignore(utf8_stop.push_token_into(0xbd, destination, destination_offset=0))
  let matched = utf8_stop.push_token_into(
    0xa0,
    destination,
    destination_offset=0,
  )
  assert_true(matched.stop_matched())
  assert_eq(matched.copied_bytes(), 0)

  let tiny = @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=32,
    max_context_tokens=64,
    max_token_id=1000,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=1001,
    max_temperature=4.0,
  )
  try
    IncrementalOutput::new(
      fixture_tokenizer(),
      @inference.StopConditions::new([], [b"END"], tiny),
      tiny,
      special_policy=Preserve,
    )
  catch {
    Invalid(rule=DecodeEnvelope, issue=Unsupported) => ()
    _ => fail("unexpected envelope error")
  } noraise {
    _ => fail("undersized decoded-delta envelope was accepted")
  }
}
