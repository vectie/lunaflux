///|
fn direct_spec(enabled : Bool) -> @tokenizer.TokenizerSpec raise {
  let vocabulary = base_vocabulary()
  // Neither of these whole pieces is reachable through the merge table.
  vocabulary.push({ token_id: 256, piece: b"abc", kind: Normal, })
  vocabulary.push({ token_id: 257, piece: b"ab", kind: Normal, })
  vocabulary.push({ token_id: 258, piece: b"<s>", kind: Special, })
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [{ left_token_id: 97, right_token_id: 98, result_token_id: 257, }],
    test_digest(),
    test_limits(),
    ignore_merges=enabled,
  )
}

///|
test "whole pretoken direct lookup does not disable fallback BPE" {
  let enabled = direct_spec(true)
  let disabled = direct_spec(false)
  debug_inspect(
    enabled.encode_text(
      "abc",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[256]",
  )
  debug_inspect(
    disabled.encode_text(
      "abc",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[257, 99]",
  )
  debug_inspect(
    enabled.encode_text(
      "abd",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[257, 100]",
  )
  debug_inspect(
    enabled.encode_text(
      "abc<s>abc",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[256, 258, 256]",
  )
  let utf8 = enabled.encode_text(
    "中文",
    special_policy=AsOrdinary,
    output_limit=16,
    truncation=RejectOverflow,
  )
  assert_eq(
    enabled.decode_text(utf8.token_ids, special_policy=Preserve),
    "中文",
  )
}

///|
test "direct lookup preserves bounded resumable worker and reuse" {
  let worker = @tokenizer.LunaTokenizerWorker::new(
    direct_spec(true),
    input_capacity=128,
    output_capacity=128,
    @tokenizer.LunaTokenizerStepBudget::new(1),
  )
  for _ in 0..<2 {
    let work = worker.begin_bytes(
      b"abc<s>abc",
      special_policy=Allow,
      output_limit=16,
      truncation=RejectOverflow,
    )
    let mut polls = 0
    while work.progress() is LunaTokenizerPending {
      assert_eq(work.last_work_units(), 1)
      polls += 1
      assert_true(polls < 1000)
    }
    assert_eq(work.token_count(), 3)
    assert_eq(work.token_status(0).token_id(), 256)
    assert_eq(work.token_status(1).token_id(), 258)
    assert_eq(work.token_status(2).token_id(), 256)
    work.abort()
  }
}

///|
test "direct lookup keeps truncation special rejection and ordinary policy" {
  let spec = direct_spec(true)
  let truncated = spec.encode_text(
    "abc<s>abc",
    special_policy=Allow,
    output_limit=2,
    truncation=Truncate,
  )
  assert_true(truncated.truncated)
  debug_inspect(truncated.token_ids, content="[256, 258]")
  try
    spec.encode_text(
      "abc<s>abc",
      special_policy=Allow,
      output_limit=2,
      truncation=RejectOverflow,
    )
  catch {
    OutputTooLong(actual=3, maximum=2) => ()
    _ => fail("unexpected overflow error")
  } noraise {
    _ => fail("expected overflow")
  }
  try
    spec.encode_text(
      "abc<s>abc",
      special_policy=Reject,
      output_limit=16,
      truncation=RejectOverflow,
    )
  catch {
    SpecialTokenRejected(token_id=258) => ()
    _ => fail("unexpected special error")
  } noraise {
    _ => fail("expected special rejection")
  }
  debug_inspect(
    spec.encode_text(
      "abc<s>abc",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[257, 99, 60, 115, 62, 257, 99]",
  )
}
