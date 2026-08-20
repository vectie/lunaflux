///|
fn luna_work_limits(
  max_input_bytes? : Int = 128,
  max_output_tokens? : Int = 128,
) -> TokenizerLimits {
  {
    max_input_bytes,
    max_output_tokens,
    max_decoded_bytes: 256,
    max_vocab_entries: 512,
    max_merge_rules: 256,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
}

///|
fn luna_work_vocabulary() -> Array[VocabularyEntry] {
  let vocabulary : Array[VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  vocabulary
}

///|
fn luna_work_digest() -> TokenizerDigest raise {
  TokenizerDigest::parse("89abcdef01234567".repeat(4))
}

///|
fn luna_work_corpus_spec() -> TokenizerSpec raise {
  let vocabulary = luna_work_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"he", kind: Normal })
  vocabulary.push({ token_id: 257, piece: b"hel", kind: Normal })
  vocabulary.push({ token_id: 258, piece: b"hell", kind: Normal })
  vocabulary.push({ token_id: 259, piece: b"hello", kind: Normal })
  vocabulary.push({ token_id: 260, piece: b"ab", kind: Normal })
  vocabulary.push({ token_id: 261, piece: b"bc", kind: Normal })
  vocabulary.push({ token_id: 262, piece: b"<e", kind: Special })
  vocabulary.push({ token_id: 263, piece: b"<eos>", kind: Special })
  TokenizerSpec::new(
    vocabulary,
    [
      { left_token_id: 104, right_token_id: 101, result_token_id: 256 },
      { left_token_id: 256, right_token_id: 108, result_token_id: 257 },
      { left_token_id: 257, right_token_id: 108, result_token_id: 258 },
      { left_token_id: 258, right_token_id: 111, result_token_id: 259 },
      { left_token_id: 97, right_token_id: 98, result_token_id: 260 },
      { left_token_id: 98, right_token_id: 99, result_token_id: 261 },
    ],
    luna_work_digest(),
    luna_work_limits(),
  )
}

///|
fn luna_overlap_spec() -> TokenizerSpec raise {
  let vocabulary = luna_work_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"aa", kind: Normal })
  TokenizerSpec::new(
    vocabulary,
    [{ left_token_id: 97, right_token_id: 97, result_token_id: 256 }],
    luna_work_digest(),
    luna_work_limits(),
  )
}

///|
fn luna_new_better_rank_spec() -> TokenizerSpec raise {
  let vocabulary = luna_work_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"ab", kind: Normal })
  vocabulary.push({ token_id: 257, piece: b"cd", kind: Normal })
  vocabulary.push({ token_id: 258, piece: b"abc", kind: Normal })
  TokenizerSpec::new(
    vocabulary,
    [
      { left_token_id: 256, right_token_id: 99, result_token_id: 258 },
      { left_token_id: 97, right_token_id: 98, result_token_id: 256 },
      { left_token_id: 99, right_token_id: 100, result_token_id: 257 },
    ],
    luna_work_digest(),
    luna_work_limits(),
  )
}

///|
struct LunaWorkRun {
  result : EncodeResult
  progress_calls : Int
  work_units : Int
}

///|
fn run_luna_work(
  spec : TokenizerSpec,
  input : Bytes,
  special_policy : SpecialTokenPolicy,
  output_limit : Int,
  truncation : TruncationPolicy,
  budget_units : Int,
) -> LunaWorkRun raise {
  let budget = LunaTokenizerStepBudget::new(budget_units)
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=128,
    output_capacity=128,
    budget,
  )
  let work = worker.begin_bytes(
    input,
    special_policy~,
    output_limit~,
    truncation~,
  )
  let mut calls = 0
  let mut total_work = 0
  for ;; {
    let progress = work.progress()
    let used = work.last_work_units()
    assert_true(used > 0)
    assert_true(used <= budget.work_units())
    calls += 1
    total_work += used
    match progress {
      LunaTokenizerPending => ()
      LunaTokenizerReady => {
        let count = work.token_count()
        let destination = FixedArray::make(count, -1)
        let mut copied = 0
        while copied < count {
          let remaining = count - copied
          let amount = if remaining < budget.work_units() {
            remaining
          } else {
            budget.work_units()
          }
          work.copy_tokens_to(
            destination,
            source_offset=copied,
            destination_offset=copied,
            length=amount,
          )
          assert_eq(work.last_work_units(), amount)
          copied += amount
        }
        let token_ids : Array[Int] = []
        for token_id in destination {
          token_ids.push(token_id)
        }
        return {
          result: {
            token_ids,
            input_bytes: input.length(),
            truncated: work.was_truncated(),
          },
          progress_calls: calls,
          work_units: total_work,
        }
      }
    }
  }
}

///|
fn assert_luna_work_equivalent(
  spec : TokenizerSpec,
  input : Bytes,
  special_policy : SpecialTokenPolicy,
  output_limit : Int,
  truncation : TruncationPolicy,
) -> Unit raise {
  let expected = reference_encode_bytes(
    spec, input, special_policy, output_limit, truncation,
  )
  let totals : Array[Int] = []
  for budget in [1, 17, 65536] {
    let actual = run_luna_work(
      spec, input, special_policy, output_limit, truncation, budget,
    )
    assert_eq(actual.result, expected)
    assert_true(actual.progress_calls > 0)
    totals.push(actual.work_units)
  }
  assert_eq(totals[0], totals[1])
  assert_eq(totals[1], totals[2])
}

///|
test "Luna tokenizer work is equivalent for byte and special-token paths" {
  let spec = luna_work_corpus_spec()
  assert_luna_work_equivalent(
    spec,
    b"hello hello",
    AsOrdinary,
    32,
    RejectOverflow,
  )
  assert_luna_work_equivalent(spec, b"hello<eos><e", Allow, 32, RejectOverflow)
  assert_luna_work_equivalent(spec, b"<e>", AsOrdinary, 16, RejectOverflow)
  assert_luna_work_equivalent(
    spec,
    b"\xff\x00\xfe",
    AsOrdinary,
    8,
    RejectOverflow,
  )
  assert_luna_work_equivalent(
    spec,
    b"\xe4\xbd\xa0\xe5\xa5\xbd\xf0\x9f\x8c\x99",
    AsOrdinary,
    32,
    RejectOverflow,
  )
}

///|
test "Luna tokenizer work recognizes an exact special token at EOF" {
  let spec = luna_work_corpus_spec()
  for budget in [1, 17, 65536] {
    let allowed = run_luna_work(
      spec,
      b"<eos>",
      Allow,
      8,
      RejectOverflow,
      budget,
    )
    assert_eq(allowed.result.token_ids, [263])
    try
      ignore(run_luna_work(spec, b"<eos>", Reject, 8, RejectOverflow, budget))
    catch {
      SpecialTokenRejected(token_id=263) => ()
      error => fail("unexpected Luna EOF-special error: \{Repr(error)}")
    } noraise {
      _ => fail("Luna tokenizer accepted a rejected EOF special token")
    }
  }
}

///|
test "Luna tokenizer work preserves overlap and new-better-rank semantics" {
  assert_luna_work_equivalent(
    luna_overlap_spec(),
    b"aaaaa",
    AsOrdinary,
    16,
    RejectOverflow,
  )
  assert_luna_work_equivalent(
    luna_new_better_rank_spec(),
    b"abcd",
    AsOrdinary,
    16,
    RejectOverflow,
  )
}

///|
test "Luna tokenizer work preserves truncation and exact failures" {
  let spec = luna_work_corpus_spec()
  assert_true(
    reference_encode_bytes(spec, b"hello!", AsOrdinary, 1, Truncate).truncated,
  )
  assert_luna_work_equivalent(spec, b"hello!", AsOrdinary, 1, Truncate)
  for budget in [1, 17, 65536] {
    let exact = run_luna_work(spec, b"hello", AsOrdinary, 1, Truncate, budget)
    assert_eq(exact.result.token_ids, [259])
    assert_false(exact.result.truncated)
  }
  for budget in [1, 17, 65536] {
    try
      ignore(
        run_luna_work(spec, b"hello!", AsOrdinary, 1, RejectOverflow, budget),
      )
    catch {
      OutputTooLong(actual=2, maximum=1) => ()
      _ => fail("unexpected Luna overflow error")
    } noraise {
      _ => fail("Luna tokenizer accepted overflowing output")
    }
    try
      ignore(run_luna_work(spec, b"<eos>", Reject, 8, RejectOverflow, budget))
    catch {
      SpecialTokenRejected(token_id=263) => ()
      error => fail("unexpected Luna special-token error: \{Repr(error)}")
    } noraise {
      _ => fail("Luna tokenizer accepted a rejected special token")
    }
  }
}

///|
test "Luna tokenizer work abort revokes aliases and storage is reusable" {
  let spec = luna_work_corpus_spec()
  let budget = LunaTokenizerStepBudget::new(1)
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=128,
    output_capacity=128,
    budget,
  )
  let first = worker.begin_bytes(
    b"hello hello",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  assert_eq(first.progress(), LunaTokenizerPending)
  try ignore(first.token_count()) catch {
    LunaTokenizerWorkerNotReady => ()
    _ => fail("unexpected running-worker state error")
  } noraise {
    _ => fail("running Luna tokenizer work exposed ready output")
  }
  try ignore(first.was_truncated()) catch {
    LunaTokenizerWorkerNotReady => ()
    _ => fail("unexpected running-worker truncation state error")
  } noraise {
    _ => fail("running Luna tokenizer work exposed truncation state")
  }
  try
    worker.begin_bytes(
      b"abc",
      special_policy=AsOrdinary,
      output_limit=8,
      truncation=RejectOverflow,
    )
  catch {
    LunaTokenizerWorkerBusy => ()
    _ => fail("unexpected busy-worker error")
  } noraise {
    _ => fail("running Luna tokenizer worker accepted another input")
  }
  first.abort()
  try first.abort() catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected repeated-abort error")
  } noraise {
    _ => fail("stale Luna tokenizer work aborted twice")
  }
  try ignore(first.token_count()) catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected post-abort state error")
  } noraise {
    _ => fail("aborted Luna tokenizer worker retained ready output")
  }
  let second = worker.begin_bytes(
    b"abc",
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  try ignore(first.progress()) catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected stale-work reuse error")
  } noraise {
    _ => fail("old Luna tokenizer work progressed a reused worker")
  }
  while second.progress() is LunaTokenizerPending {

  }
  assert_eq(second.token_count(), 2)
  second.abort()
  let third = worker.begin_bytes(
    b"hello",
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  while third.progress() is LunaTokenizerPending {

  }
  assert_eq(third.token_count(), 1)
  third.abort()
}

///|
test "Luna tokenizer budget, capacity, and copy bounds fail closed" {
  for invalid in [0, -1, 65537] {
    try ignore(LunaTokenizerStepBudget::new(invalid)) catch {
      LunaInvalidStepBudget => ()
      _ => fail("unexpected Luna step-budget error")
    } noraise {
      _ => fail("invalid Luna step budget was accepted")
    }
  }
  let spec = luna_work_corpus_spec()
  let budget = LunaTokenizerStepBudget::new(17)
  try
    ignore(
      LunaTokenizerWorker::new(
        spec,
        input_capacity=0,
        output_capacity=8,
        budget,
      ),
    )
  catch {
    LunaInvalidWorkerCapacity => ()
    _ => fail("unexpected Luna worker-capacity error")
  } noraise {
    _ => fail("invalid Luna worker capacity was accepted")
  }
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=128,
    output_capacity=128,
    budget,
  )
  let work = worker.begin_bytes(
    b"hello",
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  while work.progress() is LunaTokenizerPending {

  }
  let destination = FixedArray::make(2, -7)
  try
    work.copy_tokens_to(
      destination,
      source_offset=0,
      destination_offset=0,
      length=2,
    )
  catch {
    LunaInvalidTokenCopy => ()
    _ => fail("unexpected Luna token-copy error")
  } noraise {
    _ => fail("out-of-range Luna token copy was accepted")
  }
  assert_eq(destination[0], -7)
  assert_eq(destination[1], -7)
  work.abort()
  let copy_work = worker.begin_bytes(
    b"!!!!!!!!!!!!!!!!!!",
    special_policy=AsOrdinary,
    output_limit=32,
    truncation=RejectOverflow,
  )
  while copy_work.progress() is LunaTokenizerPending {

  }
  assert_eq(copy_work.token_count(), 18)
  let copy_destination = FixedArray::make(18, -9)
  try
    copy_work.copy_tokens_to(
      copy_destination,
      source_offset=0,
      destination_offset=0,
      length=18,
    )
  catch {
    LunaInvalidTokenCopy => ()
    _ => fail("unexpected over-budget Luna token-copy error")
  } noraise {
    _ => fail("over-budget Luna token copy was accepted")
  }
  for token_id in copy_destination {
    assert_eq(token_id, -9)
  }
  copy_work.copy_tokens_to(
    copy_destination,
    source_offset=0,
    destination_offset=0,
    length=17,
  )
  assert_eq(copy_work.last_work_units(), 17)
  copy_work.copy_tokens_to(
    copy_destination,
    source_offset=17,
    destination_offset=17,
    length=1,
  )
  assert_eq(copy_work.last_work_units(), 1)
}
