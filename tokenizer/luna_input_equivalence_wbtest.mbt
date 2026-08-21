///|
fn run_luna_streamed_input(
  spec : TokenizerSpec,
  input : Bytes,
  split : Int,
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
  let write = worker.begin_luna_input(
    byte_count=input.length(),
    special_policy~,
    output_limit~,
    truncation~,
  )
  for index in 0..<split {
    write.push_byte(input[index])
  }
  for index in split..<input.length() {
    write.push_byte(input[index])
  }
  let work = write.finish()
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
          let amount = Int::min(count - copied, budget.work_units())
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
fn assert_streamed_all_splits(
  spec : TokenizerSpec,
  input : Bytes,
  special_policy : SpecialTokenPolicy,
  output_limit : Int,
  truncation : TruncationPolicy,
) -> Unit raise {
  let expected = reference_encode_bytes(
    spec, input, special_policy, output_limit, truncation,
  )
  for budget in [1, 17, 65536] {
    let legacy = run_luna_work(
      spec, input, special_policy, output_limit, truncation, budget,
    )
    assert_eq(legacy.result, expected)
    for split in 0..<=input.length() {
      let streamed = run_luna_streamed_input(
        spec, input, split, special_policy, output_limit, truncation, budget,
      )
      assert_eq(streamed.result, expected)
      assert_eq(streamed.work_units, legacy.work_units)
    }
  }
}

///|
test "streamed Luna input matches legacy and reference at every split" {
  let corpus = luna_work_corpus_spec()
  assert_streamed_all_splits(corpus, b"", AsOrdinary, 8, RejectOverflow)
  assert_streamed_all_splits(
    corpus,
    b"hello hello",
    AsOrdinary,
    32,
    RejectOverflow,
  )
  assert_streamed_all_splits(corpus, b"hello<eos><e", Allow, 32, RejectOverflow)
  assert_streamed_all_splits(corpus, b"hello", Reject, 8, RejectOverflow)
  assert_streamed_all_splits(
    corpus,
    b"\xff\x00\xfe",
    AsOrdinary,
    8,
    RejectOverflow,
  )
  assert_streamed_all_splits(
    corpus,
    b"\xe4\xbd\xa0\xe5\xa5\xbd\xf0\x9f\x8c\x99",
    AsOrdinary,
    32,
    RejectOverflow,
  )
  assert_streamed_all_splits(corpus, b"hello!", AsOrdinary, 1, Truncate)
  assert_streamed_all_splits(
    luna_overlap_spec(),
    b"aaaaa",
    AsOrdinary,
    16,
    RejectOverflow,
  )
  assert_streamed_all_splits(
    luna_new_better_rank_spec(),
    b"abcd",
    AsOrdinary,
    16,
    RejectOverflow,
  )
}

///|
test "streamed Luna input preserves exact progress failures at every split" {
  let spec = luna_work_corpus_spec()
  for budget in [1, 17, 65536] {
    for split in 0..<=b"<eos>".length() {
      try
        ignore(
          run_luna_streamed_input(
            spec,
            b"<eos>",
            split,
            Reject,
            8,
            RejectOverflow,
            budget,
          ),
        )
      catch {
        SpecialTokenRejected(token_id=263) => ()
        error => fail("unexpected streamed special error: \{Repr(error)}")
      } noraise {
        _ => fail("streamed rejected special token was accepted")
      }
    }
    for split in 0..<=b"hello!".length() {
      try
        ignore(
          run_luna_streamed_input(
            spec,
            b"hello!",
            split,
            AsOrdinary,
            1,
            RejectOverflow,
            budget,
          ),
        )
      catch {
        OutputTooLong(actual=2, maximum=1) => ()
        error => fail("unexpected streamed overflow error: \{Repr(error)}")
      } noraise {
        _ => fail("streamed overflowing output was accepted")
      }
    }
  }
}

///|
test "streamed Luna input copies bytes instead of retaining a source alias" {
  let spec = luna_work_corpus_spec()
  for split in 0..<=b"hello".length() {
    let source : Array[Byte] = [b'h', b'e', b'l', b'l', b'o']
    let worker = LunaTokenizerWorker::new(
      spec,
      input_capacity=128,
      output_capacity=128,
      LunaTokenizerStepBudget::new(1),
    )
    let write = worker.begin_luna_input(
      byte_count=source.length(),
      special_policy=AsOrdinary,
      output_limit=8,
      truncation=RejectOverflow,
    )
    for index in 0..<split {
      write.push_byte(source[index])
    }
    for index in 0..<split {
      source[index] = b'x'
    }
    for index in split..<source.length() {
      write.push_byte(source[index])
    }
    let work = write.finish()
    while work.progress() is LunaTokenizerPending {

    }
    assert_eq(work.token_count(), 1)
    assert_eq(work.token_status(0).token_id(), 259)
    work.abort()
  }
}
