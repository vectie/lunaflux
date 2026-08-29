///|
fn finish_luna_input_work(
  work : LunaTokenizerWork,
  input_bytes : Int,
) -> EncodeResult raise {
  while work.progress() is LunaTokenizerPending {

  }
  let count = work.token_count()
  let tokens : Array[Int] = []
  for index in 0..<count {
    let status = work.token_status(index)
    assert_true(status.is_ready())
    tokens.push(status.token_id())
  }
  { token_ids: tokens, input_bytes, truncated: work.was_truncated() }
}

///|
fn run_luna_input_writer(
  spec : TokenizerSpec,
  input : Bytes,
  special_policy : SpecialTokenPolicy,
  output_limit : Int,
  truncation : TruncationPolicy,
  budget_units : Int,
) -> EncodeResult raise {
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=128,
    output_capacity=128,
    LunaTokenizerStepBudget::new(budget_units),
  )
  let write = worker.begin_luna_input(
    byte_count=input.length(),
    special_policy~,
    output_limit~,
    truncation~,
  )
  for byte in input {
    write.push_byte(byte)
  }
  let work = write.finish()
  let result = finish_luna_input_work(work, input.length())
  work.abort()
  result
}

///|
test "Luna input writer matches the compatibility facade and frozen oracle" {
  let spec = luna_work_corpus_spec()
  for
    case in [
      (b"", AsOrdinary),
      (b"hello hello", AsOrdinary),
      (b"hello<eos><e", Allow),
      (b"<eo", Allow),
      (b"\xff\x00\xfe", AsOrdinary),
      (b"\xe4\xbd\xa0\xf0\x9f\x8c\x99", AsOrdinary),
    ] {
    let (input, policy) = case
    let expected = reference_encode_bytes(
      spec,
      input,
      policy,
      32,
      RejectOverflow,
    )
    for budget in [1, 17, 65536] {
      assert_eq(
        run_luna_input_writer(spec, input, policy, 32, RejectOverflow, budget),
        expected,
      )
    }
    assert_eq(
      spec.encode_bytes(
        input,
        special_policy=policy,
        output_limit=32,
        truncation=RejectOverflow,
      ),
      expected,
    )
  }
}

///|
test "Luna input writer preserves special EOF rejection and truncation" {
  let spec = luna_work_corpus_spec()
  assert_eq(
    run_luna_input_writer(spec, b"<eos>", Allow, 8, RejectOverflow, 1).token_ids,
    [263],
  )
  try
    ignore(run_luna_input_writer(spec, b"<eos>", Reject, 8, RejectOverflow, 1))
  catch {
    SpecialTokenRejected(token_id=263) => ()
    _ => fail("unexpected special rejection")
  } noraise {
    _ => fail("expected exact-EOF special rejection")
  }
  let truncated = run_luna_input_writer(
    spec,
    b"!!!!",
    AsOrdinary,
    2,
    Truncate,
    1,
  )
  assert_eq(truncated.token_ids, [33, 33])
  assert_true(truncated.truncated)
}

///|
test "Luna input byte sizing and writer lifecycle fail closed" {
  assert_eq(LunaTokenizerWorker::required_byte_cells(128), 384UL)
  try ignore(LunaTokenizerWorker::required_byte_cells(0)) catch {
    LunaInvalidWorkerCapacity => ()
    _ => fail("unexpected Luna byte-cell error")
  } noraise {
    _ => fail("invalid Luna byte capacity was accepted")
  }
  let worker = LunaTokenizerWorker::new(
    luna_work_corpus_spec(),
    input_capacity=8,
    output_capacity=8,
    LunaTokenizerStepBudget::new(1),
  )
  let incomplete = worker.begin_luna_input(
    byte_count=2,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  incomplete.push_byte(b'a')
  try ignore(incomplete.finish()) catch {
    LunaInvalidInputWrite => ()
    _ => fail("unexpected incomplete-input error")
  } noraise {
    _ => fail("incomplete Luna input was accepted")
  }
  incomplete.push_byte(b'b')
  try incomplete.push_byte(b'c') catch {
    LunaInvalidInputWrite => ()
    _ => fail("unexpected input-overflow error")
  } noraise {
    _ => fail("over-declared Luna input was accepted")
  }
  let work = incomplete.finish()
  try incomplete.abort() catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected transferred-writer error")
  } noraise {
    _ => fail("transferred writer retained authority")
  }
  work.abort()
  let replacement = worker.begin_luna_input(
    byte_count=0,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  replacement.abort()
  try replacement.abort() catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected stale writer error")
  } noraise {
    _ => fail("aborted writer retained authority")
  }
}

///|
test "Luna input begin precedence is Busy output length then epoch" {
  let spec = luna_work_corpus_spec()
  let worker = LunaTokenizerWorker::new(
    spec,
    input_capacity=4,
    output_capacity=8,
    LunaTokenizerStepBudget::new(1),
  )
  let live = worker.begin_luna_input(
    byte_count=1,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  try
    ignore(
      worker.begin_luna_input(
        byte_count=5,
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    LunaTokenizerWorkerBusy => ()
    _ => fail("Busy must precede request validation")
  } noraise {
    _ => fail("busy worker accepted another writer")
  }
  live.push_byte(b'a')
  live.abort()
  try
    ignore(
      worker.begin_luna_input(
        byte_count=5,
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InvalidOutputLimit(requested=0, maximum=8) => ()
    _ => fail("output limit must precede byte length")
  } noraise {
    _ => fail("combined-invalid Luna input was accepted")
  }
  try
    ignore(
      worker.begin_bytes(
        b"12345",
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InvalidOutputLimit(requested=0, maximum=8) => ()
    _ => fail("begin_bytes precedence drifted")
  } noraise {
    _ => fail("combined-invalid begin_bytes was accepted")
  }
  try
    ignore(
      worker.begin_luna_input(
        byte_count=-1,
        special_policy=AsOrdinary,
        output_limit=8,
        truncation=RejectOverflow,
      ),
    )
  catch {
    LunaInvalidInputWrite => ()
    _ => fail("unexpected negative byte-count error")
  } noraise {
    _ => fail("negative byte count was accepted")
  }
  try
    ignore(
      worker.begin_luna_input(
        byte_count=5,
        special_policy=AsOrdinary,
        output_limit=8,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InputTooLong(actual=5, maximum=4) => ()
    _ => fail("unexpected declared-length error")
  } noraise {
    _ => fail("oversized declared input was accepted")
  }
  try
    ignore(
      spec.encode_bytes(
        b"x".repeat(129),
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InvalidOutputLimit(requested=0, maximum=128) => ()
    _ => fail("encode_bytes output-before-length precedence drifted")
  } noraise {
    _ => fail("combined-invalid encode_bytes was accepted")
  }
  worker.epoch = 0xffffffffffffffffUL
  try
    ignore(
      worker.begin_luna_input(
        byte_count=5,
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    InvalidOutputLimit(requested=0, maximum=8) => ()
    _ => fail("validation must precede epoch exhaustion")
  } noraise {
    _ => fail("combined-invalid exhausted begin was accepted")
  }
  assert_eq(worker.phase, LUNA_WORKER_IDLE)
  assert_eq(worker.input_length, 0)
  try
    ignore(
      worker.begin_luna_input(
        byte_count=1,
        special_policy=AsOrdinary,
        output_limit=8,
        truncation=RejectOverflow,
      ),
    )
  catch {
    LunaTokenizerWorkerEpochExhausted => ()
    _ => fail("unexpected epoch-exhaustion error")
  } noraise {
    _ => fail("exhausted worker accepted a writer")
  }
  assert_eq(worker.phase, LUNA_WORKER_IDLE)
  assert_eq(worker.input_length, 0)
}
