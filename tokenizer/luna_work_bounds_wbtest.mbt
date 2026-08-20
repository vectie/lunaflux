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

///|
test "Luna tokenizer scalar result reads hide mutable worker storage" {
  assert_eq(LunaTokenizerWorker::required_int_cells(128), 384UL)
  try ignore(LunaTokenizerWorker::required_int_cells(0)) catch {
    LunaInvalidWorkerCapacity => ()
    _ => fail("unexpected Luna cell-requirement error")
  } noraise {
    _ => fail("invalid Luna cell requirement was accepted")
  }
  let worker = LunaTokenizerWorker::new(
    luna_work_corpus_spec(),
    input_capacity=128,
    output_capacity=128,
    LunaTokenizerStepBudget::new(1),
  )
  let work = worker.begin_bytes(
    b"hello!",
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  let pending = work.token_status(0)
  assert_false(pending.is_ready())
  assert_eq(pending.token_id(), -1)
  while work.progress() is LunaTokenizerPending {

  }
  assert_eq(work.token_count(), 2)
  let first = work.token_status(0)
  assert_true(first.is_ready())
  assert_eq(first.token_id(), 259)
  assert_eq(work.token_status(1).token_id(), 33)
  assert_false(work.token_status(2).is_ready())
  work.abort()
  try ignore(work.token_status(0)) catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("unexpected stale Luna scalar-read error")
  } noraise {
    _ => fail("stale Luna work retained scalar-read authority")
  }
}
