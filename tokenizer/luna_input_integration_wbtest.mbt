///|
test "streamed input owns one cross-facade epoch through abort and reuse" {
  let worker = LunaTokenizerWorker::new(
    luna_work_corpus_spec(),
    input_capacity=4,
    output_capacity=8,
    LunaTokenizerStepBudget::new(1),
  )
  let write = worker.begin_luna_input(
    byte_count=2,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  try
    ignore(
      worker.begin_bytes(
        b"abcde",
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    LunaTokenizerWorkerBusy => ()
    _ => fail("streamed writer did not exclude compatibility begin")
  } noraise {
    _ => fail("compatibility begin replaced a live streamed writer")
  }
  write.push_byte(b'a')
  try ignore(write.finish()) catch {
    LunaInvalidInputWrite => ()
    _ => fail("unexpected retained under-write error")
  } noraise {
    _ => fail("under-written input transferred authority")
  }
  write.push_byte(b'b')
  try write.push_byte(b'c') catch {
    LunaInvalidInputWrite => ()
    _ => fail("unexpected retained over-write error")
  } noraise {
    _ => fail("over-write changed the declared input")
  }
  let work = write.finish()
  try
    ignore(
      worker.begin_luna_input(
        byte_count=-1,
        special_policy=AsOrdinary,
        output_limit=0,
        truncation=RejectOverflow,
      ),
    )
  catch {
    LunaTokenizerWorkerBusy => ()
    _ => fail("live work did not retain Busy precedence")
  } noraise {
    _ => fail("streamed begin replaced live tokenizer work")
  }
  try write.push_byte(b'c') catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("transferred writer did not become stale")
  } noraise {
    _ => fail("transferred writer retained byte authority")
  }
  work.abort()
  let replacement = worker.begin_luna_input(
    byte_count=1,
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  try write.abort() catch {
    LunaTokenizerWorkerStale => ()
    _ => fail("reused worker authenticated an old writer")
  } noraise {
    _ => fail("old writer aborted a reused worker")
  }
  replacement.push_byte(b'c')
  replacement.abort()
}

///|
test "streamed input rejects invalid counts without mutating idle authority" {
  let worker = LunaTokenizerWorker::new(
    luna_work_corpus_spec(),
    input_capacity=4,
    output_capacity=8,
    LunaTokenizerStepBudget::new(1),
  )
  let epoch = worker.epoch
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
    _ => fail("unexpected negative streamed-input count error")
  } noraise {
    _ => fail("negative streamed-input count was accepted")
  }
  assert_eq(worker.phase, LUNA_WORKER_IDLE)
  assert_eq(worker.epoch, epoch)
  assert_eq(worker.input_length, 0)
  assert_eq(worker.written_input_bytes, 0)
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
    _ => fail("unexpected oversized streamed-input count error")
  } noraise {
    _ => fail("oversized streamed-input count was accepted")
  }
  assert_eq(worker.phase, LUNA_WORKER_IDLE)
  assert_eq(worker.epoch, epoch)
  assert_eq(worker.input_length, 0)
  assert_eq(worker.written_input_bytes, 0)
}
