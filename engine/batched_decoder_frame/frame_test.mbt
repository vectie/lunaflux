///|
test "mixed chunks and decode keep independent stable slots and exact sampling" {
  let host = host()
  let (submitted, frame, writer) = plan([
    (10UL, 0UL, 0, [2, 3], false, true, true),
    (20UL, 0UL, 0, [4, 5, 6], false, false, false),
  ])
  host.stage(frame, writer)
  assert_eq(port(host, Counts), [2, 0, 2, 5, 0])
  assert_eq(port(host, SequenceOffsets), [0, 2, 5])
  assert_eq(port(host, SequenceSlots), [0, 1])
  assert_eq(port(host, RowSlots), [0, 0, 1, 1, 1])
  assert_eq(port(host, Positions), [0, 1, 0, 1, 2])
  assert_eq(port(host, RecurrentResets), [1, 1])
  assert_eq(port(host, SparseResets), [1, 1])
  assert_eq(port(host, SelectedRows), [1])
  assert_eq(host.output_entry(0), 0)
  let sampling = host.sampling_plan().prefill_row(0)
  assert_eq(sampling.sampling_mode_value(), 1)
  assert_eq(sampling.sampling_temperature(), 0.75)
  assert_eq(sampling.sampling_top_k_value(), 3)
  assert_eq(sampling.sampling_top_p_value(), 0.8)
  assert_eq(sampling.sampling_seed_value(), 11UL)
  assert_eq(sampling.sample_index(), 1)
  host.complete([7], execution_failed=false).validate_for(submitted)
  assert_eq(host.processed_tokens(10UL, 0UL), 2)
  assert_eq(host.processed_tokens(20UL, 0UL), 3)
  let (submitted, frame, writer) = plan([
    (20UL, 0UL, 3, [1, 2], false, true, false),
    (10UL, 0UL, 2, [7], true, false, true),
  ])
  host.stage(frame, writer)
  assert_eq(port(host, Counts), [1, 1, 2, 3, 0])
  assert_eq(port(host, SequenceSlots), [1, 0])
  assert_eq(port(host, RowSlots), [1, 1, 0])
  assert_eq(port(host, Positions), [3, 4, 2])
  assert_eq(port(host, SequencePositions), [3, 2])
  assert_eq(port(host, SparseResets), [0, 0])
  assert_eq(port(host, SelectedRows), [1, 2])
  assert_eq(host.output_entry(1), 1)
  assert_eq(host.sampling_plan().decode_row(0).sampling_seed_value(), 11UL)
  let completed = host.complete([8, 9], execution_failed=false)
  completed.validate_for(submitted)
  assert_eq(completed.entry(0).token_id(), 8)
  assert_eq(completed.entry(1).token_id(), 9)
  assert_eq(host.processed_tokens(10UL, 0UL), 3)
  assert_eq(host.processed_tokens(20UL, 0UL), 5)
  host.close()
}

///|
test "capacity rejection leaves no committed slot and request release is generation aware" {
  let host = host()
  let (_, frame, writer) = plan([
    (10UL, 0UL, 0, [1], false, false, false),
    (20UL, 0UL, 0, [1], false, false, false),
    (30UL, 0UL, 0, [1], false, false, false),
  ])
  host.stage(frame, writer)
  ignore(host.complete([], execution_failed=false))
  let (_, frame, writer) = plan([(40UL, 0UL, 0, [2], false, false, false)])
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::CapacityExceeded => ()
    e => raise e
  } noraise {
    _ => fail("overwrote a live history")
  }
  writer.abort()
  host.release_request(20UL, 0UL)
  let (_, frame, writer) = plan([(20UL, 1UL, 0, [2], false, false, false)])
  host.stage(frame, writer)
  assert_eq(port(host, SequenceSlots), [1])
  assert_eq(port(host, RecurrentResets), [1])
  assert_eq(port(host, SparseResets), [1])
  try host.release_request(20UL, 0UL) catch {
    @batched_decoder_frame.BatchFrameError::Busy => ()
    e => raise e
  } noraise {
    _ => fail("retired in-flight state")
  }
  ignore(host.complete([], execution_failed=false))
  try host.release_request(20UL, 0UL) catch {
    @batched_decoder_frame.BatchFrameError::GenerationMismatch => ()
    e => raise e
  } noraise {
    _ => fail("late cancellation freed new generation")
  }
  assert_eq(host.processed_tokens(20UL, 1UL), 1)
  assert_eq(host.processed_tokens(10UL, 0UL), 1)
  host.close()
}

///|
test "failed multi-entry staging does not reserve its otherwise valid first entry" {
  let host = host()
  let (_, frame, writer) = plan([
    (10UL, 0UL, 0, [1], false, false, false),
    (20UL, 0UL, 1, [1], false, false, false),
  ])
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::UnknownRequest => ()
    e => raise e
  } noraise {
    _ => fail("invented prefix state")
  }
  writer.abort()
  let (_, frame, writer) = plan([(10UL, 1UL, 0, [1], false, false, false)])
  host.stage(frame, writer)
  assert_eq(port(host, SequenceSlots), [0])
  assert_eq(port(host, SparseResets), [1])
  ignore(host.complete([], execution_failed=false))
  assert_eq(host.processed_tokens(10UL, 1UL), 1)
  host.close()
}

///|
test "nonfinal chunks produce no samples and exact context endpoint is accepted" {
  let host = host()
  for position in [0, 4] {
    let (_, frame, writer) = plan([
      (10UL, 0UL, position, [1, 2, 3, 4], false, false, false),
    ])
    host.stage(frame, writer)
    assert_eq(host.output_rows(), 0)
    assert_eq(port(host, SelectedRows), [])
    ignore(host.complete([], execution_failed=false))
  }
  assert_eq(host.processed_tokens(10UL, 0UL), 8)
  let (_, frame, writer) = plan([(10UL, 0UL, 8, [5], true, false, false)])
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::InvalidPosition => ()
    e => raise e
  } noraise {
    _ => fail("history overflow")
  }
  writer.abort()
  host.close()
}

///|
test "execution and invalid sample failures poison every potentially mutated slot" {
  for failed in [false, true] {
    let host = host()
    let (submitted, frame, writer) = plan([
      (10UL, 0UL, 0, [1], false, true, false),
      (20UL, 0UL, 0, [2], false, true, false),
    ])
    host.stage(frame, writer)
    host
    .complete(if failed { [] } else { [3, 16] }, execution_failed=failed)
    .validate_for(submitted)
    assert_true(host.is_faulted())
    try host.release_request(10UL, 0UL) catch {
      @batched_decoder_frame.BatchFrameError::Faulted => ()
      e => raise e
    } noraise {
      _ => fail("reused potentially mutated state")
    }
    host.close()
    host.close()
  }
}

///|
test "rank retirement commits without publishing and closed frame rejects staging" {
  let host = host()
  let (_, frame, writer) = plan([(10UL, 0UL, 0, [1], false, true, true)])
  host.stage(frame, writer)
  host.retire_stage(execution_failed=false)
  assert_eq(host.processed_tokens(10UL, 0UL), 1)
  let (_, frame, writer) = plan([(10UL, 0UL, 1, [2], true, false, true)])
  host.stage(frame, writer)
  host.retire_stage(execution_failed=true)
  assert_true(host.is_faulted())
  host.close()
  let (_, frame, writer) = plan([(20UL, 0UL, 0, [1], false, false, false)])
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::Closed => ()
    e => raise e
  } noraise {
    _ => fail("closed owner accepted work")
  }
  writer.abort()
}
