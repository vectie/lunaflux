///|
test "verification continuation uses the committed frontier not the rejected suffix" {
  let (_, original, _) = serial_wire_plan(7UL, position=2, tokens=[4, 5, 6])
  let destination = @wire.PlanFrameBuffer::new(
    serial_limits(),
    serial_inference_limits(),
  )
  let inputs : FixedArray[Int] = [9, 10, 11]
  begin_allocations()
  for epoch in 2..<=257 {
    let next = destination.encode_verification_after(
      original,
      committed_input_rows=1,
      draft_inputs=inputs,
      sequence_value=epoch.to_uint64(),
    )
    assert_eq(next.prefill_row(0).sequence_token_start(), 3)
    assert_eq(next.prefill_row(0).request_id_value(), 7UL)
    assert_eq(next.model_generation_value(), 9UL)
    assert_eq(next.token_at(0), 9)
    assert_eq(next.token_count(), 3)
    assert_true(next.prefill_row(0).produces_token())
    assert_eq(next.page_index_at(0), original.page_index_at(0))
    assert_eq(next.page_generation_at(0), original.page_generation_at(0))
    assert_eq(next.capability_value_at(0), original.capability_value_at(0))
  }
  assert_eq(end_allocations(), 0UL)
  let current = destination.current()
  for count in [0, -1, 4] {
    try
      destination.encode_verification_after(
        original,
        committed_input_rows=count,
        draft_inputs=inputs,
        sequence_value=3UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid committed prefix admitted")
    }
  }
  for bad in [[], [32], [-1], [1, 2, 3, 4, 5]] {
    try
      destination.encode_verification_after(
        original,
        committed_input_rows=1,
        draft_inputs=bad,
        sequence_value=3UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid draft block admitted")
    }
  }
  assert_eq(current.token_at(0), 9)
  let payload = FixedArray::make(destination.capacity(), b'\x00')
  current.copy_to(payload, destination_offset=0)
  let receiver = @wire.PlanFrameBuffer::new(
    serial_limits(),
    serial_inference_limits(),
  )
  let roundtrip = receiver.load(payload, current.length())
  assert_eq(roundtrip.prefill_row(0).sequence_token_start(), 3)
  assert_true(roundtrip.prefill_row(0).produces_token())
}

///|
test "accepted-prefix replay preserves real plan bindings with zero warmed allocation" {
  let (_, original, _) = serial_wire_plan(7UL, position=2, tokens=[4, 5, 6])
  let destination = @wire.PlanFrameBuffer::new(
    serial_limits(),
    serial_inference_limits(),
  )
  begin_allocations()
  for epoch in 2..<=257 {
    let replay = destination.encode_prefix_replay(
      original,
      accepted_rows=2,
      sequence_value=epoch.to_uint64(),
    )
    assert_eq(replay.prefill_row(0).sequence_token_start(), 2)
    assert_eq(replay.prefill_row(0).request_id_value(), 7UL)
    assert_eq(replay.prefill_row(0).request_generation_value(), 0UL)
    assert_eq(replay.model_generation_value(), 9UL)
    assert_eq(replay.token_count(), 2)
    assert_eq(replay.token_at(0), 4)
    assert_eq(replay.token_at(1), 5)
    assert_false(replay.prefill_row(0).produces_token())
    assert_eq(replay.page_index_at(0), original.page_index_at(0))
    assert_eq(replay.page_generation_at(0), original.page_generation_at(0))
    assert_eq(replay.capability_value_at(0), original.capability_value_at(0))
  }
  assert_eq(end_allocations(), 0UL)
  let replay = destination.current()
  let transport = FixedArray::make(destination.capacity(), b'\x00')
  replay.copy_to(transport, destination_offset=0)
  let receiver = @wire.PlanFrameBuffer::new(
    serial_limits(),
    serial_inference_limits(),
  )
  let decoded = receiver.load(transport, replay.length())
  assert_eq(decoded.token_count(), 2)
  assert_eq(decoded.prefill_row(0).sampling_seed_value(), 1UL)
  assert_eq(decoded.prefill_row(0).sample_index(), 0)
  for count in [0, -1, 4] {
    try
      destination.encode_prefix_replay(
        original,
        accepted_rows=count,
        sequence_value=3UL,
      )
    catch {
      @wire.WorkerWireError::InvalidFrame(_) => ()
      error => raise error
    } noraise {
      _ => fail("invalid replay length admitted")
    }
  }
  // Rejected construction preserves the previous prepared replay view.
  assert_eq(replay.token_count(), 2)
  try
    destination.encode_prefix_replay(
      original,
      accepted_rows=1,
      sequence_value=1UL,
    )
  catch {
    @wire.WorkerWireError::InvalidFrame(_) => ()
    error => raise error
  } noraise {
    _ => fail("non-increasing replay sequence admitted")
  }
}
