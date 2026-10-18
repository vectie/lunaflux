///|
fn batch_rank_fixture() -> @batch_worker.BatchedDecoderRank raise {
  @batch_worker.BatchedDecoderRank::new(
    @batch_frame.BatchedDecoderFrame::new(
      maximum_rows=4,
      maximum_sequences=2,
      cache_slots=2,
      maximum_context=16,
      vocabulary_size=16,
      model_generation=9UL,
    ),
    outputs_token=true,
  )
}

///|
test "batched rank retires DMA and commits ownership without warm allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let rank = batch_rank_fixture()
  defer (rank.close() catch { _ => () })
  rank.prepare_ports(context, maximum_device_bytes=rank.required_device_bytes())
  context.copy_from_fixed_host(
    rank.sampled_tokens(),
    [7, 0, 0, 0, 8, 0, 0, 0],
    0L,
    0L,
    8L,
  )
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (queue, function) = serial_queue(context, aot, stream, rank.input(Counts))
  defer (function.close() catch { _ => () })
  defer (rank.close() catch { _ => () })
  rank.bind_executor(
    context,
    submit=_ => {
      queue.enqueue(0)
      queue.record_completion()
    },
    poll=() => {
      if queue.poll_completion() {
        queue.reset()
        true
      } else {
        false
      }
    },
    close=() => {
      queue.abort()
      queue.close()
    },
    append_descriptors=[],
  )
  let plans = Array::makei(16, i => serial_wire_plan(i.to_uint64() + 1UL))
  let waits = blocking_syncs()
  begin_allocations()
  for (_, frame, writer) in plans {
    rank.begin(frame, writer)
    while !rank.inputs_ready() {

    }
    rank.submit()
    while !rank.progress() {

    }
    assert_false(rank.execution_failed())
    assert_eq(rank.samples().length(), 1)
    assert_eq(rank.samples()[0], 7)
    assert_eq(rank.output_entry(0), 0)
    rank.commit(pipeline_failed=false)
    rank.release_request(
      frame.prefill_row(0).request_id_value(),
      frame.prefill_row(0).request_generation_value(),
    )
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(blocking_syncs(), waits)
  let (_, frame, writer) = serial_wire_plan(99UL)
  rank.begin(frame, writer)
  rank.close()
  rank.close()
  try writer.submit() catch {
    _ => ()
  } noraise {
    _ => fail("closed rank retained completion lease")
  }
  function.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "batched rank rejects budget before allocation and poisons peer failure" {
  let context = @device.open_context(0)
  let rank = batch_rank_fixture()
  let before = live_resources()
  try
    rank.prepare_ports(
      context,
      maximum_device_bytes=rank.required_device_bytes() - 1L,
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    error => raise error
  } noraise {
    _ => fail("ignored request-port budget")
  }
  assert_eq(live_resources(), before)
  rank.prepare_ports(context, maximum_device_bytes=rank.required_device_bytes())
  context.copy_from_fixed_host(rank.sampled_tokens(), [7, 0, 0, 0], 0L, 0L, 4L)
  rank.bind_executor(context, submit=_ => (), poll=() => true, close=() => (), append_descriptors=[])
  let (_, frame, writer) = serial_wire_plan(1UL)
  rank.begin(frame, writer)
  while !rank.inputs_ready() {

  }
  rank.submit()
  while !rank.progress() {

  }
  rank.commit(pipeline_failed=true)
  try rank.release_request(1UL, 1UL) catch {
    @batch_frame.BatchFrameError::Busy => ()
    error => raise error
  } noraise {
    _ => fail("reused state after peer failure")
  }
  rank.close()
  context.close()
}
