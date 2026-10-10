///|
fn draft_fixture() -> @draft_frame.CommittedDraftKvFrame raise {
  let p = @precision.CommittedDraftKvPrecision::new(
    query_heads=2,
    head_width=8,
    window_tokens=4,
    maximum_main_rows=8,
    maximum_draft_rows=3,
    maximum_positions=32,
    maximum_state_bytes=184L,
  )
  @draft_frame.CommittedDraftKvFrame::new(
    @draft_source.committed_draft_kv_source(
      p,
      phase=Prime,
      symbol_prefix="prime",
    ),
    @draft_source.committed_draft_kv_source(
      p,
      phase=Predict,
      symbol_prefix="predict",
    ),
    committed=@draft_source.committed_draft_kv_source(
      p,
      phase=PredictCommitted,
      symbol_prefix="committed",
    ),
  )
}

///|
test "committed draft phases share a ring and parent completion without warmed allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let main = context.allocate(128L)
  defer (main.close() catch { _ => () })
  let draft = context.allocate(128L)
  defer (draft.close() catch { _ => () })
  let frame = draft_fixture()
  defer (frame.close() catch { _ => () })
  assert_eq(frame.persistent_state_bytes(), 72L)
  try frame.persistent_state() catch {
    @device.DeviceError::InvalidArgument(DeviceCopyCreate) => ()
    e => raise e
  } noraise {
    _ => fail("fresh ring exposed as prepared state")
  }
  frame.prepare(
    context,
    aot,
    main_counts=main,
    draft_counts=draft,
    reset_flag=main,
    start_position=draft,
    main_kv=main,
    queries=draft,
    draft_kv=draft,
    attention_sink=main,
    maximum_state_bytes=184L,
  )
  assert_eq(frame.prime_launches().length(), 3)
  assert_eq(frame.prediction_launches().length(), 4)
  assert_eq(frame.prediction_from_committed_launches().length(), 2)
  assert_eq(frame.output().byte_count(), 96L)
  let state = frame.persistent_state()
  assert_eq(state.length(), 2)
  assert_eq(state.map(b => b.byte_count()), [64L, 8L])
  for buffer in state {
    assert_false(buffer.same_storage(frame.output()))
    assert_false(buffer.same_storage(frame.descriptor()))
  }
  let prime = context.create_ordered_kernel_executor(
    stream,
    frame.prime_launches(),
  )
  defer (prime.close() catch { _ => () })
  defer (prime.abort() catch { _ => () })
  let predict = context.create_ordered_kernel_executor(
    stream,
    frame.prediction_launches(),
  )
  defer (predict.close() catch { _ => () })
  defer (predict.abort() catch { _ => () })
  let count = enqueues()
  let committed = context.create_ordered_kernel_executor(
    stream,
    frame.prediction_from_committed_launches(),
  )
  defer (committed.close() catch { _ => () })
  defer (committed.abort() catch { _ => () })
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<16 {
    for i in 0..<3 {
      prime.enqueue(i)
    }
    prime.record_completion()
    while !prime.poll_completion() {
      ()
    }
    prime.reset()
    for i in 0..<4 {
      predict.enqueue(i)
    }
    predict.record_completion()
    while !predict.poll_completion() {
      ()
    }
    predict.reset()
    for i in 0..<2 {
      committed.enqueue(i)
    }
    committed.record_completion()
    while !committed.poll_completion() {
      ()
    }
    committed.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - count, 144UL)
  assert_eq(blocking_syncs(), waits)
  predict.enqueue(0)
  predict.abort()
  predict.close()
  committed.close()
  prime.close()
  frame.close()
  frame.close()
  try frame.persistent_state() catch {
    @device.DeviceError::InvalidArgument(DeviceCopyCreate) => ()
    e => raise e
  } noraise {
    _ => fail("closed ring exposed as prepared state")
  }
  draft.close()
  main.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "committed draft preflight rejects count alias and budget before allocation" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let main = context.allocate(128L)
  defer main.close()
  let draft = context.allocate(128L)
  defer draft.close()
  for (counts, budget) in [(main, 184L), (draft, 183L)] {
    let frame = draft_fixture()
    defer frame.close()
    let before = live_resources()
    try
      frame.prepare(
        context,
        aot,
        main_counts=main,
        draft_counts=counts,
        reset_flag=main,
        start_position=draft,
        main_kv=main,
        queries=draft,
        draft_kv=draft,
        attention_sink=main,
        maximum_state_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid ports accepted")
    }
    assert_eq(live_resources(), before)
  }
}
