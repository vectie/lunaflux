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
  assert_eq(frame.output().byte_count(), 96L)
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
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - count, 112UL)
  assert_eq(blocking_syncs(), waits)
  predict.enqueue(0)
  predict.abort()
  predict.close()
  prime.close()
  frame.close()
  frame.close()
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
