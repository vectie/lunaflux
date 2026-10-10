///|
fn learned_text_frame_fixture() -> @text_frame.LearnedTextIoFrame raise {
  @text_frame.LearnedTextIoFrame::new(
    @precision.LearnedTextIoPrecision::new(
      vocabulary_size=17,
      hidden_width=8,
      stream_count=4,
      maximum_rows=5,
      maximum_output_rows=2,
      normalization_epsilon=1.0e-5,
      control_epsilon=1.0e-6,
      control_normalization_epsilon=1.0e-5,
      maximum_workspace_bytes=248L,
    ),
    symbol_prefix="learned_fixture",
  )
}

///|
test "learned text suffix shares one completion and warmed steps allocate nothing" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(1024L)
  defer (operand.close() catch { _ => () })
  let io = learned_text_frame_fixture()
  defer (io.close() catch { _ => () })
  io.prepare(
    context,
    aot,
    counts=operand,
    output_count=operand,
    selected_rows=operand,
    output_residual=operand,
    function_weight=operand,
    base=operand,
    scale=operand,
    norm_weight=operand,
    head_weight=operand,
    sampled_tokens=operand,
    maximum_workspace_bytes=248L,
  )
  assert_eq(io.suffix().length(), 4)
  assert_eq(io.projection_suffix().length(), 3)
  assert_eq(io.pre_normalized_hidden().byte_count(), 80L)
  assert_eq(io.logits().byte_count(), 136L)
  let queue = context.create_ordered_kernel_executor(stream, io.suffix())
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<4 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 128UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  io.close()
  io.close()
  try io.pre_normalized_hidden() catch {
    _ => ()
  } noraise {
    _ => fail("closed head scratch exposed")
  }
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "learned text budget and short-control rejection retain balanced resources" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(1024L)
  defer operand.close()
  let short = context.allocate(508L)
  defer short.close()
  for (budget, function_weight) in [(247L, operand), (248L, short)] {
    let io = learned_text_frame_fixture()
    let baseline = live_resources()
    try
      io.prepare(
        context,
        aot,
        counts=operand,
        output_count=operand,
        selected_rows=operand,
        output_residual=operand,
        function_weight~,
        base=operand,
        scale=operand,
        norm_weight=operand,
        head_weight=operand,
        sampled_tokens=operand,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid learned-head capacity accepted")
    }
    io.close()
    io.close()
    assert_eq(live_resources(), baseline)
  }
}
