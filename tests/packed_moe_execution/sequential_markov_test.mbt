///|
fn markov_fixture() -> @markov_frame.SequentialMarkovFrame raise {
  @markov_frame.SequentialMarkovFrame::new(
    @markov_source.sequential_markov_source(
      @precision.SequentialMarkovPrecision::new(
        vocabulary_size=17,
        hidden_width=8,
        embedding_width=4,
        maximum_rows=5,
        temperature=0.7,
        maximum_workspace_bytes=60L,
      ),
      symbol_prefix="markov_fixture",
    ),
  )
}

///|
test "sequential sampling borrows one queue with no warm allocation or synchronization" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let read_port = context.allocate(1024L)
  defer (read_port.close() catch { _ => () })
  let tokens = context.allocate(24L)
  defer (tokens.close() catch { _ => () })
  let logits = context.allocate(340L)
  defer (logits.close() catch { _ => () })
  let confidence = context.allocate(20L)
  defer (confidence.close() catch { _ => () })
  let frame = markov_fixture()
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  for (budget, token_port) in [(59L, tokens), (60L, read_port)] {
    try
      frame.prepare(
        context,
        aot,
        output_count=read_port,
        tokens=token_port,
        pre_normalized_hidden=read_port,
        logits~,
        embedding_weight=read_port,
        bias_weight=read_port,
        confidence_weight=read_port,
        rng=read_port,
        confidence~,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid chain prepared")
    }
    assert_eq(live_resources(), before)
  }
  frame.prepare(
    context,
    aot,
    output_count=read_port,
    tokens~,
    pre_normalized_hidden=read_port,
    logits~,
    embedding_weight=read_port,
    bias_weight=read_port,
    confidence_weight=read_port,
    rng=read_port,
    confidence~,
    maximum_workspace_bytes=60L,
  )
  assert_eq(frame.launches().length(), 20)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let calls = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<20 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - calls, 640UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  frame.close()
  tokens.close()
  logits.close()
  confidence.close()
  read_port.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
