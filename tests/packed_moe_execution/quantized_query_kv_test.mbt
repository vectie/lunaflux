///|
fn query_kv_fixture() -> @query_frame.QuantizedQueryKvFrame raise {
  @query_frame.QuantizedQueryKvFrame::new(
    @query_source.quantized_query_kv_source(
      @precision.QuantizedQueryKvPrecision::new(
        hidden_width=256,
        query_rank=128,
        head_count=2,
        head_width=128,
        maximum_rows=3,
        normalization_epsilon=1.0e-6,
        activation_amax_floor=0.0001,
        maximum_workspace_bytes=6144L,
      ),
      symbol_prefix="query_fixture",
    ),
  )
}

///|
test "quantized query prefix binds packed planes and shares one decoder completion" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let frame = query_kv_fixture()
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    counts=operand,
    hidden=operand,
    weights=[operand, operand, operand, operand, operand],
    maximum_workspace_bytes=6144L,
  )
  assert_eq(frame.launches().length(), 6)
  let outputs = frame.outputs()
  assert_eq(outputs.low_rank_queries.byte_count(), 768L)
  assert_eq(outputs.queries.byte_count(), 1536L)
  assert_eq(outputs.shared_kv.byte_count(), 768L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let submitted = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<6 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - submitted, 192UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  frame.close()
  frame.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "quantized query prefix refuses short packed scale planes before allocating" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(65536L)
  defer operand.close()
  let short = context.allocate(32769L)
  defer short.close()
  for (budget, matrix) in [(6143L, operand), (6144L, short)] {
    let baseline = live_resources()
    let frame = query_kv_fixture()
    try
      frame.prepare(
        context,
        aot,
        counts=operand,
        hidden=operand,
        weights=[matrix, operand, operand, operand, operand],
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("short scale plane/budget accepted")
    }
    frame.close()
    frame.close()
    assert_eq(live_resources(), baseline)
  }
}
