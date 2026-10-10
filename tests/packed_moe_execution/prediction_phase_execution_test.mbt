///|
test "two prediction phase queues retire and cancel without warmed allocation" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let frame = main_fixture()
  defer (frame.close() catch { _ => () })
  frame.prepare(
    context,
    aot,
    counts=operand,
    input=operand,
    matrix=operand,
    norm=operand,
    maximum_workspace_bytes=1536L,
  )
  let queue = @prediction_execution.PredictionPhaseExecution::prepare(
    context,
    stream,
    frame.launches()[:1],
    frame.launches(),
  )
  defer (queue.close() catch { _ => () })
  let count = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    queue.submit(Prime)
    while !queue.poll() {
      ()
    }
    queue.submit(Predict)
    while !queue.poll() {
      ()
    }
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - count, 96UL)
  assert_eq(blocking_syncs(), waits)
  queue.submit(Predict)
  try queue.submit(Prime) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("overlapping phase accepted")
  }
  queue.abort()
  queue.submit(Prime)
  while !queue.poll() {
    ()
  }
  queue.close()
  queue.close()
  frame.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
