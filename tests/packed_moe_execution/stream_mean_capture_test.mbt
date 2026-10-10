///|
test "stream capture uses parent completion without warmed allocations and closes deterministically" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let frame = @capture_frame.StreamMeanCaptureFrame::new(
    @capture_source.stream_mean_capture_source(
      @precision.StreamMeanCapturePrecision::new(
        stream_count=4,
        hidden_width=7,
        maximum_rows=3,
        segment_count=3,
        maximum_output_bytes=126L,
      ),
      symbol="capture_fixture",
    ),
  )
  defer (frame.close() catch { _ => () })
  let before = live_resources()
  try
    frame.prepare(context, aot, counts=operand, maximum_device_bytes=125L)
  catch {
    @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("ignored capture budget")
  }
  assert_eq(live_resources(), before)
  frame.prepare(context, aot, counts=operand, maximum_device_bytes=126L)
  assert_eq(frame.output().byte_count(), 126L)
  try frame.capture(frame.output(), segment=0) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("input/output alias must fail")
  }
  for segment in [-1, 3] {
    try frame.capture(operand, segment~) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      e => raise e
    } noraise {
      _ => fail("invalid segment must fail")
    }
  }
  let launches = Array::makei(3, segment => frame.capture(operand, segment~))
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.close() catch { _ => () })
  defer (queue.abort() catch { _ => () })
  let waits = blocking_syncs()
  let before = enqueues()
  begin_allocations()
  for _ in 0..<32 {
    for index in 0..<3 {
      queue.enqueue(index)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 96UL)
  assert_eq(blocking_syncs(), waits)
  queue.enqueue(0)
  queue.abort()
  queue.close()
  frame.close()
  frame.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
