///|
fn main_fixture() -> @main_frame.RowProjectionNormFrame raise {
  @main_frame.RowProjectionNormFrame::new(
    @main_source.row_projection_norm_source(
      @precision.RowProjectionNormPrecision::new(
        input_width=256,
        output_width=128,
        maximum_rows=3,
        normalization_epsilon=1.0e-6,
        activation_amax_floor=0.0001,
        maximum_workspace_bytes=1536L,
      ),
      symbol_prefix="main_fixture",
    ),
  )
}

///|
test "prediction main prefix shares parent completion without warmed allocation" {
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
  assert_eq(frame.launches().length(), 2)
  assert_eq(frame.output().byte_count(), 768L)
  let queue = context.create_ordered_kernel_executor(stream, frame.launches())
  defer (queue.close() catch { _ => () })
  defer (queue.abort() catch { _ => () })
  let count = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    queue.enqueue(0)
    queue.enqueue(1)
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - count, 64UL)
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

///|
test "prediction main preflight rejects undersized operands and budget before allocation" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(65536L)
  defer operand.close()
  let short_matrix = context.allocate(32769L)
  defer short_matrix.close()
  let short_input = context.allocate(1535L)
  defer short_input.close()
  for
    (budget, input, matrix) in [
      (1535L, operand, operand),
      (1536L, operand, short_matrix),
      (1536L, short_input, operand),
    ] {
    let frame = main_fixture()
    defer frame.close()
    let before = live_resources()
    try
      frame.prepare(
        context,
        aot,
        counts=operand,
        input~,
        matrix~,
        norm=operand,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid operand accepted")
    }
    assert_eq(live_resources(), before)
  }
}
