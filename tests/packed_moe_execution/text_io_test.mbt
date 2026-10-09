///|
fn text_frame_fixture() -> @text_frame.TextIoFrame raise {
  @text_frame.TextIoFrame::new(
    @text_source.text_io_source(
      @precision.TextIoPrecision::new(
        vocabulary_size=11,
        hidden_width=6,
        stream_count=1,
        maximum_rows=4,
        maximum_output_rows=2,
        normalization_epsilon=1.0e-5,
        maximum_workspace_bytes=112L,
      ),
      symbol_prefix="text_fixture",
    ),
  )
}

///|
test "text IO and recurrent decoder share one completion and no step allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let input = context.allocate(256L)
  defer (input.close() catch { _ => () })
  let output = context.allocate(256L)
  defer (output.close() catch { _ => () })
  let io = text_frame_fixture()
  defer (io.close() catch { _ => () })
  let decoder = block_fixture()
  defer (decoder.abort() catch { _ => () })
  io.prepare(
    context,
    aot,
    counts=operand,
    token_ids=operand,
    output_count=operand,
    selected_rows=operand,
    input_residual=input,
    output_residual=output,
    embedding=operand,
    norm_weight=operand,
    head_weight=operand,
    sampled_tokens=operand,
    maximum_workspace_bytes=112L,
  )
  assert_eq(io.logits().byte_count(), 88L)
  decoder.prepare_block_frame(
    context,
    aot,
    fixture_block_weights(operand),
    maximum_state_bytes=1088L,
    maximum_workspace_bytes=1936L,
    counts=operand,
    sequence_offsets=operand,
    sequence_slots=operand,
    reset_flags=operand,
    hidden=input,
    output~,
  )
  let launches = io.prefix().to_owned()
  launches.append(decoder.launches())
  launches.append(io.suffix())
  assert_eq(launches.length(), 20)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
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
  assert_eq(enqueues() - before, 640UL)
  assert_eq(blocking_syncs(), waits)
  for stage in 0..<20 {
    queue.enqueue(stage)
  }
  queue.record_completion()
  queue.abort()
  queue.close()
  decoder.close()
  io.close()
  io.close()
  output.close()
  input.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "text IO budget failure acquires no private device storage" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let io = text_frame_fixture()
  defer io.close()
  let baseline = live_resources()
  try
    io.prepare(
      context,
      aot,
      counts=operand,
      token_ids=operand,
      output_count=operand,
      selected_rows=operand,
      input_residual=operand,
      output_residual=operand,
      embedding=operand,
      norm_weight=operand,
      head_weight=operand,
      sampled_tokens=operand,
      maximum_workspace_bytes=111L,
    )
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("text workspace ceiling ignored")
  }
  assert_eq(live_resources(), baseline)
}
