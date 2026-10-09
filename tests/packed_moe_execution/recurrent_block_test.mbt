///|
fn block_fixture() -> @recurrent.RecurrentDeltaProgram raise {
  let delta = @precision.RecurrentDeltaPrecision::new(
    head_count=2,
    key_dimension=4,
    value_dimension=4,
    maximum_rows=4,
    maximum_sequences=2,
    cache_slots=4,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=512L,
  )
  let convolution = @precision.CausalConvolutionPrecision::new(
    channel_count=8,
    kernel_size=4,
    maximum_rows=4,
    maximum_sequences=2,
    cache_slots=4,
    maximum_state_bytes=192L,
  )
  @recurrent.RecurrentDeltaProgram::with_block(
    @block.recurrent_block_source(
      @precision.RecurrentBlockPrecision::new(
        delta,
        convolution,
        hidden_width=6,
        control_rank=3,
        output_normalization_epsilon=0.000001,
      ),
      symbol_prefix="complete_recurrent",
    ),
  )
}

///|
fn fixture_block_weights(
  allocation : @device.Allocation,
) -> @recurrent.RecurrentBlockWeights {
  {
    query: allocation,
    key: allocation,
    value: allocation,
    forget_low: allocation,
    forget_expansion: allocation,
    beta: allocation,
    dt_bias: allocation,
    a_log: allocation,
    gate_low: allocation,
    gate_expansion: allocation,
    norm: allocation,
    output: allocation,
    query_conv: allocation,
    key_conv: allocation,
    value_conv: allocation,
  }
}

///|
test "complete recurrent branch executes all projections without a host handoff" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let program = block_fixture()
  defer (program.abort() catch { _ => () })
  assert_eq(program.state_bytes(), 1088L)
  assert_eq(program.workspace_bytes(), 1936L)
  let text = @utf8.decode(program.source_bytes())
  for stage in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15] {
    assert_true(
      text.contains("void complete_recurrent_" + stage.to_string() + "("),
    )
  }
  assert_false(text.contains("blockIdx.x != 0 || threadIdx.x != 0"))
  program.prepare_block(
    context,
    aot,
    stream,
    fixture_block_weights(operand),
    maximum_state_bytes=1088L,
    maximum_workspace_bytes=1936L,
    counts=operand,
    sequence_offsets=operand,
    sequence_slots=operand,
    reset_flags=operand,
    hidden=operand,
    output=operand,
  )
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    program.submit(live_rows=4)
    while !program.poll() {
      ()
    }
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 512UL)
  assert_eq(blocking_syncs(), waits)
  program.submit(live_rows=1)
  program.abort()
  program.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "complete branch has an exact startup workspace budget" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let baseline = live_resources()
  let program = block_fixture()
  defer (program.abort() catch { _ => () })
  try
    program.prepare_block(
      context,
      aot,
      stream,
      fixture_block_weights(operand),
      maximum_state_bytes=1088L,
      maximum_workspace_bytes=1935L,
      counts=operand,
      sequence_offsets=operand,
      sequence_slots=operand,
      reset_flags=operand,
      hidden=operand,
      output=operand,
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    e => raise e
  } noraise {
    _ => fail("workspace underbudget accepted")
  }
  assert_eq(live_resources(), baseline)
  program.close()
}
