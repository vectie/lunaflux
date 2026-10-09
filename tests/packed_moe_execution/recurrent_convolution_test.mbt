///|
fn convolution_fixture() -> @recurrent.RecurrentDeltaProgram raise {
  @recurrent.RecurrentDeltaProgram::with_convolution(
    @delta.recurrent_delta_source(
      @precision.RecurrentDeltaPrecision::new(
        head_count=2,
        key_dimension=4,
        value_dimension=4,
        maximum_rows=4,
        maximum_sequences=2,
        cache_slots=4,
        normalization_epsilon=0.000001,
        log_decay_lower_bound=-5.0,
        maximum_state_bytes=512L,
      ),
      symbol="delta_after_conv",
    ),
    @convolution.causal_convolution_source(
      @precision.CausalConvolutionPrecision::new(
        channel_count=8,
        kernel_size=4,
        maximum_rows=4,
        maximum_sequences=2,
        cache_slots=4,
        maximum_state_bytes=192L,
      ),
      symbol="request_conv",
    ),
  )
}

///|
test "QKV convolution and recurrent update share an allocation-free ordered queue" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let program = convolution_fixture()
  defer (program.abort() catch { _ => () })
  assert_eq(program.state_bytes(), 1088L)
  assert_eq(program.workspace_bytes(), 1280L)
  let generated = @utf8.decode(program.source_bytes())
  assert_true(generated.contains("void request_conv("))
  assert_true(generated.contains("void delta_after_conv("))
  let external = [
    20L, 12L, 8L, 8L, 64L, 64L, 64L, 128L, 16L, 64L, 64L, 64L, 64L,
  ].map(size => context.allocate(size))
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  program.prepare(
    context,
    aot,
    stream,
    maximum_state_bytes=1088L,
    maximum_workspace_bytes=1280L,
    counts=external[0],
    sequence_offsets=external[1],
    sequence_slots=external[2],
    reset_flags=external[3],
    query=external[4],
    key=external[5],
    value=external[6],
    log_decay=external[7],
    beta=external[8],
    output=external[9],
    convolution_weights=FixedArray::from_array([
      external[10],
      external[11],
      external[12],
    ]),
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
  assert_eq(enqueues() - before, 128UL)
  assert_eq(blocking_syncs(), waits)
  program.submit(live_rows=1)
  program.abort()
  program.close()
  for allocation in external.rev() {
    allocation.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "partial convolution binding releases all intermediate histories" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let input = context.allocate(128L)
  defer (input.close() catch { _ => () })
  let tiny = context.allocate(2L)
  defer (tiny.close() catch { _ => () })
  let baseline = live_resources()
  let program = convolution_fixture()
  defer (program.abort() catch { _ => () })
  try
    program.prepare(
      context,
      aot,
      stream,
      maximum_state_bytes=1088L,
      maximum_workspace_bytes=1280L,
      counts=input,
      sequence_offsets=input,
      sequence_slots=input,
      reset_flags=input,
      query=input,
      key=input,
      value=input,
      log_decay=input,
      beta=input,
      output=input,
      convolution_weights=FixedArray::from_array([input, tiny, input]),
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    e => raise e
  } noraise {
    _ => fail("undersized second stream weight bound")
  }
  program.close()
  assert_eq(live_resources(), baseline)
}

///|
test "composition requires identical request capacity and channel geometry" {
  let delta = recurrent_fixture()
  // Asymmetric key/value geometry is valid standalone, not for one shared
  // convolution source reused by all three streams.
  let source = @delta.recurrent_delta_source(
    @precision.RecurrentDeltaPrecision::new(
      head_count=2,
      key_dimension=4,
      value_dimension=6,
      maximum_rows=4,
      maximum_sequences=2,
      cache_slots=4,
      normalization_epsilon=0.000001,
      log_decay_lower_bound=-5.0,
      maximum_state_bytes=768L,
    ),
    symbol="asymmetric_delta",
  )
  let convolution = @convolution.causal_convolution_source(
    @precision.CausalConvolutionPrecision::new(
      channel_count=8,
      kernel_size=4,
      maximum_rows=4,
      maximum_sequences=2,
      cache_slots=4,
      maximum_state_bytes=192L,
    ),
    symbol="conv",
  )
  try
    @recurrent.RecurrentDeltaProgram::with_convolution(source, convolution)
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("asymmetric streams silently accepted")
  }
  delta.close()
}
