///|
fn recurrent_fixture() -> @recurrent.RecurrentDeltaProgram raise {
  @recurrent.RecurrentDeltaProgram::new(
    @delta.recurrent_delta_source(
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
      symbol="request_delta",
    ),
  )
}

///|
test "recurrent Program persists request slots without step allocations or blocking waits" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let module = context.load_module(b"fake\x00")
  defer (module.close() catch { _ => () })
  let program = recurrent_fixture()
  defer (program.abort() catch { _ => () })
  let sizes = [20L, 12L, 8L, 8L, 64L, 64L, 96L, 128L, 16L, 96L]
  let external = sizes.map(size => context.allocate(size))
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  assert_eq(program.state_bytes(), 768L)
  program.prepare(
    context,
    module,
    stream,
    maximum_state_bytes=768L,
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
  let allocated = end_allocations()
  assert_eq(allocated, 0UL)
  assert_eq(enqueues() - before, 32UL)
  assert_eq(blocking_syncs(), waits)
  program.submit(live_rows=1)
  try program.close() catch {
    @device.DeviceError::Busy(_) => ()
    e => raise e
  } noraise {
    _ => fail("active cache released")
  }
  program.abort()
  program.close()
  for allocation in external.rev() {
    allocation.close()
  }
  module.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "partial recurrent binding releases its persistent cache" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let module = context.load_module(b"fake\x00")
  defer (module.close() catch { _ => () })
  let input = context.allocate(20L)
  defer (input.close() catch { _ => () })
  let baseline = live_resources()
  let program = recurrent_fixture()
  defer (program.abort() catch { _ => () })
  try
    program.prepare(
      context,
      module,
      stream,
      maximum_state_bytes=768L,
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
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    e => raise e
  } noraise {
    _ => fail("undersized buffers bound")
  }
  program.close()
  assert_eq(live_resources(), baseline)
}
