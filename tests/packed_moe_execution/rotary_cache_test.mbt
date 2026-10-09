///|
fn compact_cache_fixture() -> @rotary_frame.RotaryCacheFrame raise {
  @rotary_frame.RotaryCacheFrame::new(
    @rotary_source.rotary_cache_source(
      @precision.RotaryCachePrecision::new(
        head_count=1,
        head_width=16,
        rotary_width=8,
        maximum_rows=2,
        maximum_positions=1024,
        frequency=Base(theta=10000.0),
        simulation_block_width=8,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes=64L,
      ),
      symbol_prefix="compact_pool_cache",
    ),
  )
}

///|
test "pool compact counts and group positions feed two transforms in the same queue" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let pool = pool_fixture()
  defer (pool.close() catch { _ => () })
  pool.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    reset_flag=operand,
    input=operand,
    weights=[operand, operand, operand, operand],
    maximum_device_bytes=pool.required_device_bytes(),
  )
  let cache = compact_cache_fixture()
  defer (cache.close() catch { _ => () })
  cache.prepare(
    context,
    aot,
    counts=pool.output_counts(),
    positions=pool.output_positions(),
    input=pool.output(),
    maximum_workspace_bytes=64L,
  )
  let launches = pool.launches().to_owned()
  launches.append(cache.launches())
  assert_eq(launches.length(), 7)
  assert_eq(cache.output().byte_count(), 64L)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<7 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 224UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  cache.close()
  cache.close()
  pool.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "compact transform rejects short ports and memory before private allocation" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(65536L)
  defer operand.close()
  let short = context.allocate(7L)
  defer short.close()
  for
    (budget, positions, input) in [
      (63L, operand, operand),
      (64L, short, operand),
      (64L, operand, short),
    ] {
    let baseline = live_resources()
    let frame = compact_cache_fixture()
    try
      frame.prepare(
        context,
        aot,
        counts=operand,
        positions~,
        input~,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("short compact transform accepted")
    }
    frame.close()
    frame.close()
    assert_eq(live_resources(), baseline)
  }
}
