///|
fn rotary_kv_fixture() -> @rotary_frame.RotaryKvFrame raise {
  @rotary_frame.RotaryKvFrame::new(
    @rotary_source.rotary_kv_source(
      @precision.RotaryKvPrecision::new(
        query_heads=2,
        key_value_heads=1,
        head_width=128,
        rotary_width=64,
        maximum_rows=3,
        maximum_positions=1048576,
        frequency=Base(theta=10000.0),
        simulation_block_width=64,
        simulation_amax_floor=0.0001,
        simulation_scale=PowerOfTwo,
        maximum_workspace_bytes=2304L,
      ),
      symbol_prefix="rotary_fixture",
    ),
  )
}

///|
test "projection rotary and KV simulation share one completion and allocate nothing warm" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(65536L)
  defer (operand.close() catch { _ => () })
  let prefix = query_kv_fixture()
  defer (prefix.close() catch { _ => () })
  prefix.prepare(
    context,
    aot,
    counts=operand,
    hidden=operand,
    weights=[operand, operand, operand, operand, operand],
    maximum_workspace_bytes=6144L,
  )
  let input = prefix.outputs()
  let rotary = rotary_kv_fixture()
  defer (rotary.close() catch { _ => () })
  rotary.prepare(
    context,
    aot,
    counts=operand,
    positions=operand,
    queries=input.queries,
    shared_kv=input.shared_kv,
    maximum_workspace_bytes=2304L,
  )
  let launches = prefix.launches().to_owned()
  launches.append(rotary.launches())
  assert_eq(launches.length(), 8)
  assert_eq(rotary.outputs().queries.byte_count(), 1536L)
  assert_eq(rotary.outputs().shared_kv.byte_count(), 768L)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let submitted = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for stage in 0..<8 {
      queue.enqueue(stage)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - submitted, 256UL)
  assert_eq(blocking_syncs(), waits)
  queue.close()
  rotary.close()
  rotary.close()
  prefix.close()
  operand.close()
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "rotary rejects short positions and aggregate workspace before scratch allocation" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let operand = context.allocate(65536L)
  defer operand.close()
  let short = context.allocate(11L)
  defer short.close()
  for (budget, positions) in [(2303L, operand), (2304L, short)] {
    let baseline = live_resources()
    let frame = rotary_kv_fixture()
    try
      frame.prepare(
        context,
        aot,
        counts=operand,
        positions~,
        queries=operand,
        shared_kv=operand,
        maximum_workspace_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("short positions/budget accepted")
    }
    frame.close()
    frame.close()
    assert_eq(live_resources(), baseline)
  }
}
