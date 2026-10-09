///|
test "hash router computes weights on device and reuses eleven launches" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (program, routed_bytes, shared_bytes) = routed_program_fixture(
    token_hash=true,
  )
  defer (program.abort() catch { _ => () })
  assert_eq(program.workspace_bytes(), 1412L)
  let external = [20L, 128L, 256L, 8L, 48L, routed_bytes, shared_bytes, 128L].map(bytes => {
      context.allocate(bytes)
    },
  )
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  program.prepare_routed(
    context,
    aot,
    stream,
    maximum_workspace_bytes=1412L,
    counts=external[0],
    input=external[1],
    router_weight=external[2],
    token_ids=external[3],
    token_table=external[4],
    token_table_offset=24L,
    routed_bank=external[5],
    shared_bank=external[6],
    output=external[7],
  )
  let before = enqueues()
  begin_allocations()
  for sequence in 1UL..<17UL {
    program.submit(
      live_rows=if sequence % 2UL == 0UL { 2 } else { 1 },
      plan_sequence=sequence,
      global_sequence=sequence,
    )
    while !program.poll() {
      ()
    }
  }
  let allocations = end_allocations()
  assert_eq(allocations, 0UL)
  assert_eq(enqueues() - before, 176UL)
  program.close()
  for allocation in external.rev() {
    allocation.close()
  }
  aot.close()
  stream.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
