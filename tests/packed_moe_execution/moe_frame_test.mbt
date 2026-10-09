///|
test "complete local router/expert frame joins a caller queue without hot allocations" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let stream = context.create_stream()
  defer (stream.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let (program, routed_bytes, shared_bytes) = routed_program_fixture(
    complete_local=true,
  )
  defer (program.close() catch { _ => () })
  let external = [20L, 128L, 256L, 16L, routed_bytes, shared_bytes, 128L].map(bytes => {
    context.allocate(bytes)
  })
  defer (for allocation in external.rev() {
    allocation.close() catch {
      _ => ()
    }
  })
  program.prepare_routed_frame(
    context,
    aot,
    maximum_workspace_bytes=program.workspace_bytes(),
    counts=external[0],
    input=external[1],
    router_weight=external[2],
    correction_bias=external[3],
    routed_bank=external[4],
    shared_bank=external[5],
    output=external[6],
  )
  let launches = program.launches()
  assert_eq(launches.length(), 10)
  let queue = context.create_ordered_kernel_executor(stream, launches)
  defer (queue.abort() catch { _ => () })
  defer (queue.close() catch { _ => () })
  let before = enqueues()
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<32 {
    for i in 0..<queue.kernel_count() {
      queue.enqueue(i)
    }
    queue.record_completion()
    while !queue.poll_completion() {
      ()
    }
    queue.reset()
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(enqueues() - before, 320UL)
  assert_eq(blocking_syncs(), waits)
  queue.enqueue(0)
  queue.abort()
  queue.close()
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
test "queue-only composition does not drop missing expert-rank contributions" {
  let context = @device.open_context(0)
  defer context.close()
  let aot = context.load_module(b"fake\x00")
  defer aot.close()
  let buffer = context.allocate(20L)
  defer buffer.close()
  let (program, _, _) = routed_program_fixture()
  defer program.close()
  let baseline = live_resources()
  try
    program.prepare_routed_frame(
      context,
      aot,
      maximum_workspace_bytes=program.workspace_bytes(),
      counts=buffer,
      input=buffer,
      router_weight=buffer,
      correction_bias=buffer,
      routed_bank=buffer,
      shared_bank=buffer,
      output=buffer,
    )
  catch {
    @device.DeviceError::InvalidArgument(_) => ()
    e => raise e
  } noraise {
    _ => fail("partitioned banks require their collective-aware execution")
  }
  assert_eq(live_resources(), baseline)
}
