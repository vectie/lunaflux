///|
fn streaming_exchange(
  process : @worker_process.RootBoundWorkerProcessSupervisor,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise {
  process.begin_streaming_exchange(command)
  assert_false(process.has_exchange_capacity())
  for _ in 0..<1000000 {
    if process.progress_streaming_exchange() is Some(reply) {
      return reply
    }
  }
  raise Failure("streaming exchange failed to make progress")
}

///|
fn run_streaming_exchange(executable : Bytes) -> Unit raise {
  let capability = @streaming_ir.Capability::new(
    slots=2,
    host_bytes=128L,
    page_bytes=64L,
    page_count=8,
    reserve_pages=2,
    restore_ns=100L,
    recompute_ns=1000L,
    separate_host_memory=true,
  )
  let process = start_worker(
    executable,
    startup_contract().with_streaming(capability),
  ) catch {
    error => raise Failure("streaming startup: \{Repr(error)}")
  }
  let spill : @streaming_ir.Command = {
    worker: 1UL,
    sequence: 1UL,
    opcode: Spill,
    page: 0,
    page_generation: 1,
    host: { slot: -1, generation: 0, },
  }
  assert_true(streaming_exchange(process, spill).status is Pending)
  let stored = streaming_exchange(process, { ..spill, opcode: Poll, })
  assert_true(stored.status is HostReady)
  let restore : @streaming_ir.Command = {
    ..spill,
    sequence: 2UL,
    opcode: Restore,
    host: stored.host,
  }
  assert_true(streaming_exchange(process, restore).status is Pending)
  assert_true(
    streaming_exchange(process, { ..restore, opcode: Cancel, }).status
    is Cancelled,
  )
  let second = { ..restore, sequence: 3UL, }
  assert_true(streaming_exchange(process, second).status is Pending)
  assert_true(
    streaming_exchange(process, { ..second, opcode: Poll, }).status
    is DeviceReady,
  )
  assert_true(
    streaming_exchange(process, { ..second, sequence: 4UL, opcode: Evict, }).status
    is Evicted,
  )
  assert_true(process.has_exchange_capacity())
  process.begin_recovery()
  assert_true(process.restart() is RootBoundRestarted)
  assert_eq(process.startup_contract().streaming_epoch(), 2UL)
  try process.begin_streaming_exchange(spill) catch {
    _ => ()
  } noraise {
    _ => fail("old worker command accepted")
  }
  assert_true(
    streaming_exchange(process, { ..spill, worker: 2UL, }).status is Pending,
  )
  assert_true(
    streaming_exchange(process, { ..spill, worker: 2UL, opcode: Poll, }).status
    is HostReady,
  )
  process.shutdown_clean()
  process.close()
  println(
    "Streaming startup, cooperative transfer, cancellation and replacement passed",
  )
}
