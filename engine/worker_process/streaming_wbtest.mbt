///|
#skip("Linux descriptor-execution coverage; run with --include-skipped on Linux")
test "cooperative streaming frames traverse native process transport without graph obligations" {
  let base = test_root_bound_supervisor()
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
  let owner = {
    ..base,
    process: {
      ..base.process,
      startup: base.process.startup.with_streaming(capability),
    },
  }
  let command : @streaming_ir.Command = {
    worker: 1UL,
    sequence: 1UL,
    opcode: Spill,
    page: 0,
    page_generation: 1,
    host: { slot: -1, generation: 0, },
  }
  owner.begin_streaming_exchange(command)
  assert_true(owner.streaming_exchange_pending())
  assert_false(owner.has_exchange_capacity())
  let mut complete = false
  for _ in 0..<100000 {
    if owner.progress_streaming_exchange() is Some(reply) {
      assert_true(reply.command.same_command(command))
      assert_true(reply.status is Pending)
      complete = true
      break
    }
  }
  assert_true(complete)
  assert_true(owner.has_exchange_capacity())
  owner.close()
}
