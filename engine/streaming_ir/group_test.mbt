///|
fn group_command(opcode : @streaming_ir.Opcode) -> @streaming_ir.Command {
  {
    worker: 3UL,
    sequence: 1UL,
    opcode,
    page: 2,
    page_generation: 7,
    host: { slot: 0, generation: 1, },
  }
}

///|
test "rank barrier retains completed shards until all ranks and cancels raced restores" {
  let command = group_command(Restore)
  let initial = @streaming_ir.GroupTransfer::new(command, 3)
  let first = initial.accept(0, {
    command,
    status: DeviceReady,
    host: command.host,
  })
  assert_true(first.reply(command).status is Pending)
  assert_false(initial.rank_done(0))
  assert_true(first.rank_done(0))
  let cancelled = first.cancel()
  let second = cancelled.accept(2, {
    command: { ..command, opcode: Cancel, },
    status: Cancelled,
    host: command.host,
  })
  assert_true(second.reply(command).status is Pending)
  let last = second.accept(1, {
    command: { ..command, opcode: Poll, },
    status: DeviceReady,
    host: command.host,
  })
  assert_true(last.reply(command).status is Cancelled)
  try
    first.accept(0, { command, status: DeviceReady, host: command.host, })
  catch {
    _ => ()
  } noraise {
    _ => fail("duplicate rank admitted")
  }
  try
    first.accept(1, {
      command: { ..command, worker: 2UL, },
      status: DeviceReady,
      host: command.host,
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("stale group admitted")
  }
  try
    first.accept(1, { command, status: Cancelled, host: command.host, })
  catch {
    _ => ()
  } noraise {
    _ => fail("unsolicited cancellation admitted")
  }
}

///|
test "rank barrier rejects divergent cold payload identity and incomplete eviction" {
  let command = { ..group_command(Spill), host: { slot: -1, generation: 0, }, }
  let host : @streaming_ir.HostPage = { slot: 2, generation: 5, }
  let first = @streaming_ir.GroupTransfer::new(command, 2).accept(1, {
    command,
    status: HostReady,
    host,
  })
  assert_true(first.reply(command).status is Pending)
  try
    first.accept(0, {
      command,
      status: HostReady,
      host: { slot: 2, generation: 6, },
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("different rank slots joined")
  }
  let complete = first.accept(0, { command, status: HostReady, host, })
  assert_true(complete.reply(command).status is HostReady)
  let eviction = group_command(Evict)
  let one = @streaming_ir.GroupTransfer::new(eviction, 2).accept(0, {
    command: eviction,
    status: Evicted,
    host: eviction.host,
  })
  assert_true(one.reply(eviction).status is Pending)
  assert_true(
    one
    .accept(1, { command: eviction, status: Evicted, host: eviction.host, })
    .reply(eviction).status
    is Evicted,
  )
}

///|
test "rank budgets cannot multiply the declared pinned host ceiling" {
  let capability = @streaming_ir.Capability::new(
    slots=4,
    host_bytes=1024L,
    page_bytes=256L,
    page_count=8,
    reserve_pages=2,
    restore_ns=10L,
    recompute_ns=100L,
    separate_host_memory=true,
  )
  let shard = capability.for_rank(128L, 2)
  assert_eq(shard.host_bytes(), 512L)
  try capability.for_rank(256L, 2) catch {
    _ => ()
  } noraise {
    _ => fail("replicated shard exceeded group host budget")
  }
}
