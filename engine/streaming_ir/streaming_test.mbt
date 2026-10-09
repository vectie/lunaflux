///|
fn capability() -> @streaming_ir.Capability raise {
  @streaming_ir.Capability::new(
    slots=4,
    host_bytes=256L,
    page_bytes=64L,
    page_count=16,
    reserve_pages=4,
    restore_ns=100L,
    recompute_ns=1000L,
    separate_host_memory=true,
  )
}

///|
test "pure control rejects stale epochs replay and concurrent effects" {
  let cap = capability()
  let initial = @streaming_ir.Control::new(7UL)
  let command : @streaming_ir.Command = {
    worker: 7UL,
    sequence: 1UL,
    opcode: Spill,
    page: 3,
    page_generation: 2,
    host: { slot: -1, generation: 0, },
  }
  let started = initial.decide(command, cap)
  assert_true(started.action is BeginSpill)
  assert_true(initial.decide(command, cap).action is BeginSpill)
  assert_true(started.next.decide(command, cap).action is Reject)
  assert_true(
    started.next.decide({ ..command, opcode: Poll, }, cap).action
    is PollTransfer,
  )
  assert_true(
    started.next.decide({ ..command, opcode: Cancel, }, cap).action
    is CancelTransfer,
  )
  assert_true(
    started.next.decide({ ..command, worker: 8UL, opcode: Poll, }, cap).action
    is Reject,
  )
  assert_true(
    started.next.decide({ ..command, page_generation: 3, opcode: Poll, }, cap).action
    is Reject,
  )
  let done = started.next.completed()
  assert_true(done.decide(command, cap).action is Reject)
  assert_true(
    done.decide({ ..command, sequence: 2UL, }, cap).action is BeginSpill,
  )
}

///|
test "scalar frame roundtrip binds exact command and canonical padding" {
  let command : @streaming_ir.Command = {
    worker: 7UL,
    sequence: 9UL,
    opcode: Restore,
    page: 3,
    page_generation: 2,
    host: { slot: 1, generation: 5, },
  }
  let bytes = FixedArray::make(64, b'\xff')
  @streaming_ir.encode(
    { command, status: DeviceReady, host: command.host, },
    bytes,
  )
  let decoded = @streaming_ir.decode(bytes, 64)
  assert_true(decoded.command.same_command(command))
  assert_true(decoded.status is DeviceReady)
  assert_true(decoded.host.same(command.host))
  assert_false(decoded.command.same_command({ ..command, opcode: Poll, }))
  for offset in [4, 24, 44, 56, 63] {
    let saved = bytes[offset]
    bytes[offset] = b'\xff'
    try @streaming_ir.decode(bytes, 64) catch {
      InvalidFrame => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("accepted malformed transfer frame")
    }
    bytes[offset] = saved
  }
  try @streaming_ir.decode(bytes, 63) catch {
    InvalidFrame => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("accepted short transfer frame")
  }
}
