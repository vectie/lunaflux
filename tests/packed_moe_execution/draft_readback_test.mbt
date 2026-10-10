///|
test "prepared draft readback retires real live tokens without hot allocations or blocking sync" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let source = context.allocate(12L)
  let readback = @draft_readback.DraftTokenReadback::new(
    context,
    source,
    token_count=3,
    vocabulary_size=16,
    maximum_host_bytes=12L,
  )
  let payload : FixedArray[Byte] = [4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]
  context.copy_from_fixed_host(source, payload, 0L, 0L, 12L)
  let waits = blocking_syncs()
  begin_allocations()
  for _ in 0..<256 {
    readback.begin()
    while !readback.poll() {
      ()
    }
    assert_eq(readback.token_count(), 3)
    assert_eq(readback.token_at(0), 4)
    assert_eq(readback.token_at(1), 5)
    assert_eq(readback.token_at(2), 6)
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(blocking_syncs(), waits)
  try readback.token_at(3) catch {
    _ => ()
  } noraise {
    _ => fail("draft readback returned a padded/outside token")
  }
  try source.close() catch {
    @device.DeviceError::Busy(_) => ()
    error => raise error
  } noraise {
    _ => fail("readback source lease was not retained")
  }
  // Any invalid later sample invalidates the whole dependent proposal.
  payload[8] = 16
  context.copy_from_fixed_host(source, payload, 0L, 0L, 12L)
  readback.begin()
  try {
    while !readback.poll() {
      ()
    }
  } catch {
    _ => ()
  } noraise {
    _ => fail("invalid later draft sample was admitted")
  }
  try readback.token_at(0) catch {
    _ => ()
  } noraise {
    _ => fail("failed readback exposed a partial draft")
  }
  readback.close()
  source.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "pending draft readback drains on cancellation before releasing its source" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let source = context.allocate(12L)
  let readback = @draft_readback.DraftTokenReadback::new(
    context,
    source,
    token_count=3,
    vocabulary_size=16,
    maximum_host_bytes=12L,
  )
  readback.begin()
  try readback.token_count() catch {
    _ => ()
  } noraise {
    _ => fail("pending draft transfer exposed output")
  }
  readback.close()
  source.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
