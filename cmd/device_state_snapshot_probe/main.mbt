///|
fn expect_busy(operation : () -> Unit raise) -> Unit raise {
  try operation() catch {
    @device.DeviceError::Busy(_) => ()
    _ => fail("expected busy ownership boundary")
  } noraise {
    _ => fail("live state transaction released")
  }
}

///|
fn expect_invalid(operation : () -> Unit raise) -> Unit raise {
  try operation() catch {
    @device.DeviceError::InvalidArgument(_) => ()
    _ => fail("wrong invalid-state error")
  } noraise {
    _ => fail("invalid transaction accepted")
  }
}

///|
fn main raise {
  let context = @device.open_context(0)
  let stream = context.create_stream()
  let sizes = [128L, 8L, 20L, 256L]
  let buffers = sizes.map(bytes => context.allocate(bytes))
  let initial = sizes.mapi((i, bytes) => {
    FixedArray::make(bytes.to_int(), (17 + i).to_byte())
  })
  let changed = sizes.mapi((i, bytes) => {
    FixedArray::make(bytes.to_int(), (101 + i).to_byte())
  })
  let result = sizes.map(bytes => FixedArray::make(bytes.to_int(), b'\x00'))
  for i in 0..<buffers.length() {
    context.copy_from_fixed_host(buffers[i], initial[i], 0L, 0L, sizes[i])
  }
  try
    @snapshot.prepare_device_state_snapshot(
      context,
      stream,
      buffers,
      maximum_backup_bytes=411L,
    )
  catch {
    @plan.SnapshotBudgetExceeded => ()
    _ => fail("wrong backup budget error")
  } noraise {
    _ => fail("one-byte-short snapshot budget accepted")
  }
  let snapshot = @snapshot.prepare_device_state_snapshot(
    context,
    stream,
    buffers,
    maximum_backup_bytes=412L,
  )
  assert_eq(snapshot.backup_bytes(), 412L)
  expect_invalid(() => snapshot.commit())
  expect_invalid(() => snapshot.rollback())
  expect_busy(() => buffers[0].close())
  expect_busy(() => stream.close())
  for _ in 0..<128 {
    snapshot.begin()
    expect_invalid(() => snapshot.begin())
    expect_busy(() => snapshot.close())
    snapshot.wait_completion()
    for i in 0..<buffers.length() {
      context.copy_from_fixed_host(buffers[i], changed[i], 0L, 0L, sizes[i])
    }
    expect_busy(() => snapshot.close())
    snapshot.rollback()
    expect_busy(() => snapshot.close())
    snapshot.wait_completion()
    for i in 0..<buffers.length() {
      context.copy_to_fixed_host(buffers[i], result[i], 0L, 0L, sizes[i])
      assert_eq(result[i], initial[i])
    }
  }
  // Commit retains the accepted mutations rather than replaying the old backup.
  snapshot.begin()
  snapshot.wait_completion()
  for i in 0..<buffers.length() {
    context.copy_from_fixed_host(buffers[i], changed[i], 0L, 0L, sizes[i])
  }
  snapshot.commit()
  snapshot.begin()
  while !snapshot.poll_completion() {
    ()
  }
  for i in 0..<buffers.length() {
    context.copy_from_fixed_host(buffers[i], initial[i], 0L, 0L, sizes[i])
  }
  snapshot.rollback()
  while !snapshot.poll_completion() {
    ()
  }
  for i in 0..<buffers.length() {
    context.copy_to_fixed_host(buffers[i], result[i], 0L, 0L, sizes[i])
    assert_eq(result[i], changed[i])
  }
  snapshot.close()
  snapshot.close()
  for buffer in buffers.rev() {
    buffer.close()
  }
  stream.close()
  context.close()
  println(
    "device_state_snapshot passed: 128 rollbacks; commit/poll/budget/ownership; backup_bytes=412",
  )
}
