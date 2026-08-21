///|
fn luna_expect_stale_write(write : LunaOnlineTcpOutputWrite) -> Unit raise {
  try write.write_byte(0, b'X') catch {
    LunaOnlineTcpScratchStale => ()
    _ => fail("retained write produced the wrong error")
  } noraise {
    _ => fail("retained write mutated outside its generation and phase")
  }
}

///|
fn luna_expect_stale_flight(flight : LunaOnlineTcpOutputFlight) -> Unit raise {
  try flight.byte_length() catch {
    LunaOnlineTcpScratchStale => ()
    _ => fail("retained flight produced the wrong error")
  } noraise {
    _ => fail("retained flight remained live")
  }
  try flight.release() catch {
    LunaOnlineTcpScratchStale => ()
    _ => fail("retained flight release produced the wrong error")
  } noraise {
    _ => fail("retained flight released another phase or generation")
  }
}

///|
test "Luna online TCP scratch aliases one backing and separates mutation from flight" {
  let scratch = LunaOnlineTcpOutputScratch::new(8)
  assert_eq(scratch.byte_capacity(), 8)
  assert_eq(scratch.immutable.length(), 8)
  assert_eq(scratch.mutable.length(), 8)
  let write = scratch.begin_write()
  write.write_byte(0, b'L')
  let source = FixedArray::make(3, b'\x00')
  source[0] = b'u'
  source[1] = b'n'
  source[2] = b'a'
  write.copy_from(source, source_offset=0, destination_offset=1, length=3)
  let flight = write.publish(4)
  assert_eq(flight.byte_length(), 4)
  assert_eq(flight.byte_at(0), b'L')
  assert_eq(flight.byte_at(1), b'u')
  assert_eq(flight.byte_at(2), b'n')
  assert_eq(flight.byte_at(3), b'a')
  luna_expect_stale_write(write)
  try scratch.begin_write() catch {
    LunaOnlineTcpScratchBusy => ()
    _ => fail("in-flight scratch produced the wrong busy error")
  } noraise {
    _ => fail("scratch admitted mutation while write was in flight")
  }
  flight.release()
  assert_true(scratch.is_idle())
  luna_expect_stale_flight(flight)
}

///|
test "Luna online TCP scratch validates capacity and ranges transactionally" {
  for capacity in [0, -1, LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES + 1] {
    try LunaOnlineTcpOutputScratch::new(capacity) catch {
      LunaOnlineTcpInvalidScratchCapacity => ()
      _ => fail("invalid scratch capacity produced the wrong error")
    } noraise {
      _ => fail("invalid scratch capacity was accepted")
    }
  }
  let scratch = LunaOnlineTcpOutputScratch::new(4)
  let write = scratch.begin_write()
  try scratch.begin_write() catch {
    LunaOnlineTcpScratchBusy => ()
    _ => fail("concurrent mutation produced the wrong error")
  } noraise {
    _ => fail("concurrent mutation was admitted")
  }
  try write.write_byte(4, b'X') catch {
    LunaOnlineTcpScratchRange => ()
    _ => fail("out-of-range byte produced the wrong error")
  } noraise {
    _ => fail("out-of-range byte was written")
  }
  let source = FixedArray::make(2, b'A')
  try
    write.copy_from(source, source_offset=1, destination_offset=0, length=2)
  catch {
    LunaOnlineTcpScratchRange => ()
    _ => fail("invalid copy produced the wrong error")
  } noraise {
    _ => fail("invalid copy was accepted")
  }
  for length in [0, 5] {
    try write.publish(length) catch {
      LunaOnlineTcpScratchRange => ()
      _ => fail("invalid flight length produced the wrong error")
    } noraise {
      _ => fail("invalid flight length was accepted")
    }
  }
  write.write_byte(0, b'O')
  write.write_byte(1, b'K')
  let flight = write.publish(2)
  try flight.byte_at(2) catch {
    LunaOnlineTcpScratchRange => ()
    _ => fail("invalid flight read produced the wrong error")
  } noraise {
    _ => fail("invalid flight read was accepted")
  }
  assert_eq(flight.byte_at(0), b'O')
  assert_eq(flight.byte_at(1), b'K')
  flight.release()
}

///|
test "Luna online TCP scratch abort and reuse revoke retained aliases" {
  let scratch = LunaOnlineTcpOutputScratch::new(3)
  let aborted = scratch.begin_write()
  aborted.write_byte(0, b'A')
  aborted.abort()
  assert_true(scratch.is_idle())
  luna_expect_stale_write(aborted)

  let first_write = scratch.begin_write()
  first_write.write_byte(0, b'B')
  let first_flight = first_write.publish(1)
  first_flight.release()
  let second_write = scratch.begin_write()
  second_write.write_byte(0, b'C')
  let second_flight = second_write.publish(1)
  luna_expect_stale_write(first_write)
  luna_expect_stale_flight(first_flight)
  assert_eq(second_flight.byte_at(0), b'C')
  second_flight.release()
}

///|
test "Luna online TCP scratch refuses generation wrap without mutation" {
  let scratch = LunaOnlineTcpOutputScratch::new(1)
  scratch.generation = 0xffffffffffffffffUL
  try scratch.begin_write() catch {
    LunaOnlineTcpScratchEpochExhausted => ()
    _ => fail("exhausted scratch epoch produced the wrong error")
  } noraise {
    _ => fail("exhausted scratch epoch wrapped")
  }
  assert_true(scratch.is_idle())
  assert_eq(scratch.generation, 0xffffffffffffffffUL)
}

///|
test "Luna online TCP scratch accepts exact maximum capacity and last byte" {
  let scratch = LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES,
  )
  assert_eq(scratch.byte_capacity(), LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES)
  let write = scratch.begin_write()
  let last = LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES - 1
  write.write_byte(0, b'L')
  write.write_byte(last, b'X')
  let flight = write.publish(LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES)
  assert_eq(flight.byte_at(0), b'L')
  assert_eq(flight.byte_at(last), b'X')
  flight.release()
  assert_true(scratch.is_idle())
}

///|
test "Luna online TCP scratch repeatedly creates drops and reuses aliases" {
  for iteration in 0..<256 {
    let capacity = 1 + iteration % 257
    let scratch = LunaOnlineTcpOutputScratch::new(capacity)
    let first_write = scratch.begin_write()
    first_write.write_byte(0, (iteration & 0xff).to_byte())
    let first_flight = first_write.publish(1)
    assert_eq(first_flight.byte_at(0), (iteration & 0xff).to_byte())
    first_flight.release()
    let second_write = scratch.begin_write()
    second_write.write_byte(0, b'R')
    let second_flight = second_write.publish(1)
    assert_eq(second_flight.byte_at(0), b'R')
    second_flight.release()
    assert_true(scratch.is_idle())
  }
}
