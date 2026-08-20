///|
fn luna_storage_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=16,
    max_token_id=99,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature=4.0,
  )
}

///|
test "owned token buffers retain authenticated scalar reads" {
  let source = [1, 2, 3]
  let buffer = @inference.TokenBuffer::new(source, luna_storage_limits())
  source[0] = 90
  assert_true(buffer.is_live())
  assert_eq(buffer.length(), 3)
  assert_eq(buffer.token_at(0), 1)
  let first = buffer.token_status(0)
  assert_true(first.is_ready())
  assert_false(first.is_stale())
  assert_eq(first.token_id(), 1)
  let invalid = buffer.token_status(3)
  assert_false(invalid.is_ready())
  assert_false(invalid.is_stale())
  assert_eq(invalid.token_id(), -1)
}

///|
test "owned token buffers compare the actual cached maximum" {
  let buffer = @inference.TokenBuffer::new([1, 3, 2], luna_storage_limits())
  let exact = buffer.maximum_token_status(3)
  assert_true(exact.is_within_bound())
  assert_false(exact.is_out_of_range())
  assert_false(exact.is_stale())
  let below = buffer.maximum_token_status(2)
  assert_false(below.is_within_bound())
  assert_true(below.is_out_of_range())
  assert_false(below.is_stale())
  assert_true(buffer.maximum_token_status(99).is_within_bound())
  assert_true(buffer.maximum_token_status(-1).is_out_of_range())
}

///|
test "Luna token storage seals without exposing mutable storage" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=4,
    limits=luna_storage_limits(),
  )
  assert_eq(storage.capacity(), 4)
  assert_eq(@inference.LunaTokenBufferStorage::required_int_cells(4), 4UL)
  let write = storage.begin(token_count=3)
  write.push(7)
  write.push(8)
  write.push(9)
  assert_eq(write.written_count(), 3)
  let lease = write.seal()
  let buffer = lease.token_buffer()
  assert_true(buffer.is_live())
  assert_eq(buffer.length(), 3)
  for index, expected in [7, 8, 9] {
    let status = buffer.token_status(index)
    assert_true(status.is_ready())
    assert_eq(status.token_id(), expected)
  }
  assert_true(buffer.maximum_token_status(9).is_within_bound())
  assert_true(buffer.maximum_token_status(8).is_out_of_range())
  assert_true(buffer.maximum_token_status(99).is_within_bound())
  assert_true(buffer.maximum_token_status(-1).is_out_of_range())
}

///|
test "owned and leased bounds use the same actual maximum" {
  let limits = luna_storage_limits()
  let owned = @inference.TokenBuffer::new([2, 91, 4], limits)
  let storage = @inference.LunaTokenBufferStorage::new(capacity=3, limits~)
  let write = storage.begin(token_count=3)
  write.push(2)
  write.push(91)
  write.push(4)
  let lease = write.seal()
  let leased = lease.token_buffer()
  for bound in [90, 91, 99] {
    assert_eq(
      owned.maximum_token_status(bound).is_within_bound(),
      leased.maximum_token_status(bound).is_within_bound(),
    )
    assert_eq(
      owned.maximum_token_status(bound).is_out_of_range(),
      leased.maximum_token_status(bound).is_out_of_range(),
    )
  }
  lease.release()
}

///|
test "released and reused Luna generations reject retained aliases" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=3,
    limits=luna_storage_limits(),
  )
  let first_write = storage.begin(token_count=2)
  first_write.push(10)
  first_write.push(11)
  let first_lease = first_write.seal()
  let retained = first_lease.token_buffer()
  first_lease.release()
  assert_false(retained.is_live())
  assert_true(retained.maximum_token_status(99).is_stale())
  assert_true(retained.maximum_token_status(-1).is_stale())
  let released = retained.token_status(0)
  assert_false(released.is_ready())
  assert_true(released.is_stale())
  assert_eq(released.token_id(), -1)
  let second_write = storage.begin(token_count=1)
  second_write.push(42)
  let second_lease = second_write.seal()
  let second = second_lease.token_buffer()
  assert_eq(second.length(), 1)
  assert_eq(second.token_status(0).token_id(), 42)
  assert_true(second.maximum_token_status(42).is_within_bound())
  assert_true(second.maximum_token_status(41).is_out_of_range())
  assert_true(retained.token_status(0).is_stale())
  assert_true(retained.maximum_token_status(99).is_stale())
  second_lease.release()
}

///|
test "Luna cached maximum belongs to the exact sealed generation" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=2,
    limits=luna_storage_limits(),
  )
  let first_write = storage.begin(token_count=2)
  first_write.push(3)
  first_write.push(98)
  let first_lease = first_write.seal()
  let first = first_lease.token_buffer()
  assert_true(first.maximum_token_status(97).is_out_of_range())
  first_lease.release()

  let second_write = storage.begin(token_count=2)
  second_write.push(1)
  second_write.push(2)
  let second_lease = second_write.seal()
  let second = second_lease.token_buffer()
  assert_true(first.maximum_token_status(99).is_stale())
  assert_true(second.maximum_token_status(2).is_within_bound())
  assert_true(second.maximum_token_status(1).is_out_of_range())
  second_lease.release()
}

///|
test "stale Luna token-buffer keeps old length but rejects payload reads" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=1,
    limits=luna_storage_limits(),
  )
  let write = storage.begin(token_count=1)
  write.push(1)
  let lease = write.seal()
  let retained = lease.token_buffer()
  lease.release()
  assert_eq(retained.length(), 1)
  assert_true(retained.token_status(0).is_stale())
}

///|
test "Luna write lifecycle is transactional and epoch checked" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=2,
    limits=luna_storage_limits(),
  )
  let write = storage.begin(token_count=2)
  write.push(1)
  try write.push(100) catch {
    Invalid(field=InputToken, issue=OutOfRange, index=1) => ()
    _ => fail("unexpected invalid-token error")
  } noraise {
    _ => fail("out-of-range token was written")
  }
  assert_eq(write.written_count(), 1)
  try write.seal() catch {
    Invalid(field=LunaTokenStorage, issue=LunaStorageLifecycle, index=1) => ()
    _ => fail("unexpected incomplete-seal error")
  } noraise {
    _ => fail("incomplete write was sealed")
  }
  write.abort()
  try write.push(2) catch {
    Invalid(field=LunaTokenStorage, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale-write error")
  } noraise {
    _ => fail("aborted write remained usable")
  }
  let next = storage.begin(token_count=1)
  next.push(3)
  let lease = next.seal()
  lease.release()
  try lease.release() catch {
    Invalid(field=LunaTokenStorage, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale-lease error")
  } noraise {
    _ => fail("lease released twice")
  }
}

///|
test "rejected Luna push cannot advance or corrupt cached maximum" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=2,
    limits=luna_storage_limits(),
  )
  let write = storage.begin(token_count=2)
  write.push(5)
  try write.push(100) catch {
    Invalid(field=InputToken, issue=OutOfRange, index=1) => ()
    _ => fail("unexpected oversized-token error")
  } noraise {
    _ => fail("oversized token was accepted")
  }
  assert_eq(write.written_count(), 1)
  write.push(7)
  assert_eq(write.written_count(), 2)
  let lease = write.seal()
  let buffer = lease.token_buffer()
  assert_true(buffer.maximum_token_status(7).is_within_bound())
  assert_true(buffer.maximum_token_status(6).is_out_of_range())
  lease.release()
}

///|
test "abort and rebegin reset Luna generation maximum" {
  let storage = @inference.LunaTokenBufferStorage::new(
    capacity=2,
    limits=luna_storage_limits(),
  )
  let abandoned = storage.begin(token_count=2)
  abandoned.push(98)
  abandoned.abort()

  let write = storage.begin(token_count=1)
  write.push(2)
  let lease = write.seal()
  let buffer = lease.token_buffer()
  assert_true(buffer.maximum_token_status(2).is_within_bound())
  assert_true(buffer.maximum_token_status(1).is_out_of_range())
  lease.release()
}

///|
test "text retains its canonical immutable UTF-8" {
  let input = @inference.TextInput::from_utf8(
    b"moon \xe4\xb8\xad",
    luna_storage_limits(),
  )
  assert_eq(input.utf8(), b"moon \xe4\xb8\xad")
  assert_eq(input.utf8_bytes(), 8)
  assert_eq(input.text(), "moon 中")
}

///|
