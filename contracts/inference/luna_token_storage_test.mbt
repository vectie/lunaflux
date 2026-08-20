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
  assert_true(retained.token_status(0).is_stale())
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
