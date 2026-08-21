///|
fn semantic_hostile_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=16,
    max_new_tokens=8,
    max_context_tokens=24,
    max_token_id=99,
    max_stop_token_ids=4,
    max_stop_strings=3,
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
fn hostile_semantic_work(
  tokens : Array[Int],
  strings : Array[Bytes],
  cache : Bytes,
) -> @inference.LunaRequestSemanticWork raise {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_hostile_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let write = storage.begin(
    stop_token_count=tokens.length(),
    stop_string_count=strings.length(),
    cache_scope_byte_count=cache.length(),
    cache_permission=ReadOnly,
  )
  for token in tokens {
    write.push_stop_token(token)
  }
  for string in strings {
    write.push_stop_string_length(string.length())
    for byte in string {
      write.push_stop_string_byte(byte)
    }
  }
  for byte in cache {
    write.push_cache_scope_byte(byte)
  }
  write.finish()
}

///|
fn hostile_failure(
  work : @inference.LunaRequestSemanticWork,
) -> @inference.LunaRequestSemanticFailure raise {
  while true {
    let progress = work.progress()
    assert_true(work.last_work_units() <= 1)
    if progress.is_failed() {
      return work.failure()
    }
    if progress.is_ready() {
      fail("expected semantic validation failure")
    }
  }
  abort("unreachable")
}

///|
fn hostile_ready(
  work : @inference.LunaRequestSemanticWork,
) -> @inference.LunaRequestSemanticLease raise {
  while true {
    let progress = work.progress()
    assert_true(work.last_work_units() <= 1)
    if progress.is_ready() {
      return work.take_lease()
    }
    if progress.is_failed() {
      fail("expected valid semantic work")
    }
  }
  abort("unreachable")
}

///|
test "duplicate strings compare long prefixes and report earliest repeat" {
  let work = hostile_semantic_work([], [b"abcde", b"abcdf", b"abcde"], b"scope")
  let failure = hostile_failure(work)
  assert_eq(failure.field(), StopString)
  assert_eq(failure.issue(), Duplicate)
  assert_eq(failure.index(), 2)
  work.abort()
}

///|
test "empty and unsafe cache scopes retain canonical ordering" {
  let empty = hostile_semantic_work([], [], b"")
  let empty_failure = hostile_failure(empty)
  assert_eq(empty_failure.field(), CacheScope)
  assert_eq(empty_failure.issue(), Empty)
  empty.abort()
  let invalid_scope = hostile_semantic_work([], [], b"scope!")
  let unsafe_failure = hostile_failure(invalid_scope)
  assert_eq(unsafe_failure.field(), CacheScope)
  assert_eq(unsafe_failure.issue(), InvalidFormat)
  invalid_scope.abort()
}

///|
test "UTF-8 validation rejects every hostile class and accepts supplementary" {
  for
    malformed in [
      b"\xc0\x80", b"\xed\xa0\x80", b"\xe2\x82", b"\xe2\x28\xa1", b"\xf4\x90\x80\x80",
    ] {
    let work = hostile_semantic_work([], [malformed], b"scope")
    let failure = hostile_failure(work)
    assert_eq(failure.field(), StopString)
    assert_eq(failure.issue(), InvalidFormat)
    assert_eq(failure.index(), 0)
    work.abort()
  }
  let valid = hostile_semantic_work([], [b"\xf0\x9f\x98\x80"], b"scope")
  let lease = hostile_ready(valid)
  assert_eq(lease.view().stop_string_length(0), 4)
  lease.release()
}

///|
test "reverse token index supports present and absent-gap membership" {
  let work = hostile_semantic_work([9, 7, 2], [], b"scope")
  let lease = hostile_ready(work)
  let view = lease.view()
  for token in [2, 7, 9] {
    assert_true(view.is_stop_token(token).is_present())
  }
  let gap = view.is_stop_token(8)
  assert_true(gap.is_absent())
  assert_false(gap.is_present())
  assert_false(gap.is_stale())
  assert_true(gap.work_units() <= 2)
  lease.release()
}

///|
fn expect_live_bounds(view : @inference.LunaRequestSemanticView) -> Unit raise {
  try view.stop_token_at(9) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=9) =>
      ()
    _ => fail("unexpected token bound error")
  } noraise {
    _ => fail("expected token bound rejection")
  }
  try view.stop_string_length(-1) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=-1) =>
      ()
    _ => fail("unexpected string bound error")
  } noraise {
    _ => fail("expected string bound rejection")
  }
  try view.stop_string_byte_at(string_index=8, byte_index=0) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=8) =>
      ()
    _ => fail("unexpected string-index error")
  } noraise {
    _ => fail("expected string-index rejection")
  }
  try view.stop_string_byte_at(string_index=0, byte_index=8) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=8) =>
      ()
    _ => fail("unexpected string-byte error")
  } noraise {
    _ => fail("expected string-byte rejection")
  }
  try view.cache_scope_byte_at(8) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=8) =>
      ()
    _ => fail("unexpected cache bound error")
  } noraise {
    _ => fail("expected cache bound rejection")
  }
}

///|
fn expect_stale_view(view : @inference.LunaRequestSemanticView) -> Unit raise {
  try view.inference_limits() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale limits error")
  } noraise {
    _ => fail("expected stale limits")
  }
  try view.cache_permission() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale permission error")
  } noraise {
    _ => fail("expected stale permission")
  }
  try view.stop_token_count() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale token count")
  } noraise {
    _ => fail("expected stale token count")
  }
  try view.stop_token_at(9) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale must precede token bounds")
  } noraise {
    _ => fail("expected stale token read")
  }
  try view.stop_string_count() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale string count")
  } noraise {
    _ => fail("expected stale string count")
  }
  try view.stop_string_length(-1) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale must precede string bounds")
  } noraise {
    _ => fail("expected stale string length")
  }
  try view.stop_string_byte_at(string_index=9, byte_index=9) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale must precede string byte bounds")
  } noraise {
    _ => fail("expected stale string byte")
  }
  try view.cache_scope_length() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale cache length")
  } noraise {
    _ => fail("expected stale cache length")
  }
  try view.cache_scope_byte_at(9) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale must precede cache bounds")
  } noraise {
    _ => fail("expected stale cache byte")
  }
}

///|
test "every View accessor authenticates before bounds and release is exact" {
  let lease = hostile_ready(hostile_semantic_work([3], [b"stop"], b"scope"))
  let view = lease.view()
  expect_live_bounds(view)
  lease.release()
  expect_stale_view(view)
  assert_true(view.is_stop_token(3).is_stale())
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected double release error")
  } noraise {
    _ => fail("expected double release rejection")
  }
}

///|
test "ready Work can abort and its storage can be reused" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_hostile_limits(),
    @inference.LunaRequestSemanticStepBudget::new(65536),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  assert_true(work.progress().is_ready())
  work.abort()
  let reuse = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  reuse.abort()
}
