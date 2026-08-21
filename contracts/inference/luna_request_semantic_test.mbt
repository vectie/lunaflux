///|
fn semantic_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=16,
    max_new_tokens=8,
    max_context_tokens=24,
    max_token_id=99,
    max_stop_token_ids=3,
    max_stop_strings=2,
    max_stop_string_bytes=5,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature=4.0,
  )
}

///|
fn push_semantic_bytes(
  write : @inference.LunaRequestSemanticWrite,
  bytes : Bytes,
) -> Unit raise {
  write.push_stop_string_length(bytes.length())
  for byte in bytes {
    write.push_stop_string_byte(byte)
  }
}

///|
fn ready_semantic_lease(
  work_units : Int,
) -> (@inference.LunaRequestSemanticLease, UInt64) raise {
  let budget = @inference.LunaRequestSemanticStepBudget::new(work_units)
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    budget,
  )
  let write = storage.begin(
    stop_token_count=3,
    stop_string_count=2,
    cache_scope_byte_count=7,
    cache_permission=ReadWrite,
  )
  for token in [9, 2, 7] {
    write.push_stop_token(token)
  }
  push_semantic_bytes(write, b"END")
  push_semantic_bytes(write, b"\xc3\xa9")
  for byte in b"scope-1" {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  let mut calls = 0
  while true {
    let progress = work.progress()
    let used = work.last_work_units()
    assert_true(used <= work_units)
    calls += 1
    if progress.is_ready() {
      let total = work.total_work_units()
      return (work.take_lease(), total)
    }
    assert_true(progress.is_pending())
    assert_true(used > 0)
    if calls > 1000 {
      fail("semantic work did not terminate")
    }
  }
  abort("unreachable")
}

///|
test "Luna request semantic sizing and budgets are exact and bounded" {
  let limits = semantic_limits()
  assert_eq(
    @inference.LunaRequestSemanticStorage::required_int_cells(limits),
    10UL,
  )
  assert_eq(
    @inference.LunaRequestSemanticStorage::required_byte_cells(limits),
    18UL,
  )
  assert_eq(@inference.LunaRequestSemanticStepBudget::new(17).work_units(), 17)
  for invalid_budget in [0, -1, 65537] {
    try @inference.LunaRequestSemanticStepBudget::new(invalid_budget) catch {
      Invalid(field=LunaRequestSemantic, issue=OutOfRange, index=0) => ()
      _ => fail("unexpected semantic budget error")
    } noraise {
      _ => fail("expected semantic budget rejection")
    }
  }
}

///|
test "Luna semantic validation is deterministic across progress budgets" {
  let (one, total_one) = ready_semantic_lease(1)
  let (seventeen, total_seventeen) = ready_semantic_lease(17)
  let (maximum, total_maximum) = ready_semantic_lease(65536)
  assert_eq(total_one, total_seventeen)
  assert_eq(total_one, total_maximum)
  assert_eq(total_one, 43UL)
  one.release()
  seventeen.release()
  maximum.release()
}

///|
test "Luna semantic View exposes authenticated scalar request semantics" {
  let (lease, _) = ready_semantic_lease(1)
  let view = lease.view()
  assert_eq(view.inference_limits(), semantic_limits())
  assert_eq(view.cache_permission(), ReadWrite)
  assert_eq(view.stop_token_count(), 3)
  assert_eq(view.stop_token_at(0), 9)
  assert_eq(view.stop_token_at(1), 2)
  assert_eq(view.stop_token_at(2), 7)
  assert_eq(view.stop_string_count(), 2)
  assert_eq(view.stop_string_length(0), 3)
  assert_eq(view.stop_string_byte_at(string_index=0, byte_index=2), b'D')
  assert_eq(view.stop_string_length(1), 2)
  assert_eq(view.stop_string_byte_at(string_index=1, byte_index=0), b'\xc3')
  assert_eq(view.cache_scope_length(), 7)
  assert_eq(view.cache_scope_byte_at(6), b'1')
  for token in [2, 7, 9] {
    let found = view.is_stop_token(token)
    assert_true(found.is_present())
    assert_false(found.is_absent())
    assert_false(found.is_stale())
    assert_true(found.work_units() > 0)
    assert_true(found.work_units() <= 2)
  }
  let absent = view.is_stop_token(8)
  assert_false(absent.is_present())
  assert_true(absent.is_absent())
  assert_false(absent.is_stale())
  lease.release()
  let stale = view.is_stop_token(2)
  assert_false(stale.is_present())
  assert_false(stale.is_absent())
  assert_true(stale.is_stale())
  assert_eq(stale.work_units(), 0)
  try view.stop_token_at(-1) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale authentication must precede index validation")
  } noraise {
    _ => fail("expected stale View rejection")
  }
}

///|
test "zero stop tokens never match unused preallocated cells" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  let status = lease.view().is_stop_token(0)
  assert_true(status.is_absent())
  assert_eq(status.work_units(), 0)
  lease.release()
}

///|
test "Lease pins storage and transfer makes retained Work stale" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let write = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  write.push_cache_scope_byte(b'x')
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  try work.state() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale Work error")
  } noraise {
    _ => fail("transferred Work must be stale")
  }
  try
    storage.begin(
      stop_token_count=0,
      stop_string_count=0,
      cache_scope_byte_count=1,
      cache_permission=Disabled,
    )
  catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected pinned-storage error")
  } noraise {
    _ => fail("Lease must pin storage")
  }
  lease.release()
  let reused = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  reused.abort()
}

///|
fn expect_semantic_failure(
  work : @inference.LunaRequestSemanticWork,
  field : @inference.ContractField,
  issue : @inference.ContractIssue,
  index : Int,
) -> Unit raise {
  while true {
    let progress = work.progress()
    if progress.is_failed() {
      let failure = work.failure()
      assert_eq(failure.field(), field)
      assert_eq(failure.issue(), issue)
      assert_eq(failure.index(), index)
      assert_true(work.state().is_failed())
      work.abort()
      return
    }
  }
}

///|
test "complete writes use bounded token then string semantic precedence" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let invalid_token = storage.begin(
    stop_token_count=2,
    stop_string_count=1,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  invalid_token.push_stop_token(100)
  invalid_token.push_stop_token(100)
  push_semantic_bytes(invalid_token, b"")
  invalid_token.push_cache_scope_byte(b'!')
  expect_semantic_failure(invalid_token.finish(), StopToken, OutOfRange, 0)
  let duplicate = storage.begin(
    stop_token_count=2,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  duplicate.push_stop_token(4)
  duplicate.push_stop_token(4)
  duplicate.push_cache_scope_byte(b'x')
  expect_semantic_failure(duplicate.finish(), StopToken, Duplicate, 1)
  let invalid_utf8 = storage.begin(
    stop_token_count=0,
    stop_string_count=1,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  push_semantic_bytes(invalid_utf8, b"\xc0\x80")
  invalid_utf8.push_cache_scope_byte(b'x')
  expect_semantic_failure(invalid_utf8.finish(), StopString, InvalidFormat, 0)
}

///|
test "write-time string capacity errors keep exact builder precedence" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let first = storage.begin(
    stop_token_count=0,
    stop_string_count=2,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  try first.push_stop_string_length(6) catch {
    Invalid(field=StopString, issue=TooLong, index=0) => ()
    _ => fail("unexpected first-string precedence")
  } noraise {
    _ => fail("expected oversized first string rejection")
  }
  first.abort()
  let later = storage.begin(
    stop_token_count=0,
    stop_string_count=2,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  push_semantic_bytes(later, b"ok")
  try later.push_stop_string_length(6) catch {
    Invalid(field=StopString, issue=TooLong, index=1) => ()
    _ => fail("unexpected later-string precedence")
  } noraise {
    _ => fail("expected oversized later string rejection")
  }
  later.abort()
}

///|
test "structural declaration errors win before semantic Work exists" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let mixed = storage.begin(
    stop_token_count=2,
    stop_string_count=1,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  mixed.push_stop_token(4)
  mixed.push_stop_token(4)
  try mixed.push_stop_string_length(6) catch {
    Invalid(field=StopString, issue=TooLong, index=0) => ()
    _ => fail("structural string declaration must win at its call")
  } noraise {
    _ => fail("expected structural declaration rejection")
  }
  mixed.abort()
  try
    storage.begin(
      stop_token_count=4,
      stop_string_count=2,
      cache_scope_byte_count=9,
      cache_permission=Disabled,
    )
  catch {
    Invalid(field=StopToken, issue=TooMany, index=3) => ()
    _ => fail("begin must validate counts before cache capacity")
  } noraise {
    _ => fail("expected structural count rejection")
  }
}

///|
test "complete writes validate duplicate strings before cache grammar" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let duplicate = storage.begin(
    stop_token_count=0,
    stop_string_count=2,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  push_semantic_bytes(duplicate, b"x")
  push_semantic_bytes(duplicate, b"x")
  duplicate.push_cache_scope_byte(b'!')
  expect_semantic_failure(duplicate.finish(), StopString, Duplicate, 1)
  let bad_cache = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=ReadOnly,
  )
  bad_cache.push_cache_scope_byte(b'!')
  expect_semantic_failure(bad_cache.finish(), CacheScope, InvalidFormat, 0)
}
