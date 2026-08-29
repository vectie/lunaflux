///|
test "prefix projection exposes only authenticated cache identity scalars" {
  let (lease, _) = ready_semantic_lease(1)
  let prefix = lease.prefix_view()
  assert_true(prefix.is_live())
  assert_true(prefix.binds_stop_tokens(lease.stop_token_view()))
  assert_eq(prefix.cache_permission(), ReadWrite)
  assert_eq(prefix.scope_length(), 7)
  for index, expected in b"scope-1" {
    assert_eq(prefix.scope_byte_at(index), expected)
  }
  try prefix.scope_byte_at(7) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageLifecycle, index=7) =>
      ()
    _ => fail("unexpected live prefix bound error")
  } noraise {
    _ => fail("prefix projection accepted an oversized scope index")
  }
  lease.release()
  assert_false(prefix.is_live())
  try prefix.scope_byte_at(-1) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("stale prefix authentication lost precedence over bounds")
  } noraise {
    _ => fail("stale prefix projection remained readable")
  }
  try prefix.cache_permission() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected stale prefix permission error")
  } noraise {
    _ => fail("stale prefix permission remained readable")
  }
}

///|
test "prefix projection rejects a different live semantic lease" {
  let (first, _) = ready_semantic_lease(1)
  let (second, _) = ready_semantic_lease(1)
  assert_false(first.prefix_view().binds_stop_tokens(second.stop_token_view()))
  first.release()
  second.release()
}

///|
test "semantic storage reuse stales the retained prefix projection" {
  let storage = @inference.LunaRequestSemanticStorage::new(
    semantic_limits(),
    @inference.LunaRequestSemanticStepBudget::new(17),
  )
  let first = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=ReadOnly,
  )
  first.push_cache_scope_byte(b'a')
  let first_work = first.finish()
  while !first_work.progress().is_ready() {

  }
  let first_lease = first_work.take_lease()
  let stale = first_lease.prefix_view()
  first_lease.release()
  let second = storage.begin(
    stop_token_count=0,
    stop_string_count=0,
    cache_scope_byte_count=1,
    cache_permission=Disabled,
  )
  second.push_cache_scope_byte(b'b')
  let second_work = second.finish()
  while !second_work.progress().is_ready() {

  }
  let second_lease = second_work.take_lease()
  assert_false(stale.is_live())
  assert_eq(second_lease.prefix_view().cache_permission(), Disabled)
  second_lease.release()
}
