///|
fn stop_projection_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=16,
    max_input_tokens=8,
    max_new_tokens=4,
    max_context_tokens=12,
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
test "Luna stop-token projection carries only exact leased membership" {
  let limits = stop_projection_limits()
  let storage = @inference.LunaRequestSemanticStorage::new(
    limits,
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  let write = storage.begin(
    stop_token_count=3,
    stop_string_count=1,
    cache_scope_byte_count=5,
    cache_permission=ReadWrite,
  )
  for token in [9, 2, 7] {
    write.push_stop_token(token)
  }
  write.push_stop_string_length(3)
  for byte in b"END" {
    write.push_stop_string_byte(byte)
  }
  for byte in b"scope" {
    write.push_cache_scope_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  let lease = work.take_lease()
  let projection = lease.stop_token_view()
  let scheduler_retention = @inference.LunaRequestStopTokenRetentionSlot::new()
  let worker_retention = @inference.LunaRequestStopTokenRetentionSlot::new()
  projection.retain_into(scheduler_retention)
  projection.retain_into(worker_retention)
  assert_true(projection.is_live())
  for token in [2, 7, 9] {
    let status = projection.is_stop_token(token)
    assert_true(status.is_present())
    assert_false(status.is_absent())
    assert_false(status.is_stale())
    assert_true(status.work_units() > 0)
  }
  assert_true(projection.is_stop_token(8).is_absent())
  assert_true(scheduler_retention.is_stop_token(7).is_present())
  assert_true(worker_retention.is_stop_token(8).is_absent())
  let overflow_retention = @inference.LunaRequestStopTokenRetentionSlot::new()
  try projection.retain_into(overflow_retention) catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected third semantic-retention error")
  } noraise {
    _ => fail("semantic storage issued more than two retentions")
  }
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected retained semantic-release error")
  } noraise {
    _ => fail("lease release invalidated live semantic retentions")
  }
  scheduler_retention.release()
  assert_false(scheduler_retention.is_live())
  assert_true(worker_retention.is_live())
  try scheduler_retention.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageStale, index=0) => ()
    _ => fail("unexpected copied-retention release error")
  } noraise {
    _ => fail("copied retention released twice")
  }
  try lease.release() catch {
    Invalid(field=LunaRequestSemantic, issue=LunaStorageBusy, index=0) => ()
    _ => fail("unexpected single-retention semantic-release error")
  } noraise {
    _ => fail("lease release ignored the worker retention")
  }
  worker_retention.release()
  lease.release()
  assert_false(projection.is_live())
  let stale = projection.is_stop_token(2)
  assert_true(stale.is_stale())
  assert_false(stale.is_present())
  assert_false(stale.is_absent())
  assert_eq(stale.work_units(), 0)
}
