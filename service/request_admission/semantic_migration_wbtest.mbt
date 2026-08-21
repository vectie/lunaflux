///|
fn semantic_migration_request() -> @inference.GenerateRequest raise {
  let limits = fixture_inference()
  let base = fixture_request(fixture_identity("0"))
  @inference.GenerateRequest::new(
    protocol_version=base.protocol_version(),
    request_id=base.request_id(),
    model_identity=base.model_identity(),
    input=base.input(),
    max_new_tokens=base.max_new_tokens(),
    context_ceiling=base.context_ceiling(),
    sampling=base.sampling(),
    sampling_seed=base.sampling_seed(),
    stops=@inference.StopConditions::new(
      [9, 3],
      [b"\xf0\x9f\x8c\x99", b"END"],
      limits,
    ),
    stream_preference=base.stream_preference(),
    deadline=base.deadline(),
    cache=base.cache(),
    trace=base.trace(),
    limits~,
  )
}

///|
test "semantic import is budget invariant and preserves exact UTF-8/cache data" {
  let request = semantic_migration_request()
  let one_now = Ref(11UL)
  let one_pool = fixture_preparation_pool(one_now, 1, 1, 4096UL)
  let one_work = one_pool.try_submit(pool_received_at(request, 10UL)).work()
  drive_preparation_ready(one_pool, one_work)
  let one_total = one_work.total_work_units()
  let one_claim = one_work.take_prepared().take_claim()
  if one_claim.semantic_lease.length() != 1 {
    abort("semantic migration claim lease invariant")
  }
  let semantic_lease = one_claim.semantic_lease[0]
  let view = semantic_lease.view()
  assert_eq(view.stop_token_count(), 2)
  assert_eq(view.stop_token_at(0), 9)
  assert_eq(view.stop_token_at(1), 3)
  assert_true(one_claim.is_stop_token(3))
  assert_true(one_claim.is_stop_token(9))
  assert_false(one_claim.is_stop_token(8))
  assert_eq(view.stop_string_count(), 2)
  assert_eq(view.stop_string_length(0), 4)
  for index, expected in b"\xf0\x9f\x8c\x99" {
    assert_eq(
      view.stop_string_byte_at(string_index=0, byte_index=index),
      expected,
    )
  }
  assert_eq(view.stop_string_length(1), 3)
  for index, expected in b"END" {
    assert_eq(
      view.stop_string_byte_at(string_index=1, byte_index=index),
      expected,
    )
  }
  assert_eq(view.cache_permission(), Disabled)
  assert_eq(view.cache_scope_length(), 5)
  for index, expected in b"scope" {
    assert_eq(view.cache_scope_byte_at(index), expected)
  }
  one_claim.release()

  let wide_now = Ref(11UL)
  let wide_pool = fixture_preparation_pool(wide_now, 1, 17, 4096UL)
  let wide_work = wide_pool
    .try_submit(pool_received_at(semantic_migration_request(), 10UL))
    .work()
  drive_preparation_ready(wide_pool, wide_work)
  assert_eq(wide_work.total_work_units(), one_total)
  wide_work.take_prepared().discard()
}

///|
test "claim release is transactional while lower semantic retention is live" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let work = pool
    .try_submit(pool_received_at(semantic_migration_request(), 10UL))
    .work()
  drive_preparation_ready(pool, work)
  let claim = work.take_prepared().take_claim()
  let retained = claim
  let lower = @inference.LunaRequestStopTokenRetentionSlot::new()
  claim.stop_tokens.retain_into(lower)
  try claim.release() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected retained-claim release error")
  } noraise {
    _ => fail("claim released while lower semantic authority remained live")
  }
  assert_eq(pool.free_lane_count(), 0)
  assert_true(retained.pooled_output is Some(_))
  assert_true(retained.token_lease is Some(_))
  assert_true(retained.pool is Some(_))
  assert_eq(retained.semantic_lease.length(), 1)
  assert_eq(retained.scheduler_request().request_id().value(), 7UL)
  assert_true(retained.is_stop_token(9))
  lower.release()
  retained.release()
  assert_eq(pool.free_lane_count(), 1)
}

///|
test "output setup cancellation invalidates full and token-only semantic views" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let work = pool
    .try_submit(pool_received_at(semantic_migration_request(), 10UL))
    .work()
  drive_to_phase(pool, work, LUNA_PREPARATION_OUTPUT_SETUP)
  let lane = pool.lanes[work.lane]
  let lease = lane.require_semantic_lease()
  let full_view = lease.view()
  let stop_view = lease.stop_token_view()
  work.cancel()
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(pool.free_lane_count(), 1)
  assert_true(stop_view.is_stop_token(9).is_stale())
  try full_view.cache_scope_length() catch {
    _ => ()
  } noraise {
    _ => fail("cancelled output setup retained a full semantic view")
  }
  let reused = pool
    .try_submit(pool_received_at(semantic_migration_request(), 10UL))
    .work()
  assert_eq(reused.state(), LunaRequestPreparationPending)
  reused.cancel()
  ignore(pool.progress())
}
