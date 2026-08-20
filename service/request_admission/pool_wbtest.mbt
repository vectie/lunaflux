///|
fn pool_received_at(
  request : @inference.GenerateRequest,
  receipt_at_millis : UInt64,
) -> ReceivedRequest raise {
  let receipt : RequestReceipt = { at_millis: receipt_at_millis }
  {
    receipt,
    request,
    deadline: fixture_deadline(receipt, request),
    inference: fixture_inference(),
    consumed: false,
  }
}

///|
fn fixture_preparation_pool(
  now : Ref[UInt64],
  lane_count : Int,
  step_work_units : Int,
  total_work_units : UInt64,
) -> LunaRequestPreparationPool raise {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let identity = fixture_identity("0")
  new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    identity,
    inference,
    () => now.val,
    lane_count,
    LunaRequestPreparationStepBudget::new(step_work_units),
    LunaRequestPreparationWorkLimit::new(total_work_units),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=1024UL,
    ),
  )
}

///|
fn drive_preparation_ready(
  pool : LunaRequestPreparationPool,
  work : LunaRequestPreparationWork,
) -> Unit raise {
  let mut turns = 0
  while work.state() is LunaRequestPreparationPending {
    assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
    turns += 1
    if turns > 4096 {
      fail("preparation did not converge")
    }
  }
  assert_eq(work.state(), LunaRequestPreparationReady)
}

///|
test "saturation does not consume N+1 and release permits exact retry" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let first_received = pool_received_at(
    fixture_request(fixture_identity("0")),
    10UL,
  )
  let second_received = pool_received_at(
    fixture_request(fixture_identity("0")),
    10UL,
  )
  let first = pool.try_submit(first_received)
  assert_eq(first.kind(), LunaRequestPreparationAdmitted)
  let saturated = pool.try_submit(second_received)
  assert_eq(saturated.kind(), LunaRequestPreparationSaturated)
  assert_false(second_received.consumed)
  let first_work = first.work()
  drive_preparation_ready(pool, first_work)
  let prepared = first_work.take_prepared()
  let claim = prepared.take_claim()
  assert_eq(claim.scheduler_request().request_id().value(), 7UL)
  claim.release()
  assert_eq(pool.free_lane_count(), 1)
  let retried = pool.try_submit(second_received)
  assert_eq(retried.kind(), LunaRequestPreparationAdmitted)
  assert_true(second_received.consumed)
  retried.work().cancel()
  ignore(pool.progress())
}

///|
test "central ring advances lanes in strict round-robin quanta" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 2, 1, 4096UL)
  let first = pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  let second = pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(first.total_work_units(), 1UL)
  assert_eq(second.total_work_units(), 0UL)
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(first.total_work_units(), 1UL)
  assert_eq(second.total_work_units(), 1UL)
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(first.total_work_units(), 2UL)
  assert_eq(second.total_work_units(), 1UL)
  first.cancel()
  second.cancel()
  ignore(pool.progress())
  ignore(pool.progress())
}

///|
test "cancel revokes old work and lane reuse changes generation" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let old = pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  old.cancel()
  assert_eq(old.state(), LunaRequestPreparationCancelRequested)
  ignore(pool.progress())
  try old.state() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected cancelled-work error")
  } noraise {
    _ => fail("cancelled work retained lane authority")
  }
  let fresh = pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  assert_eq(fresh.state(), LunaRequestPreparationPending)
  try old.cancel() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected stale-generation cancel error")
  } noraise {
    _ => fail("old generation cancelled reused lane")
  }
  fresh.cancel()
  ignore(pool.progress())
}

///|
test "deadline and total-work caps fail without publishing a claim" {
  let now = Ref(11UL)
  let expired_pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let expired = expired_pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  now.val = 110UL
  ignore(expired_pool.progress())
  assert_eq(expired.state(), LunaRequestPreparationFailed)
  assert_eq(expired.failure(), Invalid(Deadline))
  expired.discard()
  let capped_now = Ref(11UL)
  let capped_pool = fixture_preparation_pool(capped_now, 1, 1, 1UL)
  let capped = capped_pool
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  ignore(capped_pool.progress())
  assert_eq(capped.state(), LunaRequestPreparationPending)
  ignore(capped_pool.progress())
  assert_eq(capped.state(), LunaRequestPreparationFailed)
  assert_eq(capped.failure(), Invalid(LunaPreparationWork))
  capped.discard()
}

///|
test "TokenIds reuse canonical storage and claim release revokes aliases" {
  let now = Ref(11UL)
  let pool = fixture_preparation_pool(now, 1, 1, 4096UL)
  let limits = fixture_inference()
  let identity = fixture_identity("0")
  let base = fixture_request(identity)
  let request = @inference.GenerateRequest::new(
    protocol_version=base.protocol_version(),
    request_id=base.request_id(),
    model_identity=base.model_identity(),
    input=@inference.Input::from_token_ids([97, 98, 99], limits),
    max_new_tokens=base.max_new_tokens(),
    context_ceiling=base.context_ceiling(),
    sampling=base.sampling(),
    sampling_seed=base.sampling_seed(),
    stops=base.stops(),
    stream_preference=base.stream_preference(),
    deadline=base.deadline(),
    cache=base.cache(),
    trace=base.trace(),
    limits~,
  )
  let work = pool.try_submit(pool_received_at(request, 10UL)).work()
  drive_preparation_ready(pool, work)
  let prepared = work.take_prepared()
  assert_eq(prepared.input_token_count(), 3)
  let claim = prepared.take_claim()
  let retained = claim
  assert_eq(claim.scheduler_request().request_id().value(), 7UL)
  claim.release()
  try retained.scheduler_request() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected released-claim error")
  } noraise {
    _ => fail("released claim retained scheduler authority")
  }
  try retained.release() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected repeated-release error")
  } noraise {
    _ => fail("claim release was replayable")
  }
}
