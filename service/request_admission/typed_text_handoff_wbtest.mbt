///|
fn begin_typed_fixture(
  storage : @inference.LunaTextRequestHandoffStorage,
  deadline_millis? : UInt64 = 100UL,
) -> @inference.LunaTextRequestHandoffLease raise {
  let limits = fixture_inference()
  let write = storage.begin(
    prompt_byte_limit=limits.max_text_bytes(),
    stop_token_count=1,
    stop_string_count=1,
    request_id=@inference.RequestId::new(71UL),
    effective=@inference.EffectiveLimits::new(
      max_new_tokens=4,
      context_ceiling=16,
      limits,
    ),
    sampling=@inference.LunaValidatedSampling::sample(
      temperature=0.75,
      has_top_k=true,
      top_k=11,
      has_top_p=true,
      top_p=0.9,
      limits,
    ),
    sampling_seed=@inference.SamplingSeed::new(99UL),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(
      deadline_millis, limits,
    ),
  )
  for byte in b"abc" {
    write.push_prompt_byte(byte)
  }
  write.push_stop_token(9)
  write.push_stop_string_length(3)
  for byte in b"END" {
    write.push_stop_string_byte(byte)
  }
  let work = write.finish()
  while work.progress().is_pending() {

  }
  work.take_lease()
}

///|
fn typed_fixture(
  identity : @spec.ModelIdentity,
  deadline_millis? : UInt64 = 100UL,
) -> (
  @inference.LunaTextRequestHandoffStorage,
  @inference.LunaTextRequestHandoffLease,
) raise {
  let limits = fixture_inference()
  let storage = @inference.LunaTextRequestHandoffStorage::new(
    limits,
    identity,
    @inference.CacheScope::from_ascii(b"scope", limits),
    @inference.LunaRequestSemanticStepBudget::new(1),
  )
  (storage, begin_typed_fixture(storage, deadline_millis~))
}

///|
test "typed handoff consumes receipt once and preserves its absolute deadline" {
  let now = Ref(10UL)
  let receipt_reads = Ref(0)
  let receipt_workspace = new_luna_receipt_workspace_with_clock(
    () => {
      receipt_reads.val += 1
      now.val
    },
    @inference.DeadlineBudget::from_milliseconds(100UL, fixture_inference()),
  )
  let receipt = receipt_workspace.begin()
  let (storage, source) = typed_fixture(fixture_identity("0"))
  now.val = 20UL
  let pool = fixture_preparation_pool(now, 1, 1, 8192UL)
  let admission = pool.try_begin_luna_text_handoff_with_receipt(receipt, source)
  assert_eq(admission.kind(), LunaRequestPreparationAdmitted)
  assert_eq(admission.consumed_bytes(), 1)
  assert_eq(receipt_reads.val, 2)
  let work = admission.work()
  assert_eq(pool.lanes[work.lane].receipt_at_millis, 10UL)
  assert_eq(pool.lanes[work.lane].hard_deadline.unwrap().at_millis(), 110UL)
  assert_eq(pool.lanes[work.lane].trusted_deadline_millis, 100UL)
  expect_receipt_error(
    () => ignore(receipt.remaining_millis()),
    Invalid(LunaPreparedLifecycle),
  )
  try source.prompt_length() catch {
    _ => ()
  } noraise {
    _ => fail("accepted typed source must be stale")
  }
  drive_preparation_ready(pool, work)
  let claim = work.take_prepared().take_claim()
  let request = claim.scheduler_request()
  assert_eq(request.request_id().value(), 71UL)
  assert_eq(request.sampling().temperature(), 0.75)
  assert_eq(request.sampling().top_k(), Some(11))
  assert_eq(request.sampling_seed().value(), 99UL)
  assert_true(claim.is_stop_token(9))
  claim.release()
  begin_typed_fixture(storage).release()
}

///|
test "typed saturation and draining retain both receipt and source for retry" {
  let now = Ref(10UL)
  let occupied = fixture_preparation_pool(now, 1, 1, 8192UL)
  let active = occupied
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  let receipt_workspace = trusted_receipt_workspace(now)
  let receipt = receipt_workspace.begin()
  let (_, source) = typed_fixture(fixture_identity("0"))
  let saturated = occupied.try_begin_luna_text_handoff_with_receipt(
    receipt, source,
  )
  assert_eq(saturated.kind(), LunaRequestPreparationSaturated)
  assert_eq(source.prompt_length(), 3)
  assert_true(receipt.remaining_millis() > 0)
  active.cancel()
  ignore(occupied.progress())
  let admitted = occupied.try_begin_luna_text_handoff_with_receipt(
    receipt, source,
  )
  assert_eq(admitted.kind(), LunaRequestPreparationAdmitted)
  admitted.work().cancel()
  ignore(occupied.progress())

  let draining = fixture_preparation_pool(now, 1, 1, 8192UL)
  draining.begin_drain()
  let drain_receipt = receipt_workspace.begin()
  let (_, drain_source) = typed_fixture(fixture_identity("0"))
  assert_eq(
    draining
    .try_begin_luna_text_handoff_with_receipt(drain_receipt, drain_source)
    .kind(),
    LunaRequestPreparationDraining,
  )
  assert_eq(drain_source.prompt_length(), 3)
  assert_true(drain_receipt.remaining_millis() > 0)
  drain_source.release()
  drain_receipt.abort()
}

///|
test "typed contract rejection is pre-transfer and cancellation releases claims" {
  let now = Ref(10UL)
  let pool = fixture_preparation_pool(now, 1, 1, 8192UL)
  let receipt_workspace = trusted_receipt_workspace(now)
  let wrong_receipt = receipt_workspace.begin()
  let (_, wrong_source) = typed_fixture(fixture_identity("2"))
  expect_receipt_error(
    () => {
      ignore(
        pool.try_begin_luna_text_handoff_with_receipt(
          wrong_receipt, wrong_source,
        ),
      )
    },
    Invalid(ModelIdentity),
  )
  assert_eq(wrong_source.prompt_length(), 3)
  assert_true(wrong_receipt.remaining_millis() > 0)
  wrong_source.release()
  wrong_receipt.abort()

  let receipt = receipt_workspace.begin()
  let (storage, source) = typed_fixture(fixture_identity("0"))
  let work = pool
    .try_begin_luna_text_handoff_with_receipt(receipt, source)
    .work()
  work.cancel()
  ignore(pool.progress())
  assert_eq(pool.free_lane_count(), 1)
  begin_typed_fixture(storage).release()
}
