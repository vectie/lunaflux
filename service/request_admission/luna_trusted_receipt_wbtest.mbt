///|
fn trusted_receipt_workspace(
  now : Ref[UInt64],
  millis? : UInt64 = 100UL,
) -> LunaRequestReceiptWorkspace raise {
  new_luna_receipt_workspace_with_clock(
    () => now.val,
    @inference.DeadlineBudget::from_milliseconds(millis, fixture_inference()),
  )
}

///|
fn expect_receipt_error(
  action : () -> Unit raise RequestAdmissionError,
  expected : RequestAdmissionError,
) -> Unit raise {
  try action() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("expected trusted receipt rejection")
  }
}

///|
test "trusted receipt is Busy stale abortable and generation bound" {
  let now = Ref(10UL)
  let workspace = trusted_receipt_workspace(now)
  let first = workspace.begin()
  assert_eq(first.remaining_millis(), 100)
  expect_receipt_error(
    () => ignore(workspace.begin()),
    Invalid(LunaPreparedLifecycle),
  )
  first.abort()
  expect_receipt_error(
    () => ignore(first.remaining_millis()),
    Invalid(LunaPreparedLifecycle),
  )
  let second = workspace.begin()
  expect_receipt_error(() => first.abort(), Invalid(LunaPreparedLifecycle))
  second.abort()
}

///|
test "trusted receipt reports expiry backward clock and clock failure safely" {
  let now = Ref(10UL)
  let workspace = trusted_receipt_workspace(now)
  let receipt = workspace.begin()
  now.val = 9UL
  expect_receipt_error(
    () => ignore(receipt.remaining_millis()),
    Invalid(ReceiptTime),
  )
  now.val = 110UL
  assert_eq(receipt.remaining_millis(), 0)
  let source : FixedArray[Byte] = [0]
  let pool = fixture_preparation_pool(now, 1, 1, 100000UL)
  expect_receipt_error(
    () => {
      ignore(
        pool.try_begin_luna_framed_with_receipt(
          receipt,
          source,
          source_offset=0,
          length=1,
        ),
      )
    },
    Invalid(Deadline),
  )
  receipt.abort()

  let fail_clock = Ref(false)
  let failed_workspace = new_luna_receipt_workspace_with_clock(
    () => if fail_clock.val { raise ClockUnavailable } else { 20UL },
    @inference.DeadlineBudget::from_milliseconds(100UL, fixture_inference()),
  )
  let failed = failed_workspace.begin()
  fail_clock.val = true
  expect_receipt_error(
    () => ignore(failed.remaining_millis()),
    ClockUnavailable,
  )
  fail_clock.val = false
  failed.abort()
}

///|
test "saturation draining and bad ranges leave trusted receipt live" {
  let now = Ref(10UL)
  let source : FixedArray[Byte] = [0]
  let occupied = fixture_preparation_pool(now, 1, 1, 100000UL)
  let active = occupied
    .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
    .work()
  let workspace = trusted_receipt_workspace(now)
  let receipt = workspace.begin()
  let saturated = occupied.try_begin_luna_framed_with_receipt(
    receipt,
    source,
    source_offset=0,
    length=1,
  )
  assert_eq(saturated.kind(), LunaRequestPreparationSaturated)
  assert_true(receipt.remaining_millis() > 0)
  active.cancel()
  ignore(occupied.progress())
  let admitted = occupied.try_begin_luna_framed_with_receipt(
    receipt,
    source,
    source_offset=0,
    length=1,
  )
  assert_eq(admitted.kind(), LunaRequestPreparationAdmitted)
  expect_receipt_error(
    () => ignore(receipt.remaining_millis()),
    Invalid(LunaPreparedLifecycle),
  )
  admitted.work().cancel()
  ignore(occupied.progress())

  let draining = fixture_preparation_pool(now, 1, 1, 100000UL)
  draining.begin_drain()
  let drain_receipt = workspace.begin()
  assert_eq(
    draining
    .try_begin_luna_framed_with_receipt(
      drain_receipt,
      source,
      source_offset=0,
      length=1,
    )
    .kind(),
    LunaRequestPreparationDraining,
  )
  assert_true(drain_receipt.remaining_millis() > 0)
  expect_receipt_error(
    () => {
      ignore(
        occupied.try_begin_luna_framed_with_receipt(
          drain_receipt,
          source,
          source_offset=1,
          length=1,
        ),
      )
    },
    Invalid(FrameContract),
  )
  assert_true(drain_receipt.remaining_millis() > 0)
  drain_receipt.abort()
}

///|
test "trusted receipt preserves first-byte time and requires wire budget equality" {
  let now = Ref(10UL)
  let workspace = trusted_receipt_workspace(now, millis=50UL)
  let receipt = workspace.begin()
  now.val = 20UL
  let (source, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0"), deadline_millis=100UL),
  )
  let pool = fixture_preparation_pool(now, 1, 17, 100000UL)
  let admission = pool.try_begin_luna_framed_with_receipt(
    receipt,
    source,
    source_offset=0,
    length=1,
  )
  assert_eq(admission.kind(), LunaRequestPreparationAdmitted)
  let work = admission.work()
  assert_eq(pool.lanes[work.lane].receipt_at_millis, 10UL)
  assert_eq(pool.lanes[work.lane].trusted_deadline_millis, 50UL)
  direct_hostile_offer_complete(
    pool,
    work,
    source,
    frame_length,
    admission.consumed_bytes(),
  )
  while work.state() == LunaRequestPreparationPending {
    ignore(pool.progress())
  }
  assert_eq(work.state(), LunaRequestPreparationFailed)
  assert_eq(work.failure(), Invalid(Deadline))
  work.cancel()
}

///|
test "trusted receipt epoch exhaustion is pre-mutation" {
  let now = Ref(10UL)
  let workspace = trusted_receipt_workspace(now)
  workspace.epoch = 0xffffffffffffffffUL
  expect_receipt_error(
    () => ignore(workspace.begin()),
    Invalid(LunaPreparationPool),
  )
  assert_eq(workspace.phase, LUNA_TRUSTED_RECEIPT_IDLE)
}

///|
test "cross-envelope receipt budget rejects before pool or receipt mutation" {
  let wider = @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=16,
    max_context_tokens=64,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=2000UL,
    max_top_k=256,
    max_temperature=4.0,
  )
  let now = Ref(10UL)
  let workspace = new_luna_receipt_workspace_with_clock(
    () => now.val,
    @inference.DeadlineBudget::from_milliseconds(2000UL, wider),
  )
  let receipt = workspace.begin()
  let pool = fixture_preparation_pool(now, 1, 1, 100000UL)
  let source : FixedArray[Byte] = [0]
  expect_receipt_error(
    () => {
      ignore(
        pool.try_begin_luna_framed_with_receipt(
          receipt,
          source,
          source_offset=0,
          length=1,
        ),
      )
    },
    Invalid(Deadline),
  )
  assert_eq(pool.free_lane_count(), 1)
  assert_eq(pool.active_count, 0)
  assert_true(receipt.remaining_millis() > 0)
  receipt.abort()
}
