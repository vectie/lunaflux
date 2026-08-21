///|
fn hostile_pool_with_clock(
  clock_read : () -> UInt64 raise RequestAdmissionError,
  lane_count : Int,
  step_work_units : Int,
  total_work_units : UInt64,
) -> LunaRequestPreparationPool raise {
  let tokenizer = fixture_tokenizer()
  new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    fixture_inference(),
    clock_read,
    lane_count,
    LunaRequestPreparationStepBudget::new(step_work_units),
    LunaRequestPreparationWorkLimit::new(total_work_units),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=2048UL,
    ),
  )
}

///|
fn hostile_work(
  pool : LunaRequestPreparationPool,
) -> LunaRequestPreparationWork raise {
  pool
  .try_submit(pool_received_at(fixture_request(fixture_identity("0")), 10UL))
  .work()
}

///|
fn drive_to_phase(
  pool : LunaRequestPreparationPool,
  work : LunaRequestPreparationWork,
  phase : Int,
) -> Unit raise {
  let lane = pool.lanes[work.lane]
  let mut turns = 0
  while lane.phase != phase {
    assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
    turns += 1
    if turns > 4096 {
      fail("preparation phase did not converge")
    }
  }
}

///|
fn drive_terminal_work(
  pool : LunaRequestPreparationPool,
  work : LunaRequestPreparationWork,
) -> Unit raise {
  let mut turns = 0
  while work.state() is LunaRequestPreparationPending {
    assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
    turns += 1
    if turns > 4096 {
      fail("preparation work did not terminate")
    }
  }
}

///|
test "deadline cuts Ready before take and Prepared before claim" {
  let ready_now = Ref(11UL)
  let ready_pool = fixture_preparation_pool(ready_now, 1, 1, 4096UL)
  let ready = hostile_work(ready_pool)
  drive_preparation_ready(ready_pool, ready)
  ready_now.val = 111UL
  try ready.take_prepared() catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected Ready expiry error")
  } noraise {
    _ => fail("expired Ready work published a prepared shell")
  }
  assert_eq(ready.state(), LunaRequestPreparationFailed)
  ready.discard()
  assert_eq(ready_pool.free_lane_count(), 1)

  let prepared_now = Ref(11UL)
  let prepared_pool = fixture_preparation_pool(prepared_now, 1, 1, 4096UL)
  let prepared_work = hostile_work(prepared_pool)
  drive_preparation_ready(prepared_pool, prepared_work)
  let prepared = prepared_work.take_prepared()
  prepared_now.val = 111UL
  try prepared.take_claim() catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected Prepared expiry error")
  } noraise {
    _ => fail("expired Prepared shell published a claim")
  }
  assert_eq(prepared_pool.free_lane_count(), 1)
  try prepared.input_token_count() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected expired-shell replay error")
  } noraise {
    _ => fail("expired Prepared shell retained authority")
  }
}

///|
test "backward and failing clocks poison only the exact lane" {
  let observed = Ref(11UL)
  let backward_pool = hostile_pool_with_clock(() => observed.val, 1, 1, 4096UL)
  let backward = hostile_work(backward_pool)
  observed.val = 9UL
  assert_eq(backward_pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(backward.state(), LunaRequestPreparationFailed)
  assert_eq(backward.failure(), Invalid(ReceiptTime))
  backward.discard()
  assert_eq(backward_pool.free_lane_count(), 1)

  let reads = Ref(0)
  let failing_pool = hostile_pool_with_clock(
    () => {
      reads.val += 1
      if reads.val == 1 {
        11UL
      } else {
        raise ClockUnavailable
      }
    },
    1,
    1,
    4096UL,
  )
  let failing = hostile_work(failing_pool)
  assert_eq(failing_pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(failing.state(), LunaRequestPreparationFailed)
  assert_eq(failing.failure(), ClockUnavailable)
  failing.discard()
  assert_eq(failing_pool.free_lane_count(), 1)
}

///|
test "total work is quantum invariant and exact ceilings are closed" {
  let one_now = Ref(11UL)
  let one_pool = fixture_preparation_pool(one_now, 1, 1, 4096UL)
  let one = hostile_work(one_pool)
  drive_preparation_ready(one_pool, one)
  let exact = one.total_work_units()
  one.discard()

  let seventeen_now = Ref(11UL)
  let seventeen_pool = fixture_preparation_pool(seventeen_now, 1, 17, 4096UL)
  let seventeen = hostile_work(seventeen_pool)
  drive_preparation_ready(seventeen_pool, seventeen)
  assert_eq(seventeen.total_work_units(), exact)
  seventeen.discard()

  let short_now = Ref(11UL)
  let short_pool = fixture_preparation_pool(short_now, 1, 17, exact - 1UL)
  let short = hostile_work(short_pool)
  drive_terminal_work(short_pool, short)
  assert_eq(short.state(), LunaRequestPreparationFailed)
  assert_eq(short.failure(), Invalid(LunaPreparationWork))
  short.discard()

  let exact_now = Ref(11UL)
  let exact_pool = fixture_preparation_pool(exact_now, 1, 17, exact)
  let exact_work = hostile_work(exact_pool)
  drive_preparation_ready(exact_pool, exact_work)
  assert_eq(exact_work.total_work_units(), exact)
  exact_work.discard()
}

///|
test "cancel during token copy and output setup returns the lane" {
  let copy_now = Ref(11UL)
  let copy_pool = fixture_preparation_pool(copy_now, 1, 1, 4096UL)
  let copy_work = hostile_work(copy_pool)
  drive_to_phase(copy_pool, copy_work, LUNA_PREPARATION_TEXT_COPY)
  copy_work.cancel()
  assert_eq(copy_work.state(), LunaRequestPreparationCancelRequested)
  assert_eq(copy_pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(copy_pool.free_lane_count(), 1)

  let output_now = Ref(11UL)
  let output_pool = fixture_preparation_pool(output_now, 1, 1, 4096UL)
  let output_work = hostile_work(output_pool)
  drive_to_phase(output_pool, output_work, LUNA_PREPARATION_OUTPUT_SETUP)
  output_work.cancel()
  assert_eq(output_pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(output_pool.free_lane_count(), 1)
}

///|
test "drain waits for Pending Ready Prepared Claimed and Failed authority" {
  let pending_now = Ref(11UL)
  let pending_pool = fixture_preparation_pool(pending_now, 1, 1, 4096UL)
  let pending = hostile_work(pending_pool)
  pending_pool.begin_drain()
  assert_false(pending_pool.is_drained())
  pending.cancel()
  ignore(pending_pool.progress())
  assert_true(pending_pool.is_drained())

  let ready_now = Ref(11UL)
  let ready_pool = fixture_preparation_pool(ready_now, 1, 1, 4096UL)
  let ready = hostile_work(ready_pool)
  drive_preparation_ready(ready_pool, ready)
  ready_pool.begin_drain()
  assert_false(ready_pool.is_drained())
  ready.discard()
  assert_true(ready_pool.is_drained())

  let prepared_now = Ref(11UL)
  let prepared_pool = fixture_preparation_pool(prepared_now, 1, 1, 4096UL)
  let prepared_work = hostile_work(prepared_pool)
  drive_preparation_ready(prepared_pool, prepared_work)
  let prepared = prepared_work.take_prepared()
  prepared_pool.begin_drain()
  assert_false(prepared_pool.is_drained())
  prepared.discard()
  assert_true(prepared_pool.is_drained())

  let claim_now = Ref(11UL)
  let claim_pool = fixture_preparation_pool(claim_now, 1, 1, 4096UL)
  let claim_work = hostile_work(claim_pool)
  drive_preparation_ready(claim_pool, claim_work)
  let claim = claim_work.take_prepared().take_claim()
  claim_pool.begin_drain()
  assert_false(claim_pool.is_drained())
  claim.release()
  assert_true(claim_pool.is_drained())

  let failed_now = Ref(11UL)
  let failed_pool = fixture_preparation_pool(failed_now, 1, 1, 4096UL)
  let failed = hostile_work(failed_pool)
  failed_now.val = 111UL
  ignore(failed_pool.progress())
  failed_pool.begin_drain()
  assert_false(failed_pool.is_drained())
  failed.discard()
  assert_true(failed_pool.is_drained())
}

///|
test "lane bounds storage and exhausted generation reject before consumption" {
  for lanes in [0, LUNA_PREPARATION_MAX_LANES + 1] {
    try hostile_pool_with_clock(() => 11UL, lanes, 1, 4096UL) catch {
      Invalid(LunaPreparationPool) => ()
      _ => fail("unexpected lane-bound error")
    } noraise {
      _ => fail("invalid lane count was accepted")
    }
  }
  let tokenizer = fixture_tokenizer()
  try
    new_pool_with_clock(
      tokenizer,
      tokenizer.digest(),
      fixture_identity("0"),
      fixture_inference(),
      () => 11UL,
      1,
      LunaRequestPreparationStepBudget::new(1),
      LunaRequestPreparationWorkLimit::new(4096UL),
      LunaRequestPreparationStorageBudget::new(
        int_cells=1UL,
        byte_cells=1UL,
        reference_cells=1UL,
      ),
    )
  catch {
    Invalid(LunaPreparationStorage) => ()
    _ => fail("unexpected storage-bound error")
  } noraise {
    _ => fail("undersized preparation storage was accepted")
  }
  try
    new_pool_with_clock(
      tokenizer,
      tokenizer.digest(),
      fixture_identity("0"),
      fixture_inference(),
      () => 11UL,
      1,
      LunaRequestPreparationStepBudget::new(1),
      LunaRequestPreparationWorkLimit::new(4096UL),
      LunaRequestPreparationStorageBudget::new(
        int_cells=1000000UL,
        byte_cells=1000000UL,
        reference_cells=1UL,
      ),
    )
  catch {
    Invalid(LunaPreparationStorage) => ()
    _ => fail("unexpected output-reference storage-bound error")
  } noraise {
    _ => fail("missing output semantic-view slot was accepted")
  }
  let exact_references = new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    fixture_inference(),
    () => 11UL,
    1,
    LunaRequestPreparationStepBudget::new(1),
    LunaRequestPreparationWorkLimit::new(4096UL),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=2UL,
    ),
  )
  exact_references.begin_drain()
  assert_true(exact_references.is_drained())

  let exhausted = hostile_pool_with_clock(() => 11UL, 1, 1, 4096UL)
  let received = pool_received_at(fixture_request(fixture_identity("0")), 10UL)
  let lane = exhausted.free_ring[exhausted.free_head]
  exhausted.lanes[lane].generation = 0xffffffffffffffffUL
  let disposition = exhausted.try_submit(received)
  assert_eq(disposition.kind(), LunaRequestPreparationDraining)
  assert_false(received.consumed)
  assert_eq(exhausted.free_lane_count(), 1)
  assert_true(exhausted.is_drained())
}
