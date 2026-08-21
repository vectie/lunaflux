///|
fn direct_hostile_pool(
  read_clock : () -> UInt64 raise RequestAdmissionError,
  lanes : Int,
  budget : Int,
  work_limit : UInt64,
) -> LunaRequestPreparationPool raise {
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    inference,
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    read_clock,
    lanes,
    LunaRequestPreparationStepBudget::new(budget),
    LunaRequestPreparationWorkLimit::new(work_limit),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=2048UL,
    ),
  )
}

///|
fn direct_hostile_offer_complete(
  pool : LunaRequestPreparationPool,
  work : LunaRequestPreparationWork,
  source : FixedArray[Byte],
  frame_length : Int,
  start : Int,
) -> Unit raise {
  let mut offset = start
  let mut turns = 0
  while offset < frame_length {
    let lane = pool.lanes[work.lane]
    if lane.phase == LUNA_PREPARATION_FRAMED_RECEIVING {
      let consumed = work.offer_luna_framed(
        source,
        source_offset=offset,
        length=source.length() - offset,
      )
      assert_true(consumed >= 0)
      assert_true(consumed <= frame_length - offset)
      if consumed == 0 {
        ignore(pool.progress())
      } else {
        offset += consumed
      }
    } else {
      ignore(pool.progress())
    }
    turns += 1
    if turns > 20000 {
      fail("direct framed hostile offer did not converge")
    }
  }
  assert_eq(offset, frame_length)
}

///|
fn direct_hostile_drive_to_phase(
  pool : LunaRequestPreparationPool,
  work : LunaRequestPreparationWork,
  phase : Int,
) -> Unit raise {
  let mut turns = 0
  while pool.lanes[work.lane].phase != phase {
    assert_eq(work.state(), LunaRequestPreparationPending)
    ignore(pool.progress())
    turns += 1
    if turns > 20000 {
      fail("direct framed hostile phase did not converge")
    }
  }
}

///|
test "direct framed tail is unconsumed and a fully received drain completes" {
  let (frame, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0")),
  )
  let source = FixedArray::make(frame_length + 7, b'Z')
  for index in 0..<frame_length {
    source[index] = frame[index]
  }
  let pool = direct_hostile_pool(() => 10UL, 1, 17, 100000UL)
  let admission = pool.try_begin_luna_framed(
    source,
    source_offset=0,
    length=source.length(),
  )
  let work = admission.work()
  direct_hostile_offer_complete(
    pool,
    work,
    source,
    frame_length,
    admission.consumed_bytes(),
  )
  for index in frame_length..<source.length() {
    assert_eq(source[index], b'Z')
  }
  pool.begin_drain()
  while work.state() is LunaRequestPreparationPending {
    ignore(pool.progress())
  }
  assert_eq(work.state(), LunaRequestPreparationReady)
  work.take_prepared().discard()
  assert_true(pool.is_drained())
}

///|
test "partial framed drain bypasses a failed clock and stales its work" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let reads = Ref(0)
  let pool = direct_hostile_pool(
    () => {
      reads.val += 1
      if reads.val == 1 {
        10UL
      } else {
        raise ClockUnavailable
      }
    },
    1,
    1,
    100000UL,
  )
  let work = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  assert_eq(reads.val, 1)
  pool.begin_drain()
  ignore(pool.progress())
  assert_eq(reads.val, 1)
  assert_true(pool.is_drained())
  try work.state() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("drained direct framed work produced the wrong stale error")
  } noraise {
    _ => fail("drained direct framed work retained authority")
  }
}

///|
test "invalid initial framed ranges are atomic before receipt capture" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let reads = Ref(0)
  let pool = direct_hostile_pool(
    () => {
      reads.val += 1
      10UL
    },
    1,
    1,
    100000UL,
  )
  try
    pool.try_begin_luna_framed(source, source_offset=0, length=0) |> ignore
  catch {
    Invalid(FrameContract) => ()
    _ => fail("empty initial framed range produced the wrong error")
  } noraise {
    _ => fail("empty initial framed range was accepted")
  }
  assert_eq(reads.val, 0)
  assert_eq(pool.free_lane_count(), 1)
  try
    pool.try_begin_luna_framed(source, source_offset=source.length(), length=1)
    |> ignore
  catch {
    Invalid(FrameContract) => ()
    _ => fail("invalid initial framed offset produced the wrong error")
  } noraise {
    _ => fail("invalid initial framed offset was accepted")
  }
  assert_eq(reads.val, 0)
  assert_eq(pool.free_lane_count(), 1)
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  assert_eq(admission.kind(), LunaRequestPreparationAdmitted)
  assert_eq(admission.consumed_bytes(), 1)
  assert_eq(reads.val, 1)
  admission.work().cancel()
  ignore(pool.progress())
  assert_eq(pool.free_lane_count(), 1)
}

///|
test "pre-View hard receipt deadline expires and frees scanner authority" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let inference = fixture_inference()
  let now = Ref(10UL)
  let pool = direct_hostile_pool(() => now.val, 1, 1, 100000UL)
  let work = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  let scanner = pool.lanes[work.lane].require_framed_work()
  assert_eq(work.total_work_units(), 1UL)
  now.val = 10UL + inference.max_deadline_millis()
  ignore(pool.progress())
  assert_eq(work.state(), LunaRequestPreparationFailed)
  assert_eq(work.failure(), Invalid(Deadline))
  assert_eq(work.last_work_units(), 0)
  assert_eq(work.total_work_units(), 1UL)
  try scanner.state() catch {
    LunaFramedRequestStale => ()
    _ => fail("expired pre-View scanner produced the wrong stale error")
  } noraise {
    _ => fail("expired pre-View scanner retained authority")
  }
  work.discard()
  assert_eq(pool.free_lane_count(), 1)
  let reused = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  reused.cancel()
  ignore(pool.progress())
}

///|
test "framed receiving rotates at zero cost before scanner work" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let pool = direct_hostile_pool(() => 10UL, 2, 1, 100000UL)
  let first = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  let second = pool
    .try_begin_luna_framed(source, source_offset=0, length=16)
    .work()
  for offset in 1..<16 {
    assert_eq(
      second.offer_luna_framed(source, source_offset=offset, length=1),
      1,
    )
  }
  assert_eq(first.total_work_units(), 1UL)
  assert_eq(second.total_work_units(), 16UL)
  ignore(pool.progress())
  assert_eq(first.last_work_units(), 0)
  assert_eq(first.total_work_units(), 1UL)
  ignore(pool.progress())
  assert_eq(second.last_work_units(), 0)
  assert_eq(second.total_work_units(), 16UL)
  ignore(pool.progress())
  assert_eq(first.last_work_units(), 0)
  ignore(pool.progress())
  assert_eq(second.last_work_units(), 1)
  assert_eq(second.total_work_units(), 17UL)
  first.cancel()
  second.cancel()
  ignore(pool.progress())
  ignore(pool.progress())
  assert_eq(pool.free_lane_count(), 2)
}

///|
test "direct framed exact deadline fails before Ready and releases its View" {
  let (source, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0"), deadline_millis=1UL),
  )
  let now = Ref(10UL)
  let pool = direct_hostile_pool(() => now.val, 1, 1, 100000UL)
  let admission = pool.try_begin_luna_framed(
    source,
    source_offset=0,
    length=frame_length,
  )
  let work = admission.work()
  direct_hostile_offer_complete(
    pool,
    work,
    source,
    frame_length,
    admission.consumed_bytes(),
  )
  direct_hostile_drive_to_phase(
    pool,
    work,
    LUNA_PREPARATION_FRAMED_BIND_SCALARS,
  )
  let lane = pool.lanes[work.lane]
  assert_true(lane.request is None)
  let stale_view = lane.require_framed_view()
  now.val = 11UL
  ignore(pool.progress())
  assert_eq(work.state(), LunaRequestPreparationFailed)
  assert_eq(work.failure(), Invalid(Deadline))
  assert_true(lane.claim is None)
  try stale_view.length() catch {
    LunaFramedRequestStale => ()
    _ => fail("expired direct framed View produced the wrong stale error")
  } noraise {
    _ => fail("expired direct framed View retained frame authority")
  }
  work.discard()
  assert_eq(pool.free_lane_count(), 1)
}

///|
test "post-take cancellation releases View and authenticates scanner reuse" {
  let (source, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0")),
  )
  let pool = direct_hostile_pool(() => 10UL, 1, 17, 100000UL)
  let admission = pool.try_begin_luna_framed(
    source,
    source_offset=0,
    length=frame_length,
  )
  let work = admission.work()
  direct_hostile_offer_complete(
    pool,
    work,
    source,
    frame_length,
    admission.consumed_bytes(),
  )
  direct_hostile_drive_to_phase(pool, work, LUNA_PREPARATION_FRAMED_BIND_MODEL)
  let stale_view = pool.lanes[work.lane].require_framed_view()
  work.cancel()
  ignore(pool.progress())
  try stale_view.request_id_value() catch {
    LunaFramedRequestStale => ()
    _ => fail("cancelled direct framed View produced the wrong stale error")
  } noraise {
    _ => fail("cancelled direct framed View retained frame authority")
  }
  let reused = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  assert_eq(reused.total_work_units(), 1UL)
  reused.cancel()
  ignore(pool.progress())
}

///|
test "direct framed total work is invariant across outer quanta" {
  let (source, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0")),
  )
  let one = direct_hostile_pool(() => 10UL, 1, 1, 100000UL)
  let one_work = drive_framed_preparation(one, source, frame_length, 1)
  let total = one_work.total_work_units()
  assert_eq(one_work.last_work_units(), 1)
  one_work.discard()

  let wide = direct_hostile_pool(() => 10UL, 1, 17, 100000UL)
  let wide_work = drive_framed_preparation(wide, source, frame_length, 17)
  assert_eq(wide_work.total_work_units(), total)
  assert_true(wide_work.last_work_units() > 0)
  assert_true(wide_work.last_work_units() <= 17)
  assert_true(wide.lanes[wide_work.lane].request is None)
  wide_work.discard()
}

///|
test "receipt-complete proof starts at the final byte and survives import" {
  let request = fixture_request(fixture_identity("0"))
  let object_pool = direct_hostile_pool(() => 10UL, 1, 17, 100000UL)
  let object_work = object_pool
    .try_submit(pool_received_at(request, 10UL))
    .work()
  assert_false(object_work.luna_framed_receipt_complete())
  object_work.cancel()
  ignore(object_pool.progress())

  let (source, frame_length) = framed_pool_source(request)
  let clock_enabled = Ref(true)
  let clock_reads = Ref(0)
  let pool = direct_hostile_pool(
    () => {
      clock_reads.val += 1
      if !clock_enabled.val {
        raise ClockUnavailable
      }
      10UL
    },
    1,
    1,
    100000UL,
  )
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  let work = admission.work()
  assert_eq(admission.consumed_bytes(), 1)
  for offset in 1..<16 {
    assert_eq(work.offer_luna_framed(source, source_offset=offset, length=1), 1)
    assert_false(work.luna_framed_receipt_complete())
  }
  // Prefix validation uses the scanner's Validating state too; it is not a
  // complete receipt and must not publish body-complete proof.
  assert_false(work.luna_framed_receipt_complete())
  ignore(pool.progress())
  assert_false(work.luna_framed_receipt_complete())
  assert_eq(pool.lanes[work.lane].phase, LUNA_PREPARATION_FRAMED_SCANNING)
  let reads_before_backpressure = clock_reads.val
  clock_enabled.val = false
  assert_eq(work.offer_luna_framed(source, source_offset=-1, length=-1), 0)
  assert_eq(clock_reads.val, reads_before_backpressure)
  clock_enabled.val = true
  while pool.lanes[work.lane].require_framed_work().state() !=
        LunaFramedRequestReceivingBody {
    ignore(pool.progress())
    assert_false(work.luna_framed_receipt_complete())
  }
  assert_eq(work.offer_luna_framed(source, source_offset=16, length=1), 1)
  assert_false(work.luna_framed_receipt_complete())
  direct_hostile_offer_complete(pool, work, source, frame_length, 17)
  // No pool progress is required after the final accepted byte.
  assert_true(work.luna_framed_receipt_complete())
  ignore(pool.progress())
  assert_true(work.luna_framed_receipt_complete())
  while work.state() == LunaRequestPreparationPending {
    ignore(pool.progress())
    assert_true(work.luna_framed_receipt_complete())
  }
  assert_eq(work.state(), LunaRequestPreparationReady)
  assert_true(work.luna_framed_receipt_complete())
  let prepared = work.take_prepared()
  try work.luna_framed_receipt_complete() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("taken receipt proof produced the wrong stale error")
  } noraise {
    _ => fail("taken preparation retained receipt proof")
  }
  prepared.discard()
  let reused = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  assert_false(reused.luna_framed_receipt_complete())
  try work.luna_framed_receipt_complete() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("reused receipt proof produced the wrong stale error")
  } noraise {
    _ => fail("old generation observed reused receipt state")
  }
  reused.cancel()
  ignore(pool.progress())
}

///|
test "failed and cancelled preparations cannot expose receipt proof" {
  let (corrupt, frame_length) = framed_pool_source(
    fixture_request(fixture_identity("0")),
  )
  corrupt[frame_length - 1] = corrupt[frame_length - 1] ^ b'\x01'
  let failed_pool = direct_hostile_pool(() => 10UL, 1, 17, 100000UL)
  let failed_admission = failed_pool.try_begin_luna_framed(
    corrupt,
    source_offset=0,
    length=1,
  )
  let failed_work = failed_admission.work()
  direct_hostile_offer_complete(
    failed_pool,
    failed_work,
    corrupt,
    frame_length,
    failed_admission.consumed_bytes(),
  )
  assert_true(failed_work.luna_framed_receipt_complete())
  while failed_work.state() == LunaRequestPreparationPending {
    ignore(failed_pool.progress())
  }
  assert_eq(failed_work.state(), LunaRequestPreparationFailed)
  try failed_work.luna_framed_receipt_complete() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("failed receipt proof produced the wrong error")
  } noraise {
    _ => fail("failed preparation exposed receipt proof")
  }
  failed_work.discard()

  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let cancelled_pool = direct_hostile_pool(() => 10UL, 1, 1, 100000UL)
  let cancelled = cancelled_pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  assert_false(cancelled.luna_framed_receipt_complete())
  cancelled.cancel()
  try cancelled.luna_framed_receipt_complete() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("cancelled receipt proof produced the wrong error")
  } noraise {
    _ => fail("cancelled preparation exposed receipt proof")
  }
  ignore(cancelled_pool.progress())
}
