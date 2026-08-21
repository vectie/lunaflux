///|
fn framed_pool_source(
  request : @inference.GenerateRequest,
) -> (FixedArray[Byte], Int) raise {
  let limits = @framed_wire.FramedWireLimits::new(
    fixture_inference(),
    max_frame_bytes=4096,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(limits).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, frame.length())
}

///|
fn framed_token_request() -> @inference.GenerateRequest raise {
  let limits = fixture_inference()
  let base = fixture_request(fixture_identity("0"))
  @inference.GenerateRequest::new(
    protocol_version=base.protocol_version(),
    request_id=base.request_id(),
    model_identity=base.model_identity(),
    input=@inference.Input::from_token_ids([3, 5, 7], limits),
    max_new_tokens=base.max_new_tokens(),
    context_ceiling=base.context_ceiling(),
    sampling=@inference.SamplingParameters::sample(
      temperature=0.75,
      top_k=8,
      top_p=0.9,
      limits,
    ),
    sampling_seed=base.sampling_seed(),
    stops=base.stops(),
    stream_preference=base.stream_preference(),
    deadline=base.deadline(),
    cache=base.cache(),
    trace=base.trace(),
    limits~,
  )
}

///|
fn drive_framed_preparation(
  pool : LunaRequestPreparationPool,
  source : FixedArray[Byte],
  length : Int,
  chunk : Int,
) -> LunaRequestPreparationWork raise {
  let initial = Int::min(chunk, length)
  let admission = pool.try_begin_luna_framed(
    source,
    source_offset=0,
    length=initial,
  )
  assert_eq(admission.kind(), LunaRequestPreparationAdmitted)
  assert_true(admission.consumed_bytes() > 0)
  let work = admission.work()
  let mut offset = admission.consumed_bytes()
  let mut turns = 0
  while offset < length {
    let lane = pool.lanes[work.lane]
    if lane.phase == LUNA_PREPARATION_FRAMED_RECEIVING {
      let offered = Int::min(chunk, length - offset)
      let consumed = work.offer_luna_framed(
        source,
        source_offset=offset,
        length=offered,
      )
      if consumed > 0 {
        offset += consumed
      } else {
        ignore(pool.progress())
      }
    } else {
      ignore(pool.progress())
    }
    turns += 1
    if turns > 20000 {
      fail("direct framed receipt did not consume the frame")
    }
  }
  while work.state() is LunaRequestPreparationPending {
    ignore(pool.progress())
    turns += 1
    if turns > 40000 {
      fail("direct framed preparation did not converge")
    }
  }
  work
}

///|
test "direct framed preparation is equivalent across every chunk size" {
  let request = fixture_request(fixture_identity("0"))
  let (source, length) = framed_pool_source(request)
  for chunk in 1..<=length {
    let now = Ref(10UL)
    let pool = fixture_preparation_pool(now, 1, 17, 100000UL)
    let work = drive_framed_preparation(pool, source, length, chunk)
    assert_eq(work.state(), LunaRequestPreparationReady)
    let prepared = work.take_prepared()
    assert_eq(prepared.input_token_count(), 3)
    let claim = prepared.take_claim()
    let scheduler_request = claim.scheduler_request()
    assert_eq(scheduler_request.request_id().value(), 7UL)
    assert_eq(scheduler_request.model_identity(), fixture_identity("0"))
    assert_true(claim.is_stop_token(9))
    claim.release()
    assert_eq(pool.free_lane_count(), 1)
  }
}

///|
test "direct framed token IDs bind sampling without object-form preparation" {
  let (source, length) = framed_pool_source(framed_token_request())
  let now = Ref(10UL)
  let pool = fixture_preparation_pool(now, 1, 1, 100000UL)
  let work = drive_framed_preparation(pool, source, length, 17)
  let prepared = work.take_prepared()
  assert_eq(prepared.input_token_count(), 3)
  let claim = prepared.take_claim()
  let request = claim.scheduler_request()
  assert_eq(request.sampling().mode(), Sample)
  assert_eq(request.sampling().temperature(), 0.75)
  assert_eq(request.sampling().top_k(), Some(8))
  assert_eq(request.sampling().top_p(), Some(0.9))
  claim.release()
}

///|
test "direct framed saturation and drain consume no bytes or clock" {
  let request = fixture_request(fixture_identity("0"))
  let (source, length) = framed_pool_source(request)
  let reads = Ref(0)
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let pool = new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    inference,
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    () => {
      reads.val += 1
      10UL
    },
    1,
    LunaRequestPreparationStepBudget::new(1),
    LunaRequestPreparationWorkLimit::new(100000UL),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=1024UL,
    ),
  )
  let first = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  assert_eq(first.consumed_bytes(), 1)
  assert_eq(reads.val, 1)
  let saturated = pool.try_begin_luna_framed(source, source_offset=0, length~)
  assert_eq(saturated.kind(), LunaRequestPreparationSaturated)
  assert_eq(saturated.consumed_bytes(), 0)
  assert_eq(reads.val, 1)
  first.work().cancel()
  ignore(pool.progress())
  pool.begin_drain()
  let draining = pool.try_begin_luna_framed(source, source_offset=0, length~)
  assert_eq(draining.kind(), LunaRequestPreparationDraining)
  assert_eq(draining.consumed_bytes(), 0)
  assert_eq(reads.val, 1)
}

///|
test "partial framed receipt is retired by drain and stale after lane reuse" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let now = Ref(10UL)
  let pool = fixture_preparation_pool(now, 1, 1, 100000UL)
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  let stale = admission.work()
  pool.begin_drain()
  ignore(pool.progress())
  assert_true(pool.is_drained())
  try stale.state() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected stale framed work error")
  } noraise {
    _ => fail("drained framed work retained authority")
  }
}

///|
test "drain preserves a fully received frame awaiting scanner progress" {
  let (source, length) = framed_pool_source(
    fixture_request(fixture_identity("0")),
  )
  let now = Ref(10UL)
  let pool = fixture_preparation_pool(now, 1, 17, 100000UL)
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length~)
  let work = admission.work()
  let mut offset = admission.consumed_bytes()
  while offset < length {
    if pool.lanes[work.lane].phase == LUNA_PREPARATION_FRAMED_RECEIVING {
      let consumed = work.offer_luna_framed(
        source,
        source_offset=offset,
        length=length - offset,
      )
      if consumed == 0 {
        ignore(pool.progress())
      } else {
        offset += consumed
      }
    } else {
      ignore(pool.progress())
    }
  }
  assert_eq(pool.lanes[work.lane].phase, LUNA_PREPARATION_FRAMED_RECEIVING)
  pool.begin_drain()
  while work.state() is LunaRequestPreparationPending {
    ignore(pool.progress())
  }
  assert_eq(work.state(), LunaRequestPreparationReady)
  work.take_prepared().discard()
  assert_true(pool.is_drained())
}

///|
test "wrong framed model fails before input or semantic authority starts" {
  let (source, length) = framed_pool_source(
    fixture_request(fixture_identity("1")),
  )
  let now = Ref(10UL)
  let pool = fixture_preparation_pool(now, 1, 17, 100000UL)
  let work = drive_framed_preparation(pool, source, length, 17)
  assert_eq(work.state(), LunaRequestPreparationFailed)
  assert_eq(work.failure(), Invalid(ModelIdentity))
  let lane = pool.lanes[work.lane]
  assert_true(lane.tokenizer_input_write is None)
  assert_true(lane.token_write is None)
  assert_true(lane.semantic_write.is_empty())
  work.discard()
  assert_eq(pool.free_lane_count(), 1)
}

///|
test "empty continuation is zero-cost and does not sample the clock" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let reads = Ref(0)
  let tokenizer = fixture_tokenizer()
  let inference = fixture_inference()
  let pool = new_pool_with_clock(
    tokenizer,
    tokenizer.digest(),
    fixture_identity("0"),
    inference,
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    () => {
      reads.val += 1
      10UL
    },
    1,
    LunaRequestPreparationStepBudget::new(1),
    LunaRequestPreparationWorkLimit::new(100000UL),
    LunaRequestPreparationStorageBudget::new(
      int_cells=1000000UL,
      byte_cells=1000000UL,
      reference_cells=1024UL,
    ),
  )
  let work = pool
    .try_begin_luna_framed(source, source_offset=0, length=1)
    .work()
  assert_eq(reads.val, 1)
  assert_eq(work.offer_luna_framed(source, source_offset=1, length=0), 0)
  assert_eq(work.last_work_units(), 0)
  assert_eq(reads.val, 1)
  work.cancel()
  ignore(pool.progress())
}
