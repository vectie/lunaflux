///|
test "fragmented Luna receipt reports authenticated awaiting input" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let clock_reads = Ref(0)
  let pool = direct_hostile_pool(
    () => {
      clock_reads.val += 1
      10UL
    },
    1,
    1,
    100000UL,
  )
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  let work = admission.work()
  assert_eq(admission.consumed_bytes(), 1)
  let total_before = work.total_work_units()
  assert_eq(pool.progress(), LunaRequestPreparationPoolAwaitingInput)
  assert_eq(work.last_work_units(), 0)
  assert_eq(work.total_work_units(), total_before)
  assert_eq(clock_reads.val, 2)
  work.cancel()
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(pool.free_lane_count(), 1)
}

///|
test "drain at scanner receiving boundary never reports awaiting input" {
  let (source, _) = framed_pool_source(fixture_request(fixture_identity("0")))
  let pool = direct_hostile_pool(() => 10UL, 1, 1, 100000UL)
  let admission = pool.try_begin_luna_framed(source, source_offset=0, length=1)
  let work = admission.work()
  for offset in 1..<16 {
    assert_eq(work.offer_luna_framed(source, source_offset=offset, length=1), 1)
  }
  assert_eq(pool.progress(), LunaRequestPreparationPoolAdvanced)
  assert_eq(pool.lanes[work.lane].phase, LUNA_PREPARATION_FRAMED_SCANNING)
  pool.begin_drain()
  let mut turns = 0
  while pool.free_lane_count() == 0 {
    assert_false(pool.progress() == LunaRequestPreparationPoolAwaitingInput)
    turns += 1
    if turns > 128 {
      fail("draining Luna scanner did not release its lane")
    }
  }
  try work.state() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("drained Luna scanner produced the wrong stale error")
  } noraise {
    _ => fail("drained Luna scanner Work remained live")
  }
}
