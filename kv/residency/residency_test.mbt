///|
fn pool(slots? : Int = 2, transfers? : Int = 1) -> @residency.Pool {
  @residency.Pool::new(
    try! @residency.Limits::new(
      slots~,
      transfers~,
      bytes_per_slot=256L,
      host_budget=512L,
      separate_host_memory=true,
    ),
  )
}

///|
test "spill restore and duplicate completion retain exact ownership" {
  let owner = pool()
  guard owner.begin_spill() is Started(spill) else { fail("spill") }
  assert_true(owner.begin_spill() is Backpressured)
  assert_true(owner.evict(spill.entry()) is Rejected)
  assert_true(owner.complete(spill) is HostReady)
  assert_true(owner.complete(spill) is Rejected)
  guard owner.begin_restore(spill.entry()) is Started(restore) else {
    fail("restore")
  }
  assert_true(owner.complete(spill) is Rejected)
  assert_true(!owner.is_host_ready(spill.entry()))
  assert_true(owner.complete(restore) is DeviceReady)
  assert_true(owner.is_host_ready(spill.entry()))
  assert_true(owner.evict(spill.entry()) is Released)
  assert_eq(owner.available(), 2)
  assert_eq(owner.inflight(), 0)
}

///|
test "cancelled spill holds reservations until completion and fences reuse" {
  let owner = pool(slots=1)
  guard owner.begin_spill() is Started(old) else { fail("spill") }
  assert_true(owner.cancel(old) is RetainUntilCompletion)
  assert_true(!owner.close())
  assert_true(owner.begin_spill() is Backpressured)
  assert_true(owner.complete(old) is Released)
  guard owner.begin_spill() is Started(fresh) else { fail("reuse") }
  assert_true(owner.complete(old) is Rejected)
  assert_true(owner.cancel(old) is Rejected)
  assert_true(owner.complete(fresh) is HostReady)
  assert_true(owner.begin_restore(old.entry()) is InvalidEntry)
}

///|
test "cancelled restore preserves immutable host copy and suppresses publication" {
  let owner = pool()
  guard owner.begin_spill() is Started(spill) else { fail("spill") }
  assert_true(owner.complete(spill) is HostReady)
  guard owner.begin_restore(spill.entry()) is Started(restore) else {
    fail("restore")
  }
  assert_true(owner.cancel(restore) is RetainUntilCompletion)
  assert_true(owner.complete(restore) is RestoreCancelled)
  assert_true(owner.is_host_ready(spill.entry()))
}

///|
test "failed effects quarantine both reservations until drain" {
  let owner = pool()
  guard owner.begin_spill() is Started(spill) else { fail("spill") }
  assert_true(owner.fail(spill) is DrainRequired)
  assert_true(owner.complete(spill) is Rejected)
  assert_true(owner.evict(spill.entry()) is Rejected)
  assert_true(!owner.close())
  assert_eq(owner.inflight(), 1)
  assert_true(owner.drained(spill) is Released)
  assert_true(owner.drained(spill) is Rejected)
  assert_eq(owner.inflight(), 0)
  assert_true(owner.close())
  assert_true(owner.begin_spill() is Closed)
}

///|
test "foreign owners cannot complete equal-numbered slots" {
  let a = pool()
  let b = pool()
  guard a.begin_spill() is Started(first) else { fail("a") }
  guard b.begin_spill() is Started(second) else { fail("b") }
  assert_true(a.complete(second) is Rejected)
  assert_true(b.cancel(first) is Rejected)
  assert_true(a.complete(first) is HostReady)
  assert_true(b.complete(second) is HostReady)
}

///|
test "startup budgets reject shared memory and multiplication overflow" {
  for separate in [true, false] {
    try
      @residency.Limits::new(
        slots=2147483647,
        transfers=1,
        bytes_per_slot=9223372036854775807L,
        host_budget=9223372036854775807L,
        separate_host_memory=separate,
      )
    catch {
      InvalidLimits => ()
    } noraise {
      _ => fail("overflow")
    }
  }
  try
    @residency.Limits::new(
      slots=1,
      transfers=1,
      bytes_per_slot=256L,
      host_budget=512L,
      separate_host_memory=false,
    )
  catch {
    InvalidLimits => ()
  } noraise {
    _ => fail("shared physical memory")
  }
}
