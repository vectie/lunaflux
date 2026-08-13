///|
test "exclusive admission blocks ordinary scheduler mutation" {
  let owner = wb_owner_with_capacity(2, 2)
  let existing = owner.admit(wb_request(801UL), 0UL)
  let prepared = owner.prepare_exclusive_admission(wb_request(802UL))
  let expected = Invalid(field=CancellationReservation, issue=InvalidTransition)
  try owner.admit(wb_request(803UL), 0UL) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("exclusive reservation allowed ordinary admission")
  }
  try owner.build_next() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("exclusive reservation allowed plan construction")
  }
  try owner.cancel(existing) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("exclusive reservation allowed cancellation")
  }
  try owner.expire_deadlines(200UL) catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("exclusive reservation allowed deadline mutation")
  }
  try owner.drain_instance_loss() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("exclusive reservation allowed instance-loss drain")
  }
  owner.abort_exclusive_admission(prepared)
  assert_eq(owner.live_request_count(), 1)
  owner.cancel(existing)
}

///|
test "exclusive admission expiry consumes the shell and restores pristine state" {
  let owner = wb_owner()
  let prepared = owner.prepare_exclusive_admission(
    wb_request(804UL, deadline=1UL),
  )
  assert_false(owner.is_pristine())
  assert_eq(
    owner.commit_exclusive_admission(prepared, 2UL),
    ExclusiveAdmissionExpired,
  )
  assert_true(owner.is_pristine())
  assert_eq(owner.live_request_count(), 0)
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
}

///|
test "exclusive admission publishes the exact preallocated handle once" {
  let owner = wb_owner()
  let prepared = owner.prepare_exclusive_admission(wb_request(805UL))
  assert_true(owner.has_exclusive_admission(prepared))
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionCommitted,
  )
  let handle = prepared.request_handle()
  assert_eq(handle.request_id(), try! @inference.RequestId::new(805UL))
  assert_eq(owner.status(handle.request_id()).unwrap().handle(), handle)
  assert_false(owner.has_exclusive_admission(prepared))
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
}

///|
test "aborted exclusive admission cannot be recommitted through an alias" {
  let owner = wb_owner()
  let prepared = owner.prepare_exclusive_admission(wb_request(806UL))
  owner.abort_exclusive_admission(prepared)
  owner.abort_exclusive_admission(prepared)
  assert_eq(
    owner.commit_exclusive_admission(prepared, 0UL),
    ExclusiveAdmissionInvalidated,
  )
  assert_true(owner.is_pristine())
  assert_eq(owner.live_request_count(), 0)
}
