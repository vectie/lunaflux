///|
test "allocation validation failure closes the acquired resource exactly once" {
  let opens = Ref(0)
  let closes = Ref(0)
  try
    allocate_owned(
      224L,
      bytes => {
        opens.val += 1
        bytes
      },
      resource => resource - 1L,
      _resource => closes.val += 1,
    )
  catch {
    AllocationSizeMismatch => ()
    _ => fail("unexpected allocation validation error")
  } noraise {
    _ => fail("expected allocation size mismatch")
  }
  assert_eq(opens.val, 1)
  assert_eq(closes.val, 1)
}

///|
test "allocation cleanup failure retains its bounded lifecycle stage" {
  try
    allocate_owned(224L, bytes => bytes, resource => resource - 1L, _resource => {
      raise Busy(AllocationClose)
    })
  catch {
    CleanupFailed(AllocationValidation, Busy(AllocationClose)) => ()
    _ => fail("unexpected cleanup failure")
  } noraise {
    _ => fail("expected cleanup failure")
  }
}

///|
test "allocation-open failure owns no resource to clean" {
  let closes = Ref(0)
  try
    allocate_owned(
      224L,
      _bytes => raise DriverFailure(AllocationCreate),
      resource => resource,
      _resource => closes.val += 1,
    )
  catch {
    DeviceFailed(DriverFailure(AllocationCreate)) => ()
    _ => fail("unexpected open failure")
  } noraise {
    _ => fail("expected allocation-open failure")
  }
  assert_eq(closes.val, 0)
}
