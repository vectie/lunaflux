///|
test "closing failure blocks execution and retry reaches closed" {
  let lifecycle : Ref[ExecutorLifecycle] = Ref(Ready)
  let output_valid = Ref(true)
  let fail_once = Ref(true)
  let cleanup_calls = Ref(0)
  try
    close_with_lifecycle(
      lifecycle.val,
      state => lifecycle.val = state,
      () => output_valid.val = false,
      () => {
        cleanup_calls.val += 1
        if fail_once.val {
          fail_once.val = false
          raise CleanupFailed(Stream, Device(DriverFailure(StreamClose)))
        }
      },
    )
  catch {
    CleanupFailed(Stream, Device(DriverFailure(StreamClose))) => ()
    error => fail("unexpected cleanup failure: \{Repr(error)}")
  } noraise {
    _ => fail("cleanup failure was swallowed")
  }
  assert_true(lifecycle.val == ClosingFailed)
  assert_false(output_valid.val)
  for _ in 0..<2 {
    try require_ready(lifecycle.val) catch {
      Closed => ()
      error => fail("unexpected lifecycle failure: \{Repr(error)}")
    } noraise {
      _ => fail("closing-failed lifecycle admitted execution")
    }
  }
  close_with_lifecycle(
    lifecycle.val,
    state => lifecycle.val = state,
    () => output_valid.val = false,
    () => cleanup_calls.val += 1,
  )
  assert_true(lifecycle.val == Closed)
  assert_eq(cleanup_calls.val, 2)
  close_with_lifecycle(
    lifecycle.val,
    state => lifecycle.val = state,
    () => output_valid.val = false,
    () => cleanup_calls.val += 1,
  )
  assert_eq(cleanup_calls.val, 2)
}

///|
priv struct FakePreparationOwner {
  id : Int
  mut open : Bool
}

///|
priv struct FakeCleanupGuard {
  owner : FakePreparationOwner
  cause : DeviceExecutorError
  cleanup_failure : DeviceExecutorError
  mut lifecycle : ExecutorLifecycle
}

///|
fn FakeCleanupGuard::retry_close(
  self : FakeCleanupGuard,
) -> Unit raise DeviceExecutorError {
  close_with_lifecycle(
    self.lifecycle,
    state => self.lifecycle = state,
    () => (),
    () => self.owner.open = false,
  )
}

///|
test "construction and cleanup failure retains explicit retry authority" {
  let owner : FakePreparationOwner = { id: 7, open: true }
  let cause = DeviceFailed(Module, DriverFailure(ModuleLoad))
  let cleanup_failure = CleanupFailed(Stream, Device(Busy(StreamClose)))
  let retained = retain_on_cleanup_failure(
    owner,
    cause,
    _ => raise cleanup_failure,
    (resource, original, cleanup) => {
      owner: resource,
      cause: original,
      cleanup_failure: cleanup,
      lifecycle: ClosingFailed,
    },
  )
  assert_true(retained.owner.open)
  assert_eq(retained.owner.id, 7)
  assert_eq(retained.cause, cause)
  assert_eq(retained.cleanup_failure, cleanup_failure)
  retained.retry_close()
  assert_false(retained.owner.open)
  assert_true(retained.lifecycle == Closed)
  retained.retry_close()
}

///|
priv struct FakeOptionalResource {
  id : Int
  mut open : Bool
}

///|
test "absent arena does not skip token lease and stream cleanup" {
  let first : Ref[DeviceExecutorError?] = Ref(None)
  let order : Array[Int] = []
  let token : FakeOptionalResource = { id: 2, open: true }
  let lease : FakeOptionalResource = { id: 3, open: true }
  let stream : FakeOptionalResource = { id: 1, open: true }
  let absent : FakeOptionalResource? = None
  close_optional_resource(
    absent,
    resource => resource.open,
    resource => order.push(resource.id),
    resource => resource.open = false,
    first,
  )
  close_optional_resource(
    Some(token),
    resource => resource.open,
    resource => order.push(resource.id),
    resource => resource.open = false,
    first,
  )
  close_optional_resource(
    Some(lease),
    resource => resource.open,
    resource => order.push(resource.id),
    resource => resource.open = false,
    first,
  )
  close_optional_resource(
    Some(stream),
    resource => resource.open,
    resource => order.push(resource.id),
    resource => resource.open = false,
    first,
  )
  assert_eq(order, [2, 3, 1])
  assert_false(token.open)
  assert_false(lease.open)
  assert_false(stream.open)
  assert_true(first.val is None)
}

///|
priv struct FakeChild {
  parent : Int
  mut open : Bool
}

///|
priv struct FakeParent {
  id : Int
  mut open : Bool
}

///|
test "failed function blocks only its own module" {
  let first : Ref[DeviceExecutorError?] = Ref(None)
  let children = [{ parent: 0, open: true }, { parent: 1, open: false }]
  let parents : Array[FakeParent] = [
    { id: 0, open: true },
    { id: 1, open: true },
  ]
  let order : Array[Int] = []
  close_unblocked_parents_reverse(
    children,
    parents,
    child => child.open,
    child => child.parent,
    parent => parent.open,
    parent => order.push(parent.id),
    parent => parent.open = false,
    first,
  )
  assert_eq(order, [1])
  assert_true(parents[0].open)
  assert_false(parents[1].open)
  children[0].open = false
  close_unblocked_parents_reverse(
    children,
    parents,
    child => child.open,
    child => child.parent,
    parent => parent.open,
    parent => order.push(parent.id),
    parent => parent.open = false,
    first,
  )
  assert_eq(order, [1, 0])
  assert_false(parents[0].open)
}
