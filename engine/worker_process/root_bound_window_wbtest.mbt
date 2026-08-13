///|
test "root-bound production supervisor serializes the device child window" {
  let owner = test_root_bound_supervisor()
  let first = owner.submit(supervisor_plan(1UL))
  assert_false(owner.has_capacity())
  try owner.submit(supervisor_plan(2UL)) catch {
    RootBoundFailure(Process(Backpressured)) => ()
    _ => fail("unexpected root-bound second-submit failure")
  } noraise {
    _ => fail("root-bound supervisor admitted a second in-flight plan")
  }
  assert_eq(first.sequence_value(), 1UL)
  owner.close()
}
