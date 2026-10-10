///|
test "snapshot placement is exact and excludes padding assumptions" {
  let plan = @state_snapshot_plan.plan_state_snapshot(
    [32L, 8L, 20L],
    maximum_backup_bytes=60L,
  )
  assert_eq(plan.backup_bytes(), 60L)
  assert_eq(plan.offsets(), ReadOnlyArray::from_array([0L, 32L, 40L]))
  assert_eq(plan.sizes(), ReadOnlyArray::from_array([32L, 8L, 20L]))
}

///|
test "snapshot budget rejects overflow and one byte short before allocation" {
  for
    (sizes, limit) in [
      ([32L, 8L, 20L], 59L),
      ([9223372036854775807L, 1L], 9223372036854775807L),
    ] {
    try
      @state_snapshot_plan.plan_state_snapshot(
        sizes,
        maximum_backup_bytes=limit,
      )
    catch {
      @state_snapshot_plan.SnapshotBudgetExceeded => ()
      _ => fail("wrong snapshot error")
    } noraise {
      _ => fail("expected snapshot budget failure")
    }
  }
  for sizes in [[], [0L], [-1L]] {
    try
      @state_snapshot_plan.plan_state_snapshot(sizes, maximum_backup_bytes=60L)
    catch {
      @state_snapshot_plan.InvalidStateSize => ()
      _ => fail("wrong snapshot error")
    } noraise {
      _ => fail("expected state size failure")
    }
  }
}
