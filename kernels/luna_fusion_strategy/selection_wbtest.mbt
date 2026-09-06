///|
fn measured(
  schedule : FusionSchedule,
  time : Int64,
  local_storage? : Int64 = 0L,
) -> FusionAlternative raise {
  FusionAlternative::new(
    schedule~,
    cost=FusionSpanCost::measured(median_nanoseconds=time, samples=3),
    workspace_bytes=0L,
    local_bytes=local_storage,
  )
}

///|
test "complete cost selects each schedule independently of enumeration order" {
  for winner in [Unfused, ProducerSeparated, FullyFused] {
    let alternatives = [Unfused, ProducerSeparated, FullyFused].map(schedule => {
      measured(schedule, if schedule == winner { 10L } else { 20L })
    })
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
      assert_eq(
        select_fusion_schedule(
          order.map(index => alternatives[index]),
          maximum_workspace_bytes=0L,
          maximum_local_bytes=0L,
        ),
        winner,
      )
    }
  }
  assert_eq(
    select_fusion_schedule(
      [measured(FullyFused, 10L), measured(Unfused, 10L)],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    ),
    Unfused,
  )
}

///|
test "resource limits filter before cost ranking" {
  let full = measured(FullyFused, 1L, local_storage=65536L)
  let partial = measured(ProducerSeparated, 10L)
  assert_eq(
    select_fusion_schedule(
      [full, partial],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=32768L,
    ),
    ProducerSeparated,
  )
  try
    select_fusion_schedule(
      [full],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=32768L,
    )
  catch {
    error => assert_eq(error, NoFeasibleSchedule)
  } noraise {
    _ => fail("resource limit ignored")
  }
}

///|
test "comparison rejects duplicate mixed and overflowing cost inputs" {
  let full = measured(FullyFused, 1L)
  try
    select_fusion_schedule(
      [full, full],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
  catch {
    error => assert_eq(error, InvalidComparison)
  } noraise {
    _ => fail("duplicate accepted")
  }
  try FusionSpanCost::measured(median_nanoseconds=1L, samples=2) catch {
    error => assert_eq(error, InvalidCost)
  } noraise {
    _ => fail("insufficient samples")
  }
  try
    FusionSpanCost::estimated(
      compute_nanoseconds=9223372036854775807L,
      traffic_nanoseconds=1L,
      synchronization_nanoseconds=0L,
      launch_nanoseconds=0L,
    )
  catch {
    error => assert_eq(error, InvalidCost)
  } noraise {
    _ => fail("overflow accepted")
  }
  let estimate = FusionAlternative::new(
    schedule=Unfused,
    cost=FusionSpanCost::estimated(
      compute_nanoseconds=1L,
      traffic_nanoseconds=1L,
      synchronization_nanoseconds=1L,
      launch_nanoseconds=1L,
    ),
    workspace_bytes=0L,
    local_bytes=0L,
  )
  try
    select_fusion_schedule(
      [full, estimate],
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    )
  catch {
    error => assert_eq(error, InvalidComparison)
  } noraise {
    _ => fail("mixed bases accepted")
  }
}
