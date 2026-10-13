///|
test "chunk plan accounts for all immutable factors and one cache commit" {
  let p = @precision.RecurrentDeltaPrecision::new(
    head_count=64,
    key_dimension=128,
    value_dimension=128,
    maximum_rows=16,
    maximum_sequences=1,
    cache_slots=1,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=4194304L,
    evaluation=ChunkTriangularFloat32,
  )
  let plan = @recurrent_physical_ir.ChunkDeltaPlan::new(
    p,
    value_tile=32,
    maximum_workspace_bytes=4390912L,
  )
  assert_eq(plan.workspace_bytes(), 4390912L)
  assert_eq(plan.head_floats(), 17152L)
  assert_eq(plan.stage_count(), 3)
  assert_true(!plan.local_factors())
  assert_eq(plan.local_storage_bytes(), 0)
  let tiled = @recurrent_physical_ir.ChunkDeltaPlan::new(
    p,
    value_tile=32,
    maximum_workspace_bytes=4390912L,
    maximum_local_storage_bytes=49152,
  )
  assert_true(tiled.local_factors())
  assert_eq(tiled.local_storage_bytes(), 19712)
  assert_eq(plan.state_effect(0), @precision.NumericEffect::StepWrite)
  assert_eq(plan.state_effect(1), @precision.NumericEffect::StepWrite)
  assert_eq(plan.state_effect(2), @precision.NumericEffect::CacheCommit)
  try
    @recurrent_physical_ir.ChunkDeltaPlan::new(
      p,
      value_tile=32,
      maximum_workspace_bytes=4390911L,
    )
  catch {
    @precision.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("scratch ceiling ignored")
  }
}
