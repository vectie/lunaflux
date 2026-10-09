///|
test "compressed selection plans exact scratch and compressed IDs including zero capacity" {
  let plan = @precision_ir.CompressedIndexSelectionPrecision::new(
    head_count=2,
    head_width=128,
    compression_ratio=4,
    maximum_rows=3,
    maximum_positions=16,
    top_k=2,
    maximum_workspace_bytes=60L,
  )
  assert_eq(plan.geometry(), (2, 128, 4, 3, 16, 2))
  assert_eq(plan.cache_capacity(), 4)
  assert_eq(plan.selected_width(), 2)
  assert_eq(plan.frame_bytes()[:].to_owned(), [24L, 24L, 12L])
  assert_eq(plan.workspace_bytes(), 60L)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::StepWrite)
  try
    @precision_ir.CompressedIndexSelectionPrecision::new(
      head_count=2,
      head_width=128,
      compression_ratio=4,
      maximum_rows=3,
      maximum_positions=16,
      top_k=2,
      maximum_workspace_bytes=59L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("aggregate scratch budget ignored")
  }
  let empty = @precision_ir.CompressedIndexSelectionPrecision::new(
    head_count=2,
    head_width=128,
    compression_ratio=4,
    maximum_rows=3,
    maximum_positions=3,
    top_k=512,
    maximum_workspace_bytes=30L,
  )
  assert_eq(empty.cache_capacity(), 0)
  assert_eq(empty.selected_width(), 0)
  assert_eq(empty.frame_bytes()[:].to_owned(), [6L, 12L, 12L])
}

///|
test "compressed selection rejects impossible context and ABI scratch sizes" {
  try
    @precision_ir.CompressedIndexSelectionPrecision::new(
      head_count=2,
      head_width=128,
      compression_ratio=4,
      maximum_rows=3,
      maximum_positions=2,
      top_k=512,
      maximum_workspace_bytes=1048576L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    error => raise error
  } noraise {
    _ => fail("context smaller than chunk accepted")
  }
  try
    @precision_ir.CompressedIndexSelectionPrecision::new(
      head_count=512,
      head_width=128,
      compression_ratio=2,
      maximum_rows=65536,
      maximum_positions=1048576,
      top_k=512,
      maximum_workspace_bytes=100000000000L,
    )
  catch {
    @precision_ir.PrecisionPlanError::ArithmeticOverflow => ()
    error => raise error
  } noraise {
    _ => fail("overlarge scratch accepted")
  }
}
