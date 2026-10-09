///|
test "MoE combination keeps global F32 accumulation until declared round" {
  let plan = @precision_ir.MoeCombinePrecision::new(
    maximum_rows=3,
    hidden=17,
    rounding=SumThenBf16,
    memory_ceiling=510L,
  )
  assert_eq(plan.geometry(), (3, 17))
  assert_eq(plan.buffer_bytes(), (204L, 102L))
  assert_eq(plan.rounding(), SumThenBf16)
  try
    @precision_ir.MoeCombinePrecision::new(
      maximum_rows=3,
      hidden=17,
      rounding=Bf16RoutedThenSum,
      memory_ceiling=509L,
    )
  catch {
    MemoryBudgetExceeded => ()
    _ => fail("incorrect memory error")
  } noraise {
    _ => fail("combination exceeded memory budget")
  }
}
