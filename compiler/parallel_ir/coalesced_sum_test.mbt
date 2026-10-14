///|
test "collective concatenation preserves independent lane extents" {
  let p = @parallel_ir.CoalescedSumPlan::new(
    maximum_rows=64,
    left_width=32768,
    right_width=4096,
  )
  assert_eq(p.geometry(), (64, 32768, 4096))
  assert_eq(p.width(), 36864)
  assert_eq(p.storage_bytes(), 9437184L)
  try
    @parallel_ir.CoalescedSumPlan::new(
      maximum_rows=1,
      left_width=2147483647,
      right_width=1,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("overflow accepted")
  }
}
