///|
test "serial scratch budgets peak not sum and preserves empty phases" {
  let sizes = [0L, 64L, 32L, 0L, 128L, 16L]
  let plan = @parallel_ir.SerialScratch::new(sizes[:])
  assert_eq(plan.bytes(), 128L)
  assert_eq(plan.saved_bytes(), 112L)
  sizes[4] = 999L
  assert_eq(plan.bytes(), 128L)
  assert_eq(@parallel_ir.SerialScratch::new([]).bytes(), 0L)
  assert_eq(@parallel_ir.SerialScratch::new([64L]).saved_bytes(), 0L)
}

///|
test "serial scratch rejects invalid sizes and overflow" {
  for sizes in [[-1L], [9223372036854775807L, 1L]] {
    try @parallel_ir.SerialScratch::new(sizes[:]) catch {
      @parallel_ir.ParallelError::Capacity => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid storage accepted")
    }
  }
}
