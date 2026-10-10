///|
test "seed/noise block bounds and metadata workspace are immutable" {
  let plan = @precision_ir.SeededTokenBlockPrecision::new(
    vocabulary_size=17,
    block_rows=5,
    maximum_main_rows=8,
    noise_token_id=16,
  )
  assert_eq(plan.geometry(), (17, 5, 8, 16))
  assert_eq(plan.workspace_bytes(), 60L)
  assert_eq(plan.maximum_seed_tokens(), 1)
  let vector = @precision_ir.SeededTokenBlockPrecision::new(
    vocabulary_size=17,
    block_rows=5,
    maximum_main_rows=8,
    noise_token_id=16,
    maximum_seed_tokens=8,
  )
  assert_eq(vector.maximum_seed_tokens(), 8)
  assert_eq(vector.workspace_bytes(), 60L)
  for capacity in [0, 9] {
    try
      @precision_ir.SeededTokenBlockPrecision::new(
        vocabulary_size=17,
        block_rows=5,
        maximum_main_rows=8,
        noise_token_id=16,
        maximum_seed_tokens=capacity,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid seed-vector capacity")
    }
  }
  for
    (rows, main_rows, noise) in [(0, 8, 16), (5, 0, 16), (5, 8, 17), (5, 8, -1)] {
    try
      @precision_ir.SeededTokenBlockPrecision::new(
        vocabulary_size=17,
        block_rows=rows,
        maximum_main_rows=main_rows,
        noise_token_id=noise,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid seed geometry accepted")
    }
  }
}
