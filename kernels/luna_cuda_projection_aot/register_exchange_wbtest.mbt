///|
test "packed register exchange preserves every accumulator coordinate" {
  let seen = FixedArray::make(256, 0)
  for lane in 0..<32 {
    for store in 0..<4 {
      let source_lane = (lane / 8 + store % 2 * 4) * 4 + lane % 4
      let first = store / 2 * 2 + (if (lane & 4) != 0 { 4 } else { 0 })
      for component in 0..<2 {
        let item = first + component
        let original_row = source_lane / 4 + item % 4 / 2 * 8
        let original_column = source_lane % 4 * 2 + item % 2 + item / 4 * 8
        let row = store * 4 + lane / 8
        let column = lane % 8 * 2 + component
        assert_eq((row, column), (original_row, original_column))
        seen[row * 16 + column] += 1
      }
    }
  }
  assert_true(seen.iter().all(count => count == 1))
  let source = projection_register_exchange_source(1024)
  assert_false(source.contains("__syncwarp"))
  assert_false(source.contains("warp_output"))
  assert_true(source.contains("__floats2bfloat162_rn"))
}
