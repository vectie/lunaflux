///|
test "sibling partition reuses weights without splitting the ordered reduction" {
  let plan = @luna_projection_tile_compiler.partition_projection_sibling_reuse(
    row_tiles=4,
    output_tiles=4,
    consumer_groups=16,
  ).unwrap()
  assert_eq(plan.row_groups(), 2)
  assert_eq(plan.output_groups(), 4)
  assert_eq(plan.rows_per_group(), 2)
  assert_eq(plan.consumer_groups(), 8)
  // Independent (sibling,row,column) ownership is a bijection.
  let counts = Array::make(32, 0)
  for sibling in 0..<2 {
    for consumer in 0..<plan.consumer_groups() {
      for row in 0..<plan.rows_per_group() {
        let r = consumer / plan.output_groups() * plan.rows_per_group() + row
        let c = consumer % plan.output_groups()
        let index = (sibling * 4 + r) * 4 + c
        counts[index] += 1
      }
    }
  }
  for count in counts {
    assert_eq(count, 1)
  }
}

///|
test "sibling reuse rejects empty insufficient and non-reusing products" {
  for
    shape in [
      (0, 4, 16),
      (4, 0, 16),
      (4, 4, 0),
      (4, 4, 15),
      (4, 4, 8),
      (1, 4, 16),
      (4, 4, 32),
    ] {
    let (rows, outputs, consumers) = shape
    assert_eq(
      @luna_projection_tile_compiler.partition_projection_sibling_reuse(
        row_tiles=rows,
        output_tiles=outputs,
        consumer_groups=consumers,
      ),
      None,
    )
  }
}

///|
test "all admitted sibling distributions fit lifetime storage and preserve tail ownership" {
  // Matrix strategy admits power-of-two consumer counts through sixteen.
  // Its sibling pipeline has four row tiles and max(1, consumers/4) columns.
  for consumers in [1, 2, 4, 8, 16] {
    let columns = Int::max(1, consumers / 4)
    let candidate = @luna_projection_tile_compiler.partition_projection_sibling_reuse(
      row_tiles=4,
      output_tiles=columns,
      consumer_groups=consumers,
    )
    if consumers < 4 {
      assert_eq(candidate, None)
      continue
    }
    let plan = candidate.unwrap()
    assert_eq(plan.rows_per_group(), 2)
    assert_eq(plan.consumer_groups() * 2, consumers)
    // Dynamic scratch already holds one F32 16x16 tile per consumer.
    let result_bytes = 4 * columns * 256 * 4
    assert_true(result_bytes <= consumers * 256 * 4)
    // Two BF16 transfer slots, each (input rows + two sibling columns)x32.
    let expired_stage_bytes = 2 * (64 + 2 * columns * 16) * 32 * 2
    assert_true(result_bytes <= expired_stage_bytes)
    for live in 1..<=64 {
      let writes = Array::make(live * columns * 16 * 2, 0)
      for sibling in 0..<2 {
        for consumer in 0..<plan.consumer_groups() {
          let first_row = consumer / columns * plan.rows_per_group()
          let column = consumer % columns
          for row in 0..<plan.rows_per_group() {
            for cell in 0..<256 {
              let logical_row = (first_row + row) * 16 + cell / 16
              if logical_row < live {
                let index = (sibling * live + logical_row) * columns * 16 +
                  column * 16 +
                  cell % 16
                writes[index] += 1
              }
            }
          }
        }
      }
      for count in writes {
        assert_eq(count, 1)
      }
    }
  }
}
