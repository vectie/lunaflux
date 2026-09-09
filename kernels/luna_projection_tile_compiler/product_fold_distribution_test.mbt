///|
test "product fold distribution exposes row parallelism without changing fold order" {
  for rows in [1, 2, 3, 4, 6, 8] {
    for columns in [1, 2, 4, 8] {
      for available in [1, 2, 4, 8, 16, 32] {
        let distribution = @luna_projection_tile_compiler.partition_projection_product_fold(
          row_tiles=rows,
          output_tiles=columns,
          consumer_groups=available,
        )
        if available < columns {
          assert_eq(distribution, None)
          continue
        }
        let plan = distribution.unwrap()
        assert_true(plan.consumer_groups() <= available)
        assert_eq(plan.row_groups() * plan.rows_per_group(), rows)
        let visits = Array::make(rows * columns, 0)
        let actual = Array::make(rows * columns, 0)
        let expected = Array::makei(rows * columns, index => {
          for k in 0..<7; state = index / columns + index % columns {
            continue state * 3 + k
          } nobreak {
            state
          }
        })
        for consumer in 0..<plan.consumer_groups() {
          let row_start = consumer / columns * plan.rows_per_group()
          let column = consumer % columns
          for local_row in 0..<plan.rows_per_group() {
            let row = row_start + local_row
            visits[row * columns + column] += 1
            // Deliberately order-sensitive recurrence: a regrouped reduction
            // is not interchangeable with distributing independent rows.
            actual[row * columns + column] = for
              k in 0..<7
              state = row + column {
              continue state * 3 + k
            } nobreak {
              state
            }
          }
        }
        for count in visits {
          assert_eq(count, 1)
        }
        assert_eq(actual, expected)
      }
    }
  }
}

///|
test "sixteen consumers split four rows of four sibling output folds" {
  let plan = @luna_projection_tile_compiler.partition_projection_product_fold(
    row_tiles=4,
    output_tiles=4,
    consumer_groups=16,
  ).unwrap()
  assert_eq(plan.row_groups(), 4)
  assert_eq(plan.rows_per_group(), 1)
  assert_eq(plan.consumer_groups(), 16)
  assert_eq(
    @luna_projection_tile_compiler.partition_projection_product_fold(
      row_tiles=0,
      output_tiles=4,
      consumer_groups=16,
    ),
    None,
  )
  assert_eq(
    @luna_projection_tile_compiler.partition_projection_product_fold(
      row_tiles=4,
      output_tiles=0,
      consumer_groups=16,
    ),
    None,
  )
}
