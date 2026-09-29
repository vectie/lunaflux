///|
test "column window ownership never silently distributes epilogue rows" {
  for rows in [1, 2, 4, 8] {
    for outputs in [1, 2, 4, 8] {
      let distribution = partition_projection_column_fold(rows, outputs, 4).unwrap()
      assert_eq(distribution.row_groups(), 1)
      assert_eq(distribution.rows_per_group(), rows)
      assert_eq(distribution.output_groups(), Int::min(outputs, 4))
      let ordinary = partition_projection_product_fold(
        row_tiles=rows,
        output_tiles=Int::min(outputs, 4),
        consumer_groups=4,
      ).unwrap()
      if outputs == 1 && rows > 1 {
        assert_true(distribution != ordinary)
      }
    }
  }
  assert_eq(partition_projection_column_fold(0, 1, 4), None)
}
