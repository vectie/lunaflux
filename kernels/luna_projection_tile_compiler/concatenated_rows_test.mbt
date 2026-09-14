///|
test "concatenated row views preserve operands offsets and total coverage" {
  for counts in [[1], [8, 2, 2], [17, 33, 65], [3, 5, 7, 11]] {
    let segments = @luna_projection_tile_compiler.plan_projection_concatenated_rows(
      counts,
    ).unwrap()
    assert_eq(segments.length(), counts.length())
    for operand, segment in segments {
      assert_eq(segment.operand, operand)
      assert_eq(segment.row_end - segment.row_begin, counts[operand])
      assert_eq(
        segment.row_begin,
        counts[:operand].fold(init=0, (a, b) => a + b),
      )
      for row in segment.row_begin..<segment.row_end {
        assert_true(
          row - segment.row_begin >= 0 &&
          row - segment.row_begin < counts[operand],
        )
      }
    }
    assert_eq(
      segments[segments.length() - 1].row_end,
      counts.fold(init=0, (a, b) => a + b),
    )
  }
  for counts in [[], [0], [-1], [1, 0], [2147483647, 1]] {
    assert_eq(
      @luna_projection_tile_compiler.plan_projection_concatenated_rows(counts),
      None,
    )
  }
  let edge = @luna_projection_tile_compiler.plan_projection_concatenated_rows([
    2147483646, 1,
  ]).unwrap()
  assert_eq(edge[1].row_begin, 2147483646)
  assert_eq(edge[1].row_end, 2147483647)
}
