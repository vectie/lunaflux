///|
test "row selection rejects malformed interval schedules" {
  for
    segments in [
      [{ operand: 0, row_begin: 1, row_end: 2, }],
      [{ operand: 1, row_begin: 0, row_end: 2, }],
      [{ operand: 0, row_begin: 0, row_end: 0, }],
      [
        { operand: 0, row_begin: 0, row_end: 2, },
        { operand: 1, row_begin: 3, row_end: 4, },
      ],
      [
        { operand: 0, row_begin: 0, row_end: 2, },
        { operand: 1, row_begin: 1, row_end: 4, },
      ],
    ] {
    assert_true(plan_projection_row_selection(segments) is None)
  }
}
