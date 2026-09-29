///|
test "operand layouts are row bijections and preserve contiguous vector lanes" {
  for columns in [16, 32, 64, 128] {
    for rotations in [1, 2, 4, 8] {
      if rotations * 8 > columns {
        continue
      }
      let p = OperandLayout::new(columns~, vector=8, rotations~, row_group=2).unwrap()
      for row in 0..<32 {
        let seen = Array::make(columns, false)
        for column in 0..<columns {
          let offset = p.offset(row, column).unwrap()
          assert_eq(offset / columns, row)
          assert_false(seen[offset % columns])
          seen[offset % columns] = true
          assert_eq(offset % 8, column % 8)
        }
        assert_eq(p.offset(row, columns), None)
      }
      assert_eq(p.offset(-1, 0), None)
      assert_eq(p.offset(2147483647, 0), None)
    }
  }
}

///|
test "layout construction rejects invalid vector or rotation geometry" {
  assert_eq(
    OperandLayout::new(columns=64, vector=3, rotations=2, row_group=1),
    None,
  )
  assert_eq(
    OperandLayout::new(columns=64, vector=8, rotations=3, row_group=1),
    None,
  )
  assert_eq(
    OperandLayout::new(columns=64, vector=8, rotations=2, row_group=0),
    None,
  )
  assert_eq(
    OperandLayout::new(columns=24, vector=8, rotations=2, row_group=1),
    None,
  )
}
