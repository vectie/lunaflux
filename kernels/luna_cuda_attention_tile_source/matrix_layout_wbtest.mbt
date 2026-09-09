///|
test "explicit matrix accumulator ownership is a bijection" {
  let seen = FixedArray::make(256, false)
  for lane in 0..<32 {
    for item in 0..<8 {
      let row = lane / 4 + item % 4 / 2 * 8
      let column = lane % 4 * 2 + item % 2 + item / 4 * 8
      let index = row * 16 + column
      assert_false(seen[index])
      seen[index] = true
    }
  }
  for value in seen {
    assert_true(value)
  }
}

///|
test "float shared permutation preserves capacity and per-instruction banks" {
  fn offset(row : Int, column : Int, columns : Int) -> Int {
    let r = row & 7
    let c = column & 15
    let bank = (r * 4) ^
      ((r & 1) * 8) ^
      (c >> 1) ^
      ((c & 1) * 16) ^
      (((column >> 4) & 1) * 8)
    ((row >> 3) * (columns / 16) + (column >> 4)) * 128 +
    ((c >> 3) + (c & 1) * 2) * 32 +
    bank
  }
  for columns in [16, 32, 64, 128, 256] {
    let seen = FixedArray::make(32 * columns, false)
    for row in 0..<32 {
      for column in 0..<columns {
        let index = offset(row, column, columns)
        assert_false(seen[index])
        seen[index] = true
      }
    }
    for value in seen {
      assert_true(value)
    }
    for item in 0..<8 {
      let banks = FixedArray::make(32, false)
      for lane in 0..<32 {
        let bank = offset(
            lane / 4 + item % 4 / 2 * 8,
            lane % 4 * 2 + item % 2 + item / 4 * 8,
            columns,
          ) %
          32
        assert_false(banks[bank])
        banks[bank] = true
      }
    }
  }
}
