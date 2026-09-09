///|
test "transport layout preserves producer vectors and consumer banks" {
  for width in [16, 32, 64, 128] {
    let layout : CudaProjectionTransportLayout = { columns: width }
    let rows = 192
    let seen = FixedArray::make(rows * width, false)
    for row in 0..<rows {
      for column in 0..<width {
        let index = layout.offset(row, column)
        assert_true(index >= 0 && index < seen.length())
        assert_false(seen[index])
        seen[index] = true
        if column % 8 == 0 {
          assert_eq(index % 8, 0)
          for item in 0..<8 {
            assert_eq(layout.offset(row, column + item), index + item)
          }
        }
      }
    }
    // Each eight-lane group copies 128 B. Check every word's bank is distinct.
    for first in 0..<(rows * width / 64) {
      let banks = FixedArray::make(32, false)
      for lane in 0..<8 {
        let logical = (first * 8 + lane) * 8
        let index = layout.offset(logical / width, logical % width)
        for word in 0..<4 {
          let bank = (index / 2 + word) % 32
          assert_false(banks[bank])
          banks[bank] = true
        }
      }
    }
    // Each scalar packed matrix-fragment read uses the same immutable map.
    for row_tile in 0..<(rows / 16) {
      for k_tile in 0..<(width / 16) {
        for part in 0..<4 {
          let banks = FixedArray::make(32, false)
          for lane in 0..<32 {
            let row = row_tile * 16 + lane / 4 + part % 2 * 8
            let col = k_tile * 16 + lane % 4 * 2 + part / 2 * 8
            let bank = layout.offset(row, col) / 2 % 32
            assert_false(banks[bank])
            banks[bank] = true
          }
        }
      }
    }
  }
}

///|
test "transport result permutation separates packed stores and coalesced reads" {
  fn result_offset(row : Int, column : Int) -> Int {
    let r = row % 8
    let bank = (r * 4) ^ (r % 2 * 8) ^ (column / 2) ^ (column % 2 * 16)
    row / 8 * 128 + (column / 8 + column % 2 * 2) * 32 + bank
  }
  let seen = FixedArray::make(256, false)
  for row in 0..<16 {
    for column in 0..<16 {
      let index = result_offset(row, column)
      assert_true(index >= 0 && index < 256)
      assert_false(seen[index])
      seen[index] = true
    }
  }
  for item in 0..<8 {
    let stores = FixedArray::make(32, false)
    let reads = FixedArray::make(32, false)
    for lane in 0..<32 {
      let store = result_offset(
          lane / 4 + item % 4 / 2 * 8,
          lane % 4 * 2 + item % 2 + item / 4 * 8,
        ) %
        32
      let element = lane + item * 32
      let read = result_offset(element / 16, element % 16) % 32
      assert_false(stores[store])
      assert_false(reads[read])
      stores[store] = true
      reads[read] = true
    }
  }
}
