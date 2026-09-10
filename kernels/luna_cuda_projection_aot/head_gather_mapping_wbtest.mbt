///|
test "selected-row register gather covers operand and result ownership at row tails" {
  // Integer identities check the exact lane/part maps independently of CUDA
  // arithmetic. Each operand pair and result cell must have one owner, and
  // masked rows may never load or store an unselected token position.
  for rows in [0, 1, 2, 7, 8, 15, 16, 17, 31, 32] {
    let token_count = rows * 3 + 5
    let offsets = FixedArray::make(rows + 1, 0)
    for row in 0..<rows {
      offsets[row + 1] = row * 3 + 2
    }
    let stores = FixedArray::make(token_count * 16, 0)
    for tile in 0..<((rows + 15) / 16) {
      let row_begin = tile * 16
      let operands = FixedArray::make(16 * 16, 0)
      for lane in 0..<32 {
        for part in 0..<4 {
          let local_row = lane / 4 + part % 2 * 8
          let local_k = lane % 4 * 2 + part / 2 * 8
          let logical_row = row_begin + local_row
          let token_row = if logical_row < rows {
            offsets[logical_row + 1] - 1
          } else {
            -1
          }
          for component in 0..<2 {
            operands[local_row * 16 + local_k + component] += 1
            if token_row >= 0 && token_row < token_count {
              assert_eq(token_row, logical_row * 3 + 1)
            } else {
              assert_true(logical_row >= rows)
            }
          }
        }
        for pair in 0..<4 {
          let logical_row = row_begin + lane / 4 + pair % 2 * 8
          let token_row = if logical_row < rows {
            offsets[logical_row + 1] - 1
          } else {
            -1
          }
          let output_column = lane % 4 * 2 + pair / 2 * 8
          if token_row >= 0 && token_row < token_count {
            stores[token_row * 16 + output_column] += 1
            stores[token_row * 16 + output_column + 1] += 1
          }
        }
      }
      for count in operands {
        assert_eq(count, 1)
      }
    }
    for token_row in 0..<token_count {
      for column in 0..<16 {
        assert_eq(
          stores[token_row * 16 + column],
          if token_row % 3 == 1 && token_row / 3 < rows {
            1
          } else {
            0
          },
        )
      }
    }
  }
}
