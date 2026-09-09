///|
test "CUDA operand layout is a pure injective aligned physical interpretation" {
  for width in [32, 64, 128] {
    let layout = cuda_projection_operand_layout(width)
    assert_eq(
      layout.row_stride(),
      cuda_projection_operand_layout(width).row_stride(),
    )
    assert_eq(layout.row_stride(), width + 16)
    assert_eq(layout.row_stride() % 16, 0)
    let unpadded = cuda_projection_operand_layout(width, padded=false)
    assert_eq(unpadded.row_stride(), width)
    assert_eq(unpadded.storage_elements(64), width * 64)
    let occupied = Array::make(layout.storage_elements(64), false)
    for row in 0..<64 {
      for column in 0..<width {
        let address = layout.offset(row, column)
        assert_false(occupied[address])
        occupied[address] = true
        assert_true(address < layout.storage_elements(64))
      }
      for vector in 0..<(width / 8) {
        assert_eq(layout.offset(row, vector * 8) % 8, 0)
      }
    }
    // Four consecutive rows rotate through four distinct 32-bit bank groups,
    // instead of width 64 aliasing every row at bank zero.
    let banks = [ for row in 0..<4 => layout.offset(row, 0) / 2 % 32 ]
    assert_eq(banks[0], 0)
    for row in 0..<4 {
      for previous in 0..<row {
        assert_true(banks[row] != banks[previous])
      }
    }
  }
}
