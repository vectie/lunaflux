///|
test "CUDA operand layout is a pure injective aligned physical interpretation" {
  for width in [32, 64, 128] {
    let layout : CudaProjectionTransportLayout = { columns: width }
    let occupied = Array::make(width * 64, false)
    for row in 0..<64 {
      for column in 0..<width {
        let address = layout.offset(row, column)
        assert_false(occupied[address])
        occupied[address] = true
        assert_true(address < width * 64)
      }
      for vector in 0..<(width / 8) {
        assert_eq(layout.offset(row, vector * 8) % 8, 0)
      }
    }
    // Eight matrix-load row vectors cover disjoint groups of four banks.
    let banks = [ for row in 0..<8 => layout.offset(row, 0) / 2 % 32 ]
    assert_eq(banks[0], 0)
    for row in 0..<8 {
      for previous in 0..<row {
        assert_true(banks[row] != banks[previous])
      }
    }
  }
}
