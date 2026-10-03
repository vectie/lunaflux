///|
test "row factoring reconstructs every producer and consumer address" {
  for columns in [32, 64, 128, 256] {
    for rotations in [2, 4] {
      let layout = @physical_tile_ir.OperandLayout::new(
        columns~,
        vector=8,
        rotations~,
        row_group=2,
      ).unwrap()
      let factored = layout.factor_rows()
      assert_eq(factored.layout(), layout)
      for row in 0..<128 {
        let (origin, mask) = factored.row(row).unwrap()
        for column in 0..<columns {
          assert_eq(
            origin + factored.column(mask, column).unwrap(),
            layout.offset(row, column).unwrap(),
          )
        }
      }
      assert_eq(factored.row(-1), None)
      assert_eq(factored.row(2147483647), None)
      let last_complete_row = (2147483647L / columns.to_int64()).to_int()
      assert_true(factored.row(last_complete_row) is Some(_))
      assert_eq(factored.row(last_complete_row + 1), None)
      assert_eq(factored.column(1, 0), None)
      assert_eq(factored.column(8 * rotations, 0), None)
      assert_eq(factored.column(0, columns), None)
    }
  }
}
