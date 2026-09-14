///|
test "QKV binder rejects concatenated extent overflow before lowering" {
  let operation = projection_fixture_plan().operations()[1]
  let parameters = projection_parameters(operation)
  let original = projection_fixture_contracts().contracts()[1].operands()
  for widths in [[2147483647, 1, 1], [1073741824, 536870912, 536870912]] {
    let operands = original[:].to_owned()
    for index in 0..<3 {
      operands[index + 2] = projection_fixture_operand(
        original[index + 2].role(),
        widths[index].to_int64() * parameters.input_width.to_int64() * 2L,
        16L,
      )
    }
    try projection_bind_qkv_widths(operation, parameters, operands) catch {
      Mismatch(OperandSize) => ()
      _ => fail("unexpected concatenated extent error")
    } noraise {
      _ => fail("overflowing concatenated view was accepted")
    }
  }
}
