///|
fn projection_operand_address_source(
  layout : @physical_tile_ir.OperandLayout,
) -> String {
  let (stride, vector, rotations, row_group) = layout.geometry()
  // The permutation repeats every eight rows; derive its physical stride
  // from the same map that is checked by the producer/consumer tests.
  let rotation = if rotations == 8 && row_group == 1 {
    "r & 7"
  } else {
    "(r / \{row_group}) & \{rotations - 1}"
  }
  (
    $|auto operand_index = [](int row, int column) {
    $|      const unsigned r = (unsigned)row, c = (unsigned)column;
    $|      return r * \{stride} + (c ^ ((\{rotation}) * \{vector}));
    $|    };
  )
}

///|
/// Only result-permuting consumers emit this map. Operand-only users such as
/// sibling products must not acquire an unused result-address declaration.
fn projection_result_address_source() -> String {
  (
    $|auto result_index = [](int row, int column) {
    $|      const unsigned r = (unsigned)row & 7, c = (unsigned)column;
    $|      const unsigned bank = (r * 4) ^ ((r & 1) * 8) ^ (c >> 1) ^ ((c & 1) * 16);
    $|      return ((unsigned)row >> 3) * 128 + ((c >> 3) + (c & 1) * 2) * 32 + bank;
    $|    };
  )
}
