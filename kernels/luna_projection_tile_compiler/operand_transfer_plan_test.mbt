///|
test "operand transfer domains cover both strips without overlap or truncation" {
  for rows in [1, 3, 16, 32, 64] {
    for columns in [1, 7, 16, 128] {
      for width in [8, 16, 64] {
        for vector in [1, 2, 4, 8] {
          for owners in [1, 32, 64, 256] {
            let plan = ProjectionOperandTransferPlan::new(
              input_rows=rows,
              weight_rows=columns,
              row_width=width,
              vector_width=vector,
              owners~,
            ).unwrap()
            assert_eq(plan.weight_offset(), rows * width)
            assert_eq(plan.slot_elements(), (rows + columns) * width)
            for
              (map, count) in [
                (plan.input_map(), rows * width / vector),
                (plan.weight_map(), columns * width / vector),
              ] {
              assert_eq(
                map.geometry(),
                (count, owners, (count - 1) / owners + 1),
              )
              // Invert the striped map for every live vector, including tails.
              for index in 0..<count {
                assert_eq(
                  map.vector_at(index % owners, index / owners),
                  Some(index),
                )
              }
            }
          }
        }
      }
    }
  }
}

///|
test "operand packing rejects invalid and overflowing storage domains" {
  for
    (rows, columns, width, vector, owners) in [
      (0, 1, 8, 8, 32),
      (1, 0, 8, 8, 32),
      (1, 1, 0, 8, 32),
      (1, 1, 8, 0, 32),
      (1, 1, 8, 8, 0),
      (1, 1, 7, 8, 32),
      (1, 1, 12, 8, 32),
      (2147483647, 1, 1, 1, 32),
      (2147483647, 2147483647, 2147483647, 1, 32),
    ] {
    assert_eq(
      ProjectionOperandTransferPlan::new(
        input_rows=rows,
        weight_rows=columns,
        row_width=width,
        vector_width=vector,
        owners~,
      ),
      None,
    )
  }
  let boundary = ProjectionOperandTransferPlan::new(
    input_rows=2147483646,
    weight_rows=1,
    row_width=1,
    vector_width=1,
    owners=32,
  ).unwrap()
  assert_eq(boundary.slot_elements(), 2147483647)
  assert_eq(boundary.weight_offset(), 2147483646)
}
