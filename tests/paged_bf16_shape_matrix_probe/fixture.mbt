///|
fn matrix_fill_static_fixture(kind : Int, output : FixedArray[Byte]) -> Bool {
  for index in 0..<output.length() {
    output[index] = b'\x00'
  }
  match kind {
    2 => {
      if output.length() != 128 {
        return false
      }
      for row in 0..<16 {
        for column in 0..<4 {
          matrix_write_bf16(
            output,
            row * 4 + column,
            Float::from_int(row - 8) * 0.125F +
            Float::from_int(column) * 0.0625F,
          )
        }
      }
      true
    }
    3 => {
      if output.length() != 8 {
        return false
      }
      let values : FixedArray[Float] = [1.0F, 0.75F, 1.25F, 0.5F]
      for index, value in values {
        matrix_write_bf16(output, index, value)
      }
      true
    }
    4 => {
      matrix_fill_weight(output, 4, 4, 0.125F, 1)
      output.length() == 32
    }
    5 => {
      matrix_fill_weight(output, 2, 4, 0.25F, 2)
      output.length() == 16
    }
    6 => {
      matrix_fill_weight(output, 2, 4, 0.0625F, 3)
      output.length() == 16
    }
    14 => {
      matrix_fill_weight(output, 4, 4, 0.125F, 7)
      output.length() == 32
    }
    15 => {
      if output.length() != 8 {
        return false
      }
      let values : FixedArray[Float] = [1.125F, 0.875F, 0.75F, 1.25F]
      for index, value in values {
        matrix_write_bf16(output, index, value)
      }
      true
    }
    16 => {
      matrix_fill_weight(output, 8, 4, 0.125F, 4)
      output.length() == 64
    }
    17 => {
      matrix_fill_weight(output, 8, 4, 0.0625F, 5)
      output.length() == 64
    }
    18 => {
      matrix_fill_weight(output, 4, 8, 0.125F, 6)
      output.length() == 64
    }
    19 => {
      if output.length() != 8 {
        return false
      }
      let values : FixedArray[Float] = [0.8125F, 1.1875F, 1.0F, 0.625F]
      for index, value in values {
        matrix_write_bf16(output, index, value)
      }
      true
    }
    20 => {
      matrix_fill_weight(output, 16, 4, 0.125F, 8)
      output.length() == 128
    }
    21 => {
      matrix_poison(output)
      true
    }
    _ => false
  }
}

///|
fn matrix_case_buffer(
  index : Int,
  case : ShapeMatrixCase,
) -> FixedArray[Byte] raise ShapeMatrixFailure {
  let output = FixedArray::make(matrix_buffer_sizes[index], b'\x00')
  match index {
    0 => matrix_fill_i32(output, case.counts)
    1 => matrix_fill_i32(output, case.token_ids)
    10 => matrix_fill_i32(output, case.positions)
    12 => matrix_fill_i32(output, case.row_offsets)
    13 => matrix_fill_i32(output, case.sequence_lengths)
    14 => matrix_fill_i32(output, case.page_offsets)
    15 => matrix_fill_i32(output, case.page_indices)
    17 => matrix_fill_cache(output, key=true)
    18 => matrix_fill_cache(output, key=false)
    _ =>
      require_matrix(
        matrix_fill_static_fixture(matrix_static_kinds[index], output),
        "static fixture \{index} has an invalid size",
      )
  }
  output
}

///|
fn matrix_fixture_bf16(kind : Int, elements : Int) -> FixedArray[Int] {
  let bytes = FixedArray::make(elements * 2, b'\x00')
  try! assert_true(matrix_fill_static_fixture(kind, bytes))
  FixedArray::makei(elements, index => {
    bytes[index * 2].to_int() | (bytes[index * 2 + 1].to_int() << 8)
  })
}

///|
fn matrix_initial_cache(key~ : Bool) -> FixedArray[Int] {
  let bytes = FixedArray::make(1024, b'\x00')
  matrix_fill_cache(bytes, key~)
  FixedArray::makei(512, index => {
    bytes[index * 2].to_int() | (bytes[index * 2 + 1].to_int() << 8)
  })
}
