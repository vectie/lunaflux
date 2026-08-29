///|
fn matrix_ordered_mul(left : Float, right : Float) -> Float {
  left * right
}

///|
fn matrix_ordered_add(left : Float, right : Float) -> Float {
  left + right
}

///|
fn matrix_poison_values(elements : Int) -> FixedArray[Int] {
  FixedArray::make(elements, matrix_bf16_bits(91.0F))
}

///|
fn matrix_embedding(
  case : ShapeMatrixCase,
  table : FixedArray[Int],
) -> FixedArray[Int] {
  let output = matrix_poison_values(12)
  for token in 0..<case.live_tokens() {
    for column in 0..<4 {
      output[token * 4 + column] = table[case.token_ids[token] * 4 + column]
    }
  }
  output
}

///|
fn matrix_project(
  input : FixedArray[Int],
  weight : FixedArray[Int],
  rows : Int,
  inner : Int,
  columns : Int,
  capacity_rows : Int,
) -> FixedArray[Int] {
  let output = matrix_poison_values(capacity_rows * columns)
  for row in 0..<rows {
    for column in 0..<columns {
      let mut accumulator = 0.0F
      for k in 0..<inner {
        accumulator = matrix_ordered_add(
          accumulator,
          matrix_ordered_mul(
            matrix_bf16_float(input[row * inner + k]),
            matrix_bf16_float(weight[column * inner + k]),
          ),
        )
      }
      output[row * columns + column] = matrix_bf16_bits(accumulator)
    }
  }
  output
}

///|
fn matrix_norm(
  input : FixedArray[Int],
  weight : FixedArray[Int],
  rows : Int,
) -> FixedArray[Int] {
  let output = matrix_poison_values(12)
  for row in 0..<rows {
    let mut sum = 0.0F
    for column in 0..<4 {
      let value = matrix_bf16_float(input[row * 4 + column])
      sum = matrix_ordered_add(sum, matrix_ordered_mul(value, value))
    }
    let inverse = 1.0F / (sum / 4.0F + 0.00001F).sqrt()
    for column in 0..<4 {
      output[row * 4 + column] = matrix_bf16_bits(
        matrix_ordered_mul(
          matrix_ordered_mul(
            matrix_bf16_float(input[row * 4 + column]),
            inverse,
          ),
          matrix_bf16_float(weight[column]),
        ),
      )
    }
  }
  output
}

///|
fn matrix_qkv(input : FixedArray[Int], rows : Int) -> FixedArray[Int] {
  let query = matrix_project(input, matrix_fixture_bf16(4, 16), rows, 4, 4, 3)
  let key = matrix_project(input, matrix_fixture_bf16(5, 8), rows, 4, 2, 3)
  let value = matrix_project(input, matrix_fixture_bf16(6, 8), rows, 4, 2, 3)
  let output = matrix_poison_values(24)
  for row in 0..<rows {
    for column in 0..<4 {
      output[row * 8 + column] = query[row * 4 + column]
    }
    for column in 0..<2 {
      output[row * 8 + 4 + column] = key[row * 2 + column]
      output[row * 8 + 6 + column] = value[row * 2 + column]
    }
  }
  output
}

///|
fn matrix_rope(
  case : ShapeMatrixCase,
  qkv : FixedArray[Int],
) -> FixedArray[Int] {
  let output = matrix_poison_values(24)
  for row in 0..<case.live_tokens() {
    let angle = Float::from_int(case.positions[row])
    let cosine = @math.cosf(angle)
    let sine = @math.sinf(angle)
    for pair in 0..<3 {
      let first = matrix_bf16_float(qkv[row * 8 + pair * 2])
      let second = matrix_bf16_float(qkv[row * 8 + pair * 2 + 1])
      output[row * 8 + pair * 2] = matrix_bf16_bits(
        matrix_ordered_add(
          matrix_ordered_mul(first, cosine),
          -matrix_ordered_mul(second, sine),
        ),
      )
      output[row * 8 + pair * 2 + 1] = matrix_bf16_bits(
        matrix_ordered_add(
          matrix_ordered_mul(second, cosine),
          matrix_ordered_mul(first, sine),
        ),
      )
    }
    output[row * 8 + 6] = qkv[row * 8 + 6]
    output[row * 8 + 7] = qkv[row * 8 + 7]
  }
  output
}

///|
fn matrix_residual(
  left : FixedArray[Int],
  right : FixedArray[Int],
  rows : Int,
) -> FixedArray[Int] {
  let output = matrix_poison_values(12)
  for index in 0..<(rows * 4) {
    output[index] = matrix_bf16_bits(
      matrix_ordered_add(
        matrix_bf16_float(left[index]),
        matrix_bf16_float(right[index]),
      ),
    )
  }
  output
}
