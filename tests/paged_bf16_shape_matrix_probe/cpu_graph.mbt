///|
fn matrix_mlp(input : FixedArray[Int], rows : Int) -> FixedArray[Int] {
  let gate = matrix_fixture_bf16(16, 32)
  let up = matrix_fixture_bf16(17, 32)
  let down = matrix_fixture_bf16(18, 32)
  let output = matrix_poison_values(12)
  for row in 0..<rows {
    for column in 0..<4 {
      let mut accumulator = 0.0F
      for middle in 0..<8 {
        let mut gate_value = 0.0F
        let mut up_value = 0.0F
        for inner in 0..<4 {
          let value = matrix_bf16_float(input[row * 4 + inner])
          gate_value = matrix_ordered_add(
            gate_value,
            matrix_ordered_mul(
              value,
              matrix_bf16_float(gate[middle * 4 + inner]),
            ),
          )
          up_value = matrix_ordered_add(
            up_value,
            matrix_ordered_mul(value, matrix_bf16_float(up[middle * 4 + inner])),
          )
        }
        let combined = matrix_ordered_mul(
          gate_value / matrix_ordered_add(1.0F, @math.expf(-gate_value)),
          up_value,
        )
        accumulator = matrix_ordered_add(
          accumulator,
          matrix_ordered_mul(
            combined,
            matrix_bf16_float(down[column * 8 + middle]),
          ),
        )
      }
      output[row * 4 + column] = matrix_bf16_bits(accumulator)
    }
  }
  output
}

///|
fn matrix_reference(case : ShapeMatrixCase) -> FixedArray[FixedArray[Int]] {
  let rows = case.live_tokens()
  let embedding = matrix_embedding(case, matrix_fixture_bf16(2, 64))
  let norm1 = matrix_norm(embedding, matrix_fixture_bf16(3, 4), rows)
  let qkv = matrix_qkv(norm1, rows)
  let rope = matrix_rope(case, qkv)
  let (attention, keys, values) = matrix_attention(case, rope)
  let projected = matrix_project(
    attention,
    matrix_fixture_bf16(14, 16),
    rows,
    4,
    4,
    3,
  )
  let residual1 = matrix_residual(embedding, projected, rows)
  let norm2 = matrix_norm(residual1, matrix_fixture_bf16(15, 4), rows)
  let mlp = matrix_mlp(norm2, rows)
  let residual2 = matrix_residual(residual1, mlp, rows)
  let final_norm = matrix_norm(residual2, matrix_fixture_bf16(19, 4), rows)
  let logits = matrix_project(
    final_norm,
    matrix_fixture_bf16(20, 64),
    rows,
    4,
    16,
    3,
  )
  [
    embedding, norm1, qkv, rope, attention, keys, values, projected, residual1, norm2,
    mlp, residual2, final_norm, logits,
  ]
}

///|
fn matrix_compare_output(
  actual : Bytes,
  expected : FixedArray[Int],
  tolerance : Float,
  maximum_error : Ref[Double],
) -> Bool {
  if actual.length() != expected.length() * 2 {
    return false
  }
  for index in 0..<expected.length() {
    let bits = actual[index * 2].to_int() |
      (actual[index * 2 + 1].to_int() << 8)
    let error = (matrix_bf16_float(bits) - matrix_bf16_float(expected[index])).abs()
    if error.is_nan() || error.is_inf() || error > tolerance {
      return false
    }
    if error.to_double() > maximum_error.val {
      maximum_error.val = error.to_double()
    }
  }
  true
}

///|
fn matrix_validate_reference(
  case : ShapeMatrixCase,
  outputs : FixedArray[Bytes],
) -> (Int, Double) raise ShapeMatrixFailure {
  require_matrix(outputs.length() == 14, "matrix output set is incomplete")
  let expected = matrix_reference(case)
  let tolerances : FixedArray[Float] = [
    0.0F, 0.008F, 0.008F, 0.016F, 0.063F, 0.0F, 0.0F, 0.008F, 0.008F, 0.016F, 0.016F,
    0.016F, 0.016F, 0.008F,
  ]
  let maximum_error = Ref(0.0)
  for index in 0..<outputs.length() {
    require_matrix(
      matrix_compare_output(
        outputs[index],
        expected[index],
        tolerances[index],
        maximum_error,
      ),
      "CPU referee mismatch for \{case.name} at boundary \{index}",
    )
  }
  let final_row = case.live_tokens() - 1
  let logits = outputs[13]
  let mut selected = 0
  let first_offset = final_row * 16 * 2
  let mut maximum = matrix_bf16_float(
    logits[first_offset].to_int() | (logits[first_offset + 1].to_int() << 8),
  )
  for token in 1..<16 {
    let offset = (final_row * 16 + token) * 2
    let value = matrix_bf16_float(
      logits[offset].to_int() | (logits[offset + 1].to_int() << 8),
    )
    if value > maximum {
      maximum = value
      selected = token
    }
  }
  let mut expected_selected = 0
  let mut expected_maximum = matrix_bf16_float(expected[13][final_row * 16])
  for token in 1..<16 {
    let value = matrix_bf16_float(expected[13][final_row * 16 + token])
    if value > expected_maximum {
      expected_maximum = value
      expected_selected = token
    }
  }
  require_matrix(
    selected == expected_selected,
    "sample mismatch for \{case.name}",
  )
  (selected, maximum_error.val)
}
