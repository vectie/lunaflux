///|
fn project_bf16(
  input : FixedArray[Int],
  weight : FixedArray[Int],
  rows : Int,
  inner : Int,
  columns : Int,
) -> FixedArray[Int] {
  FixedArray::makei(rows * columns, index => {
    let row = index / columns
    let column = index % columns
    let mut accumulator = 0.0F
    for k in 0..<inner {
      accumulator = ordered_add(
        accumulator,
        ordered_mul(
          bf16_float(input[row * inner + k]),
          bf16_float(weight[column * inner + k]),
        ),
      )
    }
    bf16_bits(accumulator)
  })
}

///|
fn norm_bf16(
  input : FixedArray[Int],
  weight : FixedArray[Int],
) -> FixedArray[Int] {
  let output = FixedArray::make(12, 0)
  for row in 0..<3 {
    let mut sum = 0.0F
    for column in 0..<4 {
      let value = bf16_float(input[row * 4 + column])
      sum = ordered_add(sum, ordered_mul(value, value))
    }
    let inverse = 1.0F / (sum / 4.0F + 0.00001F).sqrt()
    for column in 0..<4 {
      output[row * 4 + column] = bf16_bits(
        ordered_mul(
          ordered_mul(bf16_float(input[row * 4 + column]), inverse),
          bf16_float(weight[column]),
        ),
      )
    }
  }
  output
}

///|
fn cache_index(page : Int, position : Int, component : Int) -> Int {
  page * 128 + position % 2 * 2 + component
}

///|
fn attention_value(
  qkv : FixedArray[Int],
  cache : FixedArray[Int],
  row : Int,
  position : Int,
  component : Int,
  offset : Int,
) -> Float {
  if row == 0 {
    bf16_float(qkv[position * 8 + offset + component])
  } else if position == 2 {
    bf16_float(qkv[2 * 8 + offset + component])
  } else {
    bf16_float(cache[cache_index(1, position, component)])
  }
}

///|
fn attention_bf16(
  qkv : FixedArray[Int],
  keys : FixedArray[Int],
  values : FixedArray[Int],
) -> FixedArray[Int] {
  let pages : FixedArray[Int] = [0, 0, 2]
  for token in 0..<3 {
    for component in 0..<2 {
      keys[cache_index(pages[token], token, component)] = qkv[token * 8 +
        4 +
        component]
      values[cache_index(pages[token], token, component)] = qkv[token * 8 +
        6 +
        component]
    }
  }
  let output = FixedArray::make(12, 0)
  let scale = 1.0F / 2.0F.sqrt()
  for token in 0..<3 {
    let row = if token < 2 { 0 } else { 1 }
    for head in 0..<2 {
      let scores = FixedArray::make(3, 0.0F)
      let mut maximum = -3.4028235e38F
      for key in 0..<=token {
        let mut dot = 0.0F
        for component in 0..<2 {
          dot = ordered_add(
            dot,
            ordered_mul(
              bf16_float(qkv[token * 8 + head * 2 + component]),
              attention_value(qkv, keys, row, key, component, 4),
            ),
          )
        }
        scores[key] = dot * scale
        maximum = maximum.max(scores[key])
      }
      let mut denominator = 0.0F
      for key in 0..<=token {
        denominator += @math.expf(scores[key] - maximum)
      }
      for component in 0..<2 {
        let mut sum = 0.0F
        for key in 0..<=token {
          sum += @math.expf(scores[key] - maximum) *
            attention_value(qkv, values, row, key, component, 6) /
            denominator
        }
        output[token * 4 + head * 2 + component] = bf16_bits(sum)
      }
    }
  }
  output
}

///|
fn compare_output(
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
    let error = (bf16_float(bits) - bf16_float(expected[index])).abs()
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
fn build_graph_reference() -> FixedArray[FixedArray[Int]] {
  let table = fixture_bf16(2, 64)
  let tokens : FixedArray[Int] = [2, 5, 7]
  let embedding = FixedArray::makei(12, index => {
    table[tokens[index / 4] * 4 + index % 4]
  })
  let norm1 = norm_bf16(embedding, fixture_bf16(3, 4))
  let query = project_bf16(norm1, fixture_bf16(4, 16), 3, 4, 4)
  let key = project_bf16(norm1, fixture_bf16(5, 8), 3, 4, 2)
  let value = project_bf16(norm1, fixture_bf16(6, 8), 3, 4, 2)
  let qkv = FixedArray::make(24, 0)
  for row in 0..<3 {
    for column in 0..<4 {
      qkv[row * 8 + column] = query[row * 4 + column]
    }
    for column in 0..<2 {
      qkv[row * 8 + 4 + column] = key[row * 2 + column]
      qkv[row * 8 + 6 + column] = value[row * 2 + column]
    }
  }
  let rope = FixedArray::make(24, 0)
  for row in 0..<3 {
    let angle = Float::from_int(row)
    let cosine = @math.cosf(angle)
    let sine = @math.sinf(angle)
    for pair in 0..<3 {
      let first = bf16_float(qkv[row * 8 + pair * 2])
      let second = bf16_float(qkv[row * 8 + pair * 2 + 1])
      rope[row * 8 + pair * 2] = bf16_bits(
        ordered_add(ordered_mul(first, cosine), -ordered_mul(second, sine)),
      )
      rope[row * 8 + pair * 2 + 1] = bf16_bits(
        ordered_add(ordered_mul(second, cosine), ordered_mul(first, sine)),
      )
    }
    rope[row * 8 + 6] = qkv[row * 8 + 6]
    rope[row * 8 + 7] = qkv[row * 8 + 7]
  }
  let keys = fixture_bf16(12, 512)
  let values = fixture_bf16(13, 512)
  let attention = attention_bf16(rope, keys, values)
  let projected = project_bf16(attention, fixture_bf16(14, 16), 3, 4, 4)
  let residual1 = FixedArray::makei(12, index => {
    bf16_bits(
      ordered_add(bf16_float(embedding[index]), bf16_float(projected[index])),
    )
  })
  let norm2 = norm_bf16(residual1, fixture_bf16(15, 4))
  let gate = fixture_bf16(16, 32)
  let up = fixture_bf16(17, 32)
  let down = fixture_bf16(18, 32)
  let mlp = FixedArray::make(12, 0)
  for row in 0..<3 {
    for column in 0..<4 {
      let mut accumulator = 0.0F
      for middle in 0..<8 {
        let mut gate_value = 0.0F
        let mut up_value = 0.0F
        for inner in 0..<4 {
          let input = bf16_float(norm2[row * 4 + inner])
          gate_value = ordered_add(
            gate_value,
            ordered_mul(input, bf16_float(gate[middle * 4 + inner])),
          )
          up_value = ordered_add(
            up_value,
            ordered_mul(input, bf16_float(up[middle * 4 + inner])),
          )
        }
        let combined = ordered_mul(
          gate_value / ordered_add(1.0F, @math.expf(-gate_value)),
          up_value,
        )
        accumulator = ordered_add(
          accumulator,
          ordered_mul(combined, bf16_float(down[column * 8 + middle])),
        )
      }
      mlp[row * 4 + column] = bf16_bits(accumulator)
    }
  }
  let residual2 = FixedArray::makei(12, index => {
    bf16_bits(ordered_add(bf16_float(residual1[index]), bf16_float(mlp[index])))
  })
  let final_norm = norm_bf16(residual2, fixture_bf16(19, 4))
  let logits = project_bf16(final_norm, fixture_bf16(20, 64), 3, 4, 16)
  [
    embedding, norm1, qkv, rope, attention, keys, values, projected, residual1, norm2,
    mlp, residual2, final_norm, logits,
  ]
}

///|
fn validate_graph_reference(
  outputs : FixedArray[Bytes],
) -> (Int, Double) raise GraphProbeFailure {
  require_graph(outputs.length() == 14, "graph output set is incomplete")
  let expected = build_graph_reference()
  let tolerances : FixedArray[Float] = [
    0.0F, 0.008F, 0.008F, 0.016F, 0.063F, 0.0F, 0.0F, 0.008F, 0.008F, 0.016F, 0.016F,
    0.016F, 0.016F, 0.008F,
  ]
  let maximum_error = Ref(0.0)
  for index in 0..<outputs.length() {
    require_graph(
      compare_output(
        outputs[index],
        expected[index],
        tolerances[index],
        maximum_error,
      ),
      "independent CPU referee mismatch at boundary \{index}",
    )
  }
  let logits = outputs[13]
  let mut selected = 0
  let mut maximum = bf16_float(logits[64].to_int() | (logits[65].to_int() << 8))
  for token in 1..<16 {
    let offset = 64 + token * 2
    let value = bf16_float(
      logits[offset].to_int() | (logits[offset + 1].to_int() << 8),
    )
    if value > maximum {
      maximum = value
      selected = token
    }
  }
  let mut expected_selected = 0
  let mut expected_maximum = bf16_float(expected[13][32])
  for token in 1..<16 {
    let value = bf16_float(expected[13][32 + token])
    if value > expected_maximum {
      expected_maximum = value
      expected_selected = token
    }
  }
  require_graph(
    selected == expected_selected,
    "deterministic host sample mismatch",
  )
  (selected, maximum_error.val)
}
