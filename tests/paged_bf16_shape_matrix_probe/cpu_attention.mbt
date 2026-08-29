///|
fn matrix_cache_index(page : Int, position : Int, component : Int) -> Int {
  page * 128 + position % 2 * 2 + component
}

///|
fn matrix_attention_value(
  case : ShapeMatrixCase,
  qkv : FixedArray[Int],
  cache : FixedArray[Int],
  row : Int,
  key_position : Int,
  component : Int,
  qkv_offset : Int,
) -> Float {
  let row_start = case.row_offsets[row]
  let row_end = case.row_offsets[row + 1]
  let chunk_start = case.positions[row_start]
  if key_position >= chunk_start {
    for token_index in row_start..<row_end {
      if case.positions[token_index] == key_position {
        return matrix_bf16_float(qkv[token_index * 8 + qkv_offset + component])
      }
    }
  }
  let page_slot = case.page_offsets[row] + key_position / 2
  let physical_page = case.page_indices[page_slot]
  matrix_bf16_float(
    cache[matrix_cache_index(physical_page, key_position, component)],
  )
}

///|
fn matrix_attention(
  case : ShapeMatrixCase,
  qkv : FixedArray[Int],
) -> (FixedArray[Int], FixedArray[Int], FixedArray[Int]) {
  let keys = matrix_initial_cache(key=true)
  let values = matrix_initial_cache(key=false)
  for token in 0..<case.live_tokens() {
    let row = case.row_for_token(token)
    let position = case.positions[token]
    let page_slot = case.page_offsets[row] + position / 2
    let physical_page = case.page_indices[page_slot]
    for component in 0..<2 {
      keys[matrix_cache_index(physical_page, position, component)] = qkv[token *
        8 +
        4 +
        component]
      values[matrix_cache_index(physical_page, position, component)] = qkv[token *
        8 +
        6 +
        component]
    }
  }
  let output = matrix_poison_values(12)
  let scale = 1.0F / 2.0F.sqrt()
  for token in 0..<case.live_tokens() {
    let row = case.row_for_token(token)
    let position = case.positions[token]
    for head in 0..<2 {
      let scores = FixedArray::make(position + 1, 0.0F)
      let mut maximum = -3.4028235e38F
      for key_position in 0..<=position {
        let mut dot = 0.0F
        for component in 0..<2 {
          dot = matrix_ordered_add(
            dot,
            matrix_ordered_mul(
              matrix_bf16_float(qkv[token * 8 + head * 2 + component]),
              matrix_attention_value(
                case, qkv, keys, row, key_position, component, 4,
              ),
            ),
          )
        }
        scores[key_position] = dot * scale
        maximum = maximum.max(scores[key_position])
      }
      let mut denominator = 0.0F
      for key_position in 0..<=position {
        denominator += @math.expf(scores[key_position] - maximum)
      }
      for component in 0..<2 {
        let mut accumulator = 0.0F
        for key_position in 0..<=position {
          accumulator += @math.expf(scores[key_position] - maximum) *
            matrix_attention_value(
              case, qkv, values, row, key_position, component, 6,
            ) /
            denominator
        }
        output[token * 4 + head * 2 + component] = matrix_bf16_bits(accumulator)
      }
    }
  }
  (output, keys, values)
}
