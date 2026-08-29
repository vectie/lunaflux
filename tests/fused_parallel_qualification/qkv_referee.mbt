///|
fn fused_qualification_dot(
  input : ArrayView[Int],
  weights : ArrayView[Int],
  token : Int,
  column : Int,
) -> Float {
  let mut sum = 0.0F
  for inner in 0..<FUSED_QUALIFICATION_INPUT_WIDTH {
    sum += fused_qualification_bf16_float(
        input[token * FUSED_QUALIFICATION_INPUT_WIDTH + inner],
      ) *
      fused_qualification_bf16_float(
        weights[column * FUSED_QUALIFICATION_INPUT_WIDTH + inner],
      )
  }
  sum
}

///|
fn fused_qualification_projected(
  input : ArrayView[Int],
  q_weight : ArrayView[Int],
  k_weight : ArrayView[Int],
  v_weight : ArrayView[Int],
  token : Int,
  column : Int,
) -> Float {
  let value = if column < FUSED_QUALIFICATION_QUERY_WIDTH {
    fused_qualification_dot(input, q_weight, token, column)
  } else if column <
    FUSED_QUALIFICATION_QUERY_WIDTH + FUSED_QUALIFICATION_KV_WIDTH {
    fused_qualification_dot(
      input,
      k_weight,
      token,
      column - FUSED_QUALIFICATION_QUERY_WIDTH,
    )
  } else {
    fused_qualification_dot(
      input,
      v_weight,
      token,
      column - FUSED_QUALIFICATION_QUERY_WIDTH - FUSED_QUALIFICATION_KV_WIDTH,
    )
  }
  fused_qualification_bf16_float(fused_qualification_bf16_bits(value))
}

///|
fn fused_qualification_rotated(
  input : ArrayView[Int],
  q_weight : ArrayView[Int],
  k_weight : ArrayView[Int],
  v_weight : ArrayView[Int],
  token : Int,
  position : Int,
  column : Int,
) -> Int {
  let projected = fused_qualification_projected(
    input, q_weight, k_weight, v_weight, token, column,
  )
  let qk_width = FUSED_QUALIFICATION_QUERY_WIDTH + FUSED_QUALIFICATION_KV_WIDTH
  if column >= qk_width {
    return fused_qualification_bf16_bits(projected)
  }
  let component = column % FUSED_QUALIFICATION_HEAD_DIMENSION
  let half = FUSED_QUALIFICATION_HEAD_DIMENSION / 2
  let paired_component = if component < half {
    component + half
  } else {
    component - half
  }
  let paired = fused_qualification_projected(
    input,
    q_weight,
    k_weight,
    v_weight,
    token,
    column - component + paired_component,
  )
  let exponent = 2.0F *
    Float::from_int(component % half) /
    Float::from_int(FUSED_QUALIFICATION_HEAD_DIMENSION)
  let angle = Float::from_int(position) / @math.powf(10000.0F, exponent)
  let value = if component < half {
    projected * @math.cosf(angle) - paired * @math.sinf(angle)
  } else {
    paired * @math.sinf(angle) + projected * @math.cosf(angle)
  }
  fused_qualification_bf16_bits(value)
}

///|
fn fused_qualification_qkv_reference(
  shape : FusedQualificationShape,
  input : ArrayView[Int],
  q_weight : ArrayView[Int],
  k_weight : ArrayView[Int],
  v_weight : ArrayView[Int],
) -> (Array[Int], Array[Int], Array[Int]) raise FusedQualificationError {
  if input.length() != shape.token_count * FUSED_QUALIFICATION_INPUT_WIDTH ||
    q_weight.length() !=
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_QUERY_WIDTH ||
    k_weight.length() !=
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_KV_WIDTH ||
    v_weight.length() !=
    FUSED_QUALIFICATION_INPUT_WIDTH * FUSED_QUALIFICATION_KV_WIDTH {
    raise InvalidRefereeInput
  }
  let output = Array::make(
    shape.token_count * FUSED_QUALIFICATION_OUTPUT_WIDTH,
    FUSED_QUALIFICATION_POISON,
  )
  let key_cache = Array::make(
    FUSED_QUALIFICATION_CACHE_ELEMENTS,
    FUSED_QUALIFICATION_POISON,
  )
  let value_cache = Array::make(
    FUSED_QUALIFICATION_CACHE_ELEMENTS,
    FUSED_QUALIFICATION_POISON,
  )
  for token in 0..<shape.token_count {
    let position = shape.positions[token]
    let logical_page = position / FUSED_QUALIFICATION_TOKENS_PER_PAGE
    let physical_page = shape.page_table[logical_page]
    for column in 0..<FUSED_QUALIFICATION_OUTPUT_WIDTH {
      let value = fused_qualification_rotated(
        input, q_weight, k_weight, v_weight, token, position, column,
      )
      output[token * FUSED_QUALIFICATION_OUTPUT_WIDTH + column] = value
      if column >= FUSED_QUALIFICATION_QUERY_WIDTH {
        let is_value = column >=
          FUSED_QUALIFICATION_QUERY_WIDTH + FUSED_QUALIFICATION_KV_WIDTH
        let component = column -
          (if is_value {
            FUSED_QUALIFICATION_QUERY_WIDTH + FUSED_QUALIFICATION_KV_WIDTH
          } else {
            FUSED_QUALIFICATION_QUERY_WIDTH
          })
        let offset = physical_page *
          FUSED_QUALIFICATION_TOKENS_PER_PAGE *
          FUSED_QUALIFICATION_KV_WIDTH +
          position %
          FUSED_QUALIFICATION_TOKENS_PER_PAGE *
          FUSED_QUALIFICATION_KV_WIDTH +
          component
        if is_value {
          value_cache[offset] = value
        } else {
          key_cache[offset] = value
        }
      }
    }
  }
  (output, key_cache, value_cache)
}
