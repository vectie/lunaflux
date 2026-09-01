///|
priv struct PostprocessReferenceResult {
  rotated_qkv : Array[Float]
  key_cache : Array[Float]
  value_cache : Array[Float]
}

///|
fn postprocess_reference_assert_close(
  actual : ArrayView[Float],
  expected : ArrayView[Float],
  tolerance : Float,
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for index in 0..<actual.length() {
    if (actual[index] - expected[index]).abs() > tolerance {
      fail("reference mismatch at " + index.to_string())
    }
  }
}

///|
fn postprocess_scalar_reference(
  projected_qkv : ArrayView[Float],
  query_norm_weight : ArrayView[Float],
  key_norm_weight : ArrayView[Float],
  positions : ArrayView[Int],
  row_offsets : ArrayView[Int],
  sequence_lengths : ArrayView[Int],
  page_table_offsets : ArrayView[Int],
  page_table_pages : ArrayView[Int],
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  norm_epsilon : Float,
  rope_theta : Float,
  tokens_per_page : Int,
  total_page_count : Int,
  page_stride_values : Int,
) -> PostprocessReferenceResult raise {
  let rows = sequence_lengths.length()
  let tokens = positions.length()
  let packed_heads = query_heads + 2 * kv_heads
  let qk_heads = query_heads + kv_heads
  let qkv_width = packed_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    head_dimension <= 0 ||
    head_dimension % 2 != 0 ||
    rows <= 0 ||
    tokens <= 0 ||
    row_offsets.length() != rows + 1 ||
    page_table_offsets.length() != rows + 1 ||
    row_offsets[0] != 0 ||
    row_offsets[rows] != tokens ||
    page_table_offsets[0] != 0 ||
    page_table_offsets[rows] != page_table_pages.length() ||
    projected_qkv.length() != tokens * qkv_width ||
    query_norm_weight.length() != head_dimension ||
    key_norm_weight.length() != head_dimension ||
    tokens_per_page <= 0 ||
    total_page_count <= 0 ||
    page_stride_values < tokens_per_page * kv_width {
    fail("invalid scalar reference input")
  }
  let rotated = Array::makei(projected_qkv.length(), index => {
    projected_qkv[index]
  })
  let key_cache = Array::make(total_page_count * page_stride_values, -999.0F)
  let value_cache = Array::make(total_page_count * page_stride_values, -999.0F)
  for row in 0..<rows {
    if row_offsets[row] < 0 ||
      row_offsets[row] >= row_offsets[row + 1] ||
      row_offsets[row + 1] > tokens ||
      sequence_lengths[row] <= 0 ||
      page_table_offsets[row] < 0 ||
      page_table_offsets[row] >= page_table_offsets[row + 1] ||
      page_table_offsets[row + 1] > page_table_pages.length() {
      fail("invalid scalar reference descriptor")
    }
    for token in row_offsets[row]..<row_offsets[row + 1] {
      let position = positions[token]
      let logical_page = position / tokens_per_page
      if position < 0 ||
        position >= sequence_lengths[row] ||
        logical_page >= page_table_offsets[row + 1] - page_table_offsets[row] {
        fail("invalid scalar reference position")
      }
      let physical_page = page_table_pages[page_table_offsets[row] +
        logical_page]
      if physical_page < 0 || physical_page >= total_page_count {
        fail("invalid scalar reference page")
      }
      for head in 0..<qk_heads {
        let base = token * qkv_width + head * head_dimension
        let norm_weight = if head < query_heads {
          query_norm_weight
        } else {
          key_norm_weight
        }
        let mut square_sum = 0.0F
        for component in 0..<head_dimension {
          let value = projected_qkv[base + component]
          square_sum += value * value
        }
        let inverse_rms = 1.0F /
          (square_sum / Float::from_int(head_dimension) + norm_epsilon).sqrt()
        let normalized = Array::make(head_dimension, 0.0F)
        for component in 0..<head_dimension {
          normalized[component] = projected_qkv[base + component] *
            inverse_rms *
            norm_weight[component]
        }
        for pair in 0..<(head_dimension / 2) {
          let angle = Float::from_int(position) /
            @math.powf(
              rope_theta,
              Float::from_int(pair * 2) / Float::from_int(head_dimension),
            )
          let first = normalized[pair]
          let second = normalized[pair + head_dimension / 2]
          rotated[base + pair] = first * @math.cosf(angle) -
            second * @math.sinf(angle)
          rotated[base + pair + head_dimension / 2] = second * @math.cosf(angle) +
            first * @math.sinf(angle)
        }
      }
      let cache_base = physical_page * page_stride_values +
        position % tokens_per_page * kv_width
      let key_base = token * qkv_width + query_heads * head_dimension
      let value_base = key_base + kv_width
      for component in 0..<kv_width {
        key_cache[cache_base + component] = rotated[key_base + component]
        value_cache[cache_base + component] = rotated[value_base + component]
      }
    }
  }
  { rotated_qkv: rotated, key_cache, value_cache }
}

///|
test "scalar QKNorm, positioned RoPE, and paged KV write preserve layout" {
  let projected = [
    1.0F,
    2.0F,
    3.0F,
    4.0F,
    -2.0F,
    1.0F,
    0.5F,
    -1.0F,
    5.0F,
    6.0F,
    7.0F,
    8.0F,
  ]
  let result = postprocess_scalar_reference(
    projected,
    [1.0F, 1.0F, 1.0F, 1.0F],
    [0.5F, 1.0F, 1.5F, 2.0F],
    [3],
    [0, 1],
    [4],
    [0, 1],
    [1],
    1,
    1,
    4,
    0.000001F,
    10000.0F,
    4,
    2,
    16,
  )
  let expected_v = [5.0F, 6.0F, 7.0F, 8.0F]
  postprocess_reference_assert_close(result.rotated_qkv[8:12], expected_v, 0.0F)
  postprocess_reference_assert_close(
    result.value_cache[28:32],
    expected_v,
    0.0F,
  )
  postprocess_reference_assert_close(
    result.key_cache[28:32],
    result.rotated_qkv[4:8],
    0.000001F,
  )
  assert_eq(result.key_cache[0], -999.0F)
  assert_eq(result.value_cache[0], -999.0F)
}
