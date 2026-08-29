///|
fn readonly_test_bf16_bits(value : Float) -> Int {
  let bits = value.reinterpret_as_uint()
  ((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16).reinterpret_as_int()
}

///|
fn readonly_test_bf16(bits : Int) -> Float {
  Float::reinterpret_from_uint(bits.reinterpret_as_uint() << 16)
}

///|
fn readonly_test_value(position : Int, component : Int, phase : Int) -> Int {
  readonly_test_bf16_bits(
    Float::from_int(
      (position * (phase + 3) + component * 5 + phase * 11) % 37 - 18,
    ) *
    0.03125F,
  )
}

///|
fn readonly_test_cache_index(
  physical_page : Int,
  token_in_page : Int,
  component : Int,
  tokens_per_page : Int,
  head_dimension : Int,
) -> Int {
  8 +
  (physical_page * tokens_per_page + token_in_page) * head_dimension +
  component
}

///|
fn readonly_test_fill_history(
  key_cache : FixedArray[Int],
  value_cache : FixedArray[Int],
  sequence_length : Int,
  chunk_start : Int,
  tokens_per_page : Int,
  page_indices : FixedArray[Int],
  head_dimension : Int,
) -> Unit {
  for position in 0..<chunk_start.min(sequence_length) {
    let logical_page = position / tokens_per_page
    let physical_page = page_indices[logical_page]
    for component in 0..<head_dimension {
      let index = readonly_test_cache_index(
        physical_page,
        position % tokens_per_page,
        component,
        tokens_per_page,
        head_dimension,
      )
      key_cache[index] = readonly_test_value(position, component, 1)
      value_cache[index] = readonly_test_value(position, component, 2)
    }
  }
}

///|
fn readonly_test_chunk(
  chunk_start : Int,
  chunk_count : Int,
  query_heads : Int,
  head_dimension : Int,
) -> FixedArray[Int] {
  let query_width = query_heads * head_dimension
  let qkv_width = query_width + 2 * head_dimension
  FixedArray::makei(chunk_count * qkv_width, index => {
    let row = index / qkv_width
    let component = index % qkv_width
    let position = chunk_start + row
    if component < query_width {
      readonly_test_value(position, component, 0)
    } else if component < query_width + head_dimension {
      readonly_test_value(position, component - query_width, 1)
    } else {
      readonly_test_value(position, component - query_width - head_dimension, 2)
    }
  })
}

///|
fn readonly_test_standalone_write(
  qkv : FixedArray[Int],
  key_cache : FixedArray[Int],
  value_cache : FixedArray[Int],
  chunk_start : Int,
  chunk_count : Int,
  query_heads : Int,
  head_dimension : Int,
  tokens_per_page : Int,
  page_indices : FixedArray[Int],
) -> Unit {
  let query_width = query_heads * head_dimension
  let qkv_width = query_width + 2 * head_dimension
  for row in 0..<chunk_count {
    let position = chunk_start + row
    let physical_page = page_indices[position / tokens_per_page]
    for component in 0..<head_dimension {
      let cache_index = readonly_test_cache_index(
        physical_page,
        position % tokens_per_page,
        component,
        tokens_per_page,
        head_dimension,
      )
      key_cache[cache_index] = qkv[row * qkv_width + query_width + component]
      value_cache[cache_index] = qkv[row * qkv_width +
        query_width +
        head_dimension +
        component]
    }
  }
}

///|
fn readonly_test_attention(
  qkv : FixedArray[Int],
  key_cache : FixedArray[Int],
  value_cache : FixedArray[Int],
  chunk_start : Int,
  chunk_count : Int,
  query_heads : Int,
  head_dimension : Int,
  tokens_per_page : Int,
  page_indices : FixedArray[Int],
  fused_current_chunk : Bool,
) -> FixedArray[Int] {
  let query_width = query_heads * head_dimension
  let qkv_width = query_width + 2 * head_dimension
  let output = FixedArray::make(chunk_count * query_width, 0)
  let scores = FixedArray::make(chunk_start + chunk_count, 0.0F)
  let scale = 1.0F / Float::from_int(head_dimension).sqrt()
  for row in 0..<chunk_count {
    let query_position = chunk_start + row
    for query_head in 0..<query_heads {
      let query_offset = row * qkv_width + query_head * head_dimension
      for key_position in 0..<=query_position {
        let mut dot = 0.0F
        for component in 0..<head_dimension {
          let key = if fused_current_chunk && key_position >= chunk_start {
            let local_row = key_position - chunk_start
            readonly_test_bf16(
              qkv[local_row * qkv_width + query_width + component],
            )
          } else {
            let physical_page = page_indices[key_position / tokens_per_page]
            readonly_test_bf16(
              key_cache[readonly_test_cache_index(
                physical_page,
                key_position % tokens_per_page,
                component,
                tokens_per_page,
                head_dimension,
              )],
            )
          }
          dot += readonly_test_bf16(qkv[query_offset + component]) * key
        }
        scores[key_position] = dot * scale
      }
      let mut maximum = scores[0]
      for key_position in 1..<=query_position {
        maximum = maximum.max(scores[key_position])
      }
      let mut denominator = 0.0F
      for key_position in 0..<=query_position {
        scores[key_position] = @math.expf(scores[key_position] - maximum)
        denominator += scores[key_position]
      }
      for component in 0..<head_dimension {
        let mut accumulator = 0.0F
        for key_position in 0..<=query_position {
          let value = if fused_current_chunk && key_position >= chunk_start {
            let local_row = key_position - chunk_start
            readonly_test_bf16(
              qkv[local_row * qkv_width +
              query_width +
              head_dimension +
              component],
            )
          } else {
            let physical_page = page_indices[key_position / tokens_per_page]
            readonly_test_bf16(
              value_cache[readonly_test_cache_index(
                physical_page,
                key_position % tokens_per_page,
                component,
                tokens_per_page,
                head_dimension,
              )],
            )
          }
          accumulator += scores[key_position] * value / denominator
        }
        output[row * query_width + query_head * head_dimension + component] = readonly_test_bf16_bits(
          accumulator,
        )
      }
    }
  }
  output
}

///|
fn readonly_test_case(
  sequence_length : Int,
  chunk_start : Int,
  chunk_count : Int,
  tokens_per_page : Int,
  physical_pages : Int,
) -> Unit raise {
  let query_heads = 2
  let head_dimension = 4
  let logical_pages = (sequence_length + tokens_per_page - 1) / tokens_per_page
  let page_indices = FixedArray::makei(logical_pages, index => {
    (index * 7 + 3) % physical_pages
  })
  let cache_values = physical_pages * tokens_per_page * head_dimension
  let initial_key = FixedArray::make(cache_values + 16, 0x5a5a)
  let initial_value = FixedArray::make(cache_values + 16, 0x6b6b)
  readonly_test_fill_history(
    initial_key, initial_value, sequence_length, chunk_start, tokens_per_page, page_indices,
    head_dimension,
  )
  let qkv = readonly_test_chunk(
    chunk_start, chunk_count, query_heads, head_dimension,
  )
  let baseline_key = FixedArray::makei(initial_key.length(), index => {
    initial_key[index]
  })
  let baseline_value = FixedArray::makei(initial_value.length(), index => {
    initial_value[index]
  })
  readonly_test_standalone_write(
    qkv, baseline_key, baseline_value, chunk_start, chunk_count, query_heads, head_dimension,
    tokens_per_page, page_indices,
  )
  let baseline = readonly_test_attention(
    qkv, initial_key, initial_value, chunk_start, chunk_count, query_heads, head_dimension,
    tokens_per_page, page_indices, true,
  )
  let before_key = FixedArray::makei(baseline_key.length(), index => {
    baseline_key[index]
  })
  let before_value = FixedArray::makei(baseline_value.length(), index => {
    baseline_value[index]
  })
  let observed = readonly_test_attention(
    qkv, baseline_key, baseline_value, chunk_start, chunk_count, query_heads, head_dimension,
    tokens_per_page, page_indices, false,
  )
  assert_eq(observed, baseline)
  assert_eq(baseline_key, before_key)
  assert_eq(baseline_value, before_value)
  for index in 0..<8 {
    assert_eq(baseline_key[index], 0x5a5a)
    assert_eq(baseline_value[index], 0x6b6b)
  }
  for index in (cache_values + 8)..<(cache_values + 16) {
    assert_eq(baseline_key[index], 0x5a5a)
    assert_eq(baseline_value[index], 0x6b6b)
  }
}

///|
test "read-only attention matches fused write-attend scalar and page boundary cases" {
  readonly_test_case(1, 0, 1, 2, 4)
  readonly_test_case(5, 3, 2, 2, 8)
}

///|
test "read-only attention matches long context and leaves cache bytes/canaries unchanged" {
  readonly_test_case(257, 254, 3, 16, 32)
}
