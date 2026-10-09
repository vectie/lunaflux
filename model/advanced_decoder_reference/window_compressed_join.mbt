///|
/// Reproduces the official deterministic-attention index boundary for one
/// query row. Window indices retain their original ring/causal ordering and
/// padding; the compressed row is appended literally without sorting or
/// deduplication. `decode_mode=false` models `start_pos == 0`; `true` models
/// the single-token `start_pos > 0` branch.
pub fn deterministic_window_compressed_index_join_v1(
  window_size~ : Int,
  compression_ratio~ : Int,
  query_position~ : Int,
  sequence_length~ : Int,
  decode_mode~ : Bool,
  compressed_count~ : Int,
  compressed_indices : ArrayView[Int],
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Int] raise AdvancedDecoderReferenceError {
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    query_position < 0 ||
    sequence_length <= 0 ||
    (decode_mode && (sequence_length != 1 || query_position == 0)) ||
    (!decode_mode && query_position >= sequence_length) {
    raise InvalidIndexLayout
  }
  let window_width = if decode_mode {
    window_size
  } else {
    sequence_length.min(window_size)
  }
  let compressed_width = if decode_mode {
    (query_position + 1) / compression_ratio
  } else {
    sequence_length / compression_ratio
  }
  let expected_count = (query_position + 1) / compression_ratio
  if compressed_count != expected_count ||
    compressed_count > compressed_width ||
    compressed_indices.length() != compressed_width ||
    window_width > limits.max_width - compressed_width {
    raise InvalidIndexLayout
  }
  let compressed_base = if decode_mode { window_size } else { sequence_length }
  for slot, index in compressed_indices {
    let expected = if slot < compressed_count {
      compressed_base + slot
    } else {
      -1
    }
    if index != expected {
      raise InvalidIndexLayout
    }
  }
  let joined : Array[Int] = []
  if decode_mode {
    if query_position >= window_size - 1 {
      let first = (query_position % window_size + 1) % window_size
      for rank in 0..<window_size {
        joined.push((first + rank) % window_size)
      }
    } else {
      for rank in 0..<window_size {
        joined.push(if rank <= query_position { rank } else { -1 })
      }
    }
  } else {
    let first = (query_position - window_size + 1).max(0)
    for rank in 0..<window_width {
      let index = first + rank
      joined.push(if index <= query_position { index } else { -1 })
    }
  }
  for index in compressed_indices {
    joined.push(index)
  }
  ReadOnlyArray::from_array(joined)
}
