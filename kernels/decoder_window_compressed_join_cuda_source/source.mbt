///|
fn valid_window_join_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Emits a serial per-row correctness kernel. `query_modes[row]` is zero for
/// official prefill (`start_pos == 0`) and one for single-token decode
/// (`start_pos > 0`). The compressed dense row must retain its producer's
/// valid prefix and `-1` padding. Output is literal window-first concatenation
/// followed by `-1` padding to the baked maximum width; no sort or dedup occurs.
pub fn deterministic_window_compressed_join_i32_source(
  window_size~ : Int,
  compression_ratio~ : Int,
  maximum_position_count~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderWindowCompressedJoinCudaSource raise DecoderWindowCompressedJoinCudaSourceError {
  if window_size <= 0 {
    raise InvalidWindowSize
  }
  if compression_ratio <= 1 {
    raise InvalidCompressionRatio
  }
  if maximum_position_count <= 0 ||
    maximum_position_count > 2147483647 ||
    window_size > maximum_position_count ||
    compression_ratio > maximum_position_count {
    raise InvalidMaximumPositionCount
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if !valid_window_join_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let maximum_compressed_width = maximum_position_count / compression_ratio
  if window_size > 2147483647 - maximum_compressed_width {
    raise SizeOverflow
  }
  let maximum_joined_width = window_size + maximum_compressed_width
  if maximum_rows.to_int64() >
    9223372036854775807L / maximum_joined_width.to_int64() {
    raise SizeOverflow
  }
  let source = @utf8.encode(
    "#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *query_positions, const int32_t *sequence_lengths, const int32_t *query_modes, const int32_t *compressed_counts, const int32_t *compressed_indices, int32_t *joined_counts, int32_t *joined_indices) {\n" +
    "  if (threadIdx.x != 0) return; const int32_t row = (int32_t)blockIdx.x; if (row >= " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const unsigned long long joined_base = (unsigned long long)row * " +
    maximum_joined_width.to_string() +
    "ULL; for (int32_t slot = 0; slot < " +
    maximum_joined_width.to_string() +
    "; ++slot) joined_indices[joined_base + (unsigned long long)slot] = -1; joined_counts[row] = 0;\n" +
    "  const int32_t live_rows = counts[3]; if (live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    " || row >= live_rows) return;\n" +
    "  const int32_t position = query_positions[row]; const int32_t sequence_length = sequence_lengths[row]; const int32_t mode = query_modes[row];\n" +
    "  if (position < 0 || position >= " +
    maximum_position_count.to_string() +
    " || sequence_length <= 0 || sequence_length > " +
    maximum_position_count.to_string() +
    " || (mode != 0 && mode != 1) || (mode == 0 && position >= sequence_length) || (mode == 1 && (sequence_length != 1 || position == 0))) return;\n" +
    "  const int32_t window_width = mode == 0 ? (sequence_length < " +
    window_size.to_string() +
    " ? sequence_length : " +
    window_size.to_string() +
    ") : " +
    window_size.to_string() +
    "; const int32_t compressed_width = mode == 0 ? sequence_length / " +
    compression_ratio.to_string() +
    " : (position + 1) / " +
    compression_ratio.to_string() +
    "; const int32_t expected_compressed = (position + 1) / " +
    compression_ratio.to_string() +
    "; const int32_t compressed_base_value = mode == 0 ? sequence_length : " +
    window_size.to_string() +
    ";\n" +
    "  if (compressed_counts[row] != expected_compressed || expected_compressed > compressed_width || compressed_width > " +
    maximum_compressed_width.to_string() +
    ") return; const unsigned long long compressed_base = (unsigned long long)row * " +
    maximum_compressed_width.to_string() +
    "ULL;\n" +
    "  for (int32_t slot = 0; slot < compressed_width; ++slot) { const int32_t expected = slot < expected_compressed ? compressed_base_value + slot : -1; if (compressed_indices[compressed_base + (unsigned long long)slot] != expected) return; }\n" +
    "  if (mode == 0) { const int32_t first = position >= " +
    (window_size - 1).to_string() +
    " ? position - " +
    (window_size - 1).to_string() +
    " : 0; for (int32_t rank = 0; rank < window_width; ++rank) { const int32_t index = first + rank; joined_indices[joined_base + (unsigned long long)rank] = index <= position ? index : -1; } } else if (position >= " +
    (window_size - 1).to_string() +
    ") { const int32_t first = (position % " +
    window_size.to_string() +
    " + 1) % " +
    window_size.to_string() +
    "; for (int32_t rank = 0; rank < " +
    window_size.to_string() +
    "; ++rank) joined_indices[joined_base + (unsigned long long)rank] = (first + rank) % " +
    window_size.to_string() +
    "; } else { for (int32_t rank = 0; rank < " +
    window_size.to_string() +
    "; ++rank) joined_indices[joined_base + (unsigned long long)rank] = rank <= position ? rank : -1; }\n" +
    "  for (int32_t slot = 0; slot < compressed_width; ++slot) joined_indices[joined_base + (unsigned long long)window_width + (unsigned long long)slot] = compressed_indices[compressed_base + (unsigned long long)slot]; joined_counts[row] = window_width + compressed_width;\n" +
    "}\n",
  )
  {
    operation: DeterministicWindowCompressedJoinI32V1,
    numerical_contract: BitExactOfficialWindowThenCompressedI32V1,
    window_size,
    compression_ratio,
    maximum_position_count,
    maximum_compressed_width,
    maximum_joined_width,
    maximum_rows,
    function_symbol,
    source,
  }
}
